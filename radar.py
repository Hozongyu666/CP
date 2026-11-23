import io
import json
import os
import time
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf

import ccxt
import numpy as np
import pandas as pd
import requests

# ==========================================
# 1. 核心設定區
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_TEST = os.environ.get("FORCE_TEST", "false").lower() == "true"

TW_TZ = timezone(timedelta(hours=8))  # 台灣時區 (UTC+8)

TIMEFRAMES = ["15m", "1h", "4h"]     # 同時追蹤的三個時區
MIN_SCORE_TO_ALERT = 70              # 綜合評分 >= 70 分才發送新訊號通知
COOLDOWN_HOURS = 4                   # 同一幣種＋時區 4 小時內不重複發新單
MAX_TRACK_HOURS = 72                 # 單筆訊號最長追蹤 72 小時
HEARTBEAT_HOURS = 3                  # 若連續 3 小時無訊號，發送一次存活心跳與潛力榜
STATE_FILE = "state.json"

# 100U 複利資金管理參數（5x ~ 10x 槓桿）
INITIAL_CAPITAL = 100.0              # 初始本金 100U
RISK_PCT_PER_TRADE = 0.035           # 單筆最大打損風險：當前總資金的 3.5%
MIN_MARGIN_USDT = 10.0               # 單筆最低保證金：10U
MAX_MARGIN_PCT = 0.25                # 單筆最高保證金上限：當前總資金的 25%

BLACKLIST = {"USDC/USDT", "FDUSD/USDT", "TUSD/USDT", "USDP/USDT", "EUR/USDT", "USD1/USDT"}


# ==========================================
# 2. 時間與價格精度格式化工具
# ==========================================
def now_tw_str() -> str:
    return datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")


def format_price(p: float) -> str:
    if p == 0:
        return "0"
    abs_p = abs(p)
    if abs_p < 0.00001:
        return f"{p:.8f}"
    elif abs_p < 0.0001:
        return f"{p:.7f}"
    elif abs_p < 0.01:
        return f"{p:.6f}"
    elif abs_p < 1:
        return f"{p:.4f}"
    elif abs_p < 100:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


# ==========================================
# 3. 狀態讀寫（冷卻 + 追蹤訂單 + 持倉量快照 + 實戰戰績）
# ==========================================
def load_state() -> dict:
    default_state = {
        "cooldowns": {},
        "active_signals": {},
        "oi_snapshot": {},
        "last_heartbeat": 0,
        "stats": {"wins": 0, "losses": 0, "net_pnl_u": 0.0},
    }
    if not os.path.exists(STATE_FILE):
        return default_state
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default_state.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return default_state


def save_state(state: dict):
    now = time.time()
    state["cooldowns"] = {
        k: v for k, v in state.get("cooldowns", {}).items() if now - v < 86400
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_stats_footer(state: dict) -> str:
    st = state.get("stats", {"wins": 0, "losses": 0, "net_pnl_u": 0.0})
    total = st["wins"] + st["losses"]
    wr = (st["wins"] / total * 100) if total > 0 else 0.0
    cur_cap = INITIAL_CAPITAL + st["net_pnl_u"]
    return f"戰績: {st['wins']}勝{st['losses']}敗 ({wr:.0f}%) ｜ 模擬資金: {cur_cap:.2f}U ({st['net_pnl_u']:+.2f}U)"


# ==========================================
# 4. 抓全市場永續合約「異常持倉量 (OI)」快照
# ==========================================
def fetch_okx_oi_map() -> dict[str, float]:
    """透過 OKX 公開接口一次抓回全市場永續合約持倉量（免 API Key）"""
    try:
        url = "https://www.okx.com/api/v5/public/open-interest?instType=SWAP"
        resp = requests.get(url, timeout=10).json()
        oi_map = {}
        for item in resp.get("data", []):
            inst_id = item.get("instId", "")
            if inst_id.endswith("-USDT-SWAP"):
                sym = inst_id.replace("-USDT-SWAP", "/USDT")
                oi_map[sym] = float(item.get("oiCcy") or 0)
        return oi_map
    except Exception as e:
        print(f"[OI 抓取略過] {e}")
        return {}


def fetch_funding_rate(symbol: str) -> float | None:
    """抓取單一幣種永續合約資金費率（負費率代表空頭擁擠、易發生軋空暴漲）"""
    try:
        inst_id = symbol.replace("/USDT", "-USDT-SWAP")
        url = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}"
        resp = requests.get(url, timeout=6).json()
        data = resp.get("data", [])
        if data:
            return float(data[0].get("fundingRate", 0)) * 100  # 轉成百分比
    except Exception:
        pass
    return None


# ==========================================
# 5. 多流派技術指標計算引擎
# ==========================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, open_, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # 1. EMA 均線系統
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # 2. 滾動 24 根 VWAP（機構平均持倉成本線）
    typical_price = (high + low + close) / 3.0
    tp_vol = (typical_price * vol).rolling(24).sum()
    vol_sum = vol.rolling(24).sum().replace(0, np.nan)
    df["vwap"] = tp_vol / vol_sum

    # 3. 布林帶與布林帶壓縮度 (Bollinger Band Squeeze)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2.0 * bb_std
    df["bb_lower"] = bb_mid - 2.0 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
    df["bb_width_min30"] = df["bb_width"].rolling(30).min().shift(1)

    # 4. 唐奇安通道 & 結構低點
    df["donchian_high"] = high.rolling(20).max().shift(1)
    df["swing_low"] = low.rolling(10).min().shift(1)

    # 5. 成交量倍數 & 上影線比例（防插針出貨）
    df["vol_ma20"] = vol.rolling(20).mean().shift(1)
    df["vol_ratio"] = vol / df["vol_ma20"].replace(0, np.nan)

    candle_range = (high - low).replace(0, np.nan)
    upper_wick = high - np.maximum(open_, close)
    df["upper_wick_ratio"] = (upper_wick / candle_range).fillna(0)

    # 6. ATR (14)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # 7. RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 8. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 9. ADX (14)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return df


# ==========================================
# 6. 多流派融合評分（含異常持倉、布林壓縮、防插針）
# ==========================================
def evaluate_setup(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    macro_bullish_4h: bool,
    btc_score: int,
    oi_change_pct: float | None,
    current_capital: float
) -> dict:
    bar = df.iloc[-2]
    live = df.iloc[-1]
    price = float(live["close"])
    breakdown = {}
    tags = []

    # --- 維度 1：趨勢與機構成本 VWAP (滿分 20) ---
    s_trend = 0
    if bar["ema20"] > bar["ema50"] and price > bar["vwap"]:
        s_trend += 12
    elif price > bar["vwap"]:
        s_trend += 6
    if macro_bullish_4h:
        s_trend += 8
    breakdown["趨勢/VWAP"] = f"{s_trend}/20"

    # --- 維度 2：布林壓縮爆破 + 唐奇安突破 + MACD (滿分 25) ---
    s_mom = 0
    # 檢查近期是否剛經歷布林帶極致壓縮後突破上軌
    was_squeezed = df["bb_width"].iloc[-4:-1].min() <= (bar["bb_width_min30"] * 1.15)
    bb_break = price > bar["bb_upper"] or bar["close"] > bar["bb_upper"]
    donchian_break = price > bar["donchian_high"] or bar["close"] > bar["donchian_high"]

    if was_squeezed and bb_break:
        s_mom += 18
        tags.append("💥 布林壓縮爆破")
    elif donchian_break or bb_break:
        s_mom += 14
        tags.append("📈 通道突破")
    elif price > bar["donchian_high"] * 0.992:
        s_mom += 7

    if bar["macd_hist"] > 0 and bar["macd_hist"] > df.iloc[-3]["macd_hist"]:
        s_mom += 7
    elif bar["macd_hist"] > 0:
        s_mom += 4
    breakdown["突破/壓縮"] = f"{s_mom}/25"

    # --- 維度 3：主力異常持倉 (OI) + 爆量確認 (滿分 30) ---
    vol_r = max(float(bar["vol_ratio"]), float(live["vol_ratio"]))
    s_vol = 15 if vol_r >= 2.5 else (11 if vol_r >= 1.7 else (7 if vol_r >= 1.3 else 0))

    s_oi = 0
    if oi_change_pct is not None:
        if oi_change_pct >= 5.0:
            s_oi = 15
            tags.append(f"🐋 異常暴增倉 +{oi_change_pct:.1f}%")
        elif oi_change_pct >= 2.0:
            s_oi = 10
            tags.append(f"🐳 主力增倉 +{oi_change_pct:.1f}%")
        elif oi_change_pct >= 0.5:
            s_oi = 5
        oi_str = f"OI:{oi_change_pct:+.1f}%"
    else:
        # 尚未累積到前次 OI 快照時，依量能動能平滑給分
        s_oi = 8 if vol_r >= 2.2 else 4
        oi_str = "OI:暖機中"

    breakdown["量能/持倉"] = f"{s_vol + s_oi}/30 ({vol_r:.1f}x | {oi_str})"

    # --- 維度 4：趨勢強度、RSI 與 BTC 大盤 (滿分 25) ---
    s_env = 0
    adx_val = float(bar["adx"])
    rsi_val = float(live["rsi"])
    if adx_val >= 24:
        s_env += 6
    elif adx_val >= 19:
        s_env += 3

    if 55 <= rsi_val <= 75:
        s_env += 6
    elif 50 <= rsi_val < 55 or 75 < rsi_val <= 80:
        s_env += 3

    s_env += min(13, btc_score)
    breakdown["強度/大盤"] = f"{s_env}/25 (ADX:{adx_val:.0f} RSI:{rsi_val:.0f})"

    # --- 防插針誘多懲罰（上影線過長直接扣分） ---
    wick_ratio = max(float(bar["upper_wick_ratio"]), float(live["upper_wick_ratio"]))
    penalty = 0
    if wick_ratio >= 0.50:
        penalty = 15
        tags.append("⚠️ 長上影線扣分")
    elif wick_ratio >= 0.38:
        penalty = 7

    total_score = int(np.clip(s_trend + s_mom + s_vol + s_env - penalty, 0, 100))

    # --- 計算 Entry / SL / TP1 / TP2 ---
    atr = float(bar["atr"])
    atr_sl = price - 1.8 * atr
    struct_sl = float(bar["swing_low"]) * 0.997
    sl = max(atr_sl, struct_sl)
    sl_pct = (price - sl) / price
    if sl_pct < 0.012:
        sl = price * (1 - 0.012)
        sl_pct = 0.012
    elif sl_pct > 0.060:
        sl = price * (1 - 0.060)
        sl_pct = 0.060

    risk_per_unit = price - sl
    tp1 = price + 1.5 * risk_per_unit
    tp2 = price + 3.0 * risk_per_unit
    tp1_pct = (tp1 - price) / price
    tp2_pct = (tp2 - price) / price

    # --- 動態槓桿 (5x ~ 10x) 與複利倉位計算 ---
    if sl_pct <= 0.022 and total_score >= 80:
        leverage = 10
    elif sl_pct <= 0.028:
        leverage = 8
    elif sl_pct <= 0.040:
        leverage = 6
    else:
        leverage = 5

    target_risk_u = max(2.0, current_capital * RISK_PCT_PER_TRADE)
    max_margin_u = max(15.0, current_capital * MAX_MARGIN_PCT)
    raw_margin = target_risk_u / (sl_pct * leverage)
    margin_usdt = round(float(np.clip(raw_margin, MIN_MARGIN_USDT, max_margin_u)), 1)
    notional_usdt = margin_usdt * leverage

    est_fee = notional_usdt * 0.001
    est_loss_sl = notional_usdt * sl_pct + est_fee
    est_profit_tp1 = notional_usdt * tp1_pct - est_fee
    est_profit_tp2 = notional_usdt * tp2_pct - est_fee

    if total_score >= 85:
        win_rate_str = "65% ~ 72%"
    elif total_score >= 78:
        win_rate_str = "58% ~ 64%"
    elif total_score >= 70:
        win_rate_str = "53% ~ 57%"
    else:
        win_rate_str = "45% ~ 50%"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "score": total_score,
        "tags": tags,
        "signal_time_tw": now_tw_str(),
        "current_price": price,
        "entry": price,
        "sl": sl,
        "sl_pct": sl_pct * 100,
        "tp1": tp1,
        "tp1_pct": tp1_pct * 100,
        "tp2": tp2,
        "tp2_pct": tp2_pct * 100,
        "leverage": leverage,
        "margin_usdt": margin_usdt,
        "notional_usdt": notional_usdt,
        "est_loss_sl": est_loss_sl,
        "est_profit_tp1": est_profit_tp1,
        "est_profit_tp2": est_profit_tp2,
        "rr_tp1": 1.5,
        "rr_tp2": 3.0,
        "win_rate_str": win_rate_str,
        "breakdown": breakdown,
    }


# ==========================================
# 7. mplfinance K 線圖表生成（含 VWAP 機構成本線）
# ==========================================
def generate_chart_image(df: pd.DataFrame, setup: dict) -> io.BytesIO:
    sub = df.iloc[-60:].copy()
    sub.index = pd.to_datetime(sub["ts"], unit="ms") + pd.Timedelta(hours=8)
    sub = sub.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    mc = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume={"up": "#26a69a", "down": "#ef5350"},
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#1e1f22",
        figcolor="#1e1f22",
        gridcolor="#2b2d31",
        gridstyle="--",
        y_on_right=True,
        rc={
            "axes.labelcolor": "#b5bac1",
            "xtick.color": "#b5bac1",
            "ytick.color": "#b5bac1",
        },
    )

    apds = [
        mpf.make_addplot(sub["ema20"], color="#f1c40f", width=1.2, label="EMA 20"),
        mpf.make_addplot(sub["ema50"], color="#e67e22", width=1.2, label="EMA 50"),
        mpf.make_addplot(sub["vwap"], color="#00d2d3", linestyle="--", width=1.2, label="VWAP (主力成本)"),
    ]

    levels = [setup["tp2"], setup["tp1"], setup["entry"], setup["sl"]]
    level_colors = ["#00e676", "#2ecc71", "#3498db", "#ff5252"]
    level_styles = ["--", "--", "-", "--"]
    level_widths = [1.2, 1.2, 1.5, 1.5]

    fig, axes = mpf.plot(
        sub,
        type="candle",
        style=style,
        volume=True,
        addplot=apds,
        hlines=dict(
            hlines=levels,
            colors=level_colors,
            linestyle=level_styles,
            linewidths=level_widths,
        ),
        title=f"\n{setup['symbol']} [{setup['timeframe']}] | Score: {setup['score']}/100 | {setup['leverage']}x | TW {setup['signal_time_tw']}",
        ylabel="Price (USDT)",
        ylabel_lower="Volume",
        figsize=(11, 6.2),
        panel_ratios=(3.2, 1),
        tight_layout=True,
        returnfig=True,
    )

    ax_main = axes[0]
    ax_main.legend(
        loc="upper left",
        facecolor="#2b2d31",
        edgecolor="#3f4147",
        labelcolor="white",
        fontsize=8.5,
    )

    tags = [
        (setup["tp2"], f" TP2 {format_price(setup['tp2'])} ", "#00c853", "black"),
        (setup["tp1"], f" TP1 {format_price(setup['tp1'])} ", "#2ecc71", "black"),
        (setup["entry"], f" ENTRY {format_price(setup['entry'])} ", "#3498db", "white"),
        (setup["sl"], f" SL {format_price(setup['sl'])} ", "#ff5252", "white"),
    ]
    x_pos = len(sub) - 0.5
    for y_val, text, bg_color, fg_color in tags:
        ax_main.text(
            x_pos,
            y_val,
            text,
            color=fg_color,
            fontsize=8,
            fontweight="bold",
            va="center",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, edgecolor="none", alpha=0.92),
        )

    y_min = min(sub["Low"].min(), setup["sl"])
    y_max = max(sub["High"].max(), setup["tp2"])
    pad = (y_max - y_min) * 0.08
    ax_main.set_ylim(y_min - pad, y_max + pad)

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
    finally:
        plt.close(fig)

    return buf


# ==========================================
# 8. Discord Webhook 推播（新訊號 + 戰報 + 存活心跳）
# ==========================================
def send_discord_alert(setup: dict, df: pd.DataFrame, state: dict):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 未設定 DISCORD_WEBHOOK_URL 環境變數，跳過發送")
        return

    # 補充抓取資金費率，若為負費率代表軋空潛力極高
    fr = fetch_funding_rate(setup["symbol"])
    fr_str = f"`{fr:+.4f}%`" if fr is not None else "`無合約`"
    if fr is not None and fr <= -0.01:
        setup["tags"].append(f"🔥 負費率軋空 ({fr:+.3f}%)")

    score = setup["score"]
    if score >= 85:
        grade = "🔥 極強共振訊號（S級）"
        color = 0x2ECC71
    elif score >= 70:
        grade = "⚡ 強勢突破訊號（A級）"
        color = 0xF1C40F
    else:
        grade = "🧪 手動測試預覽（已納入追蹤）"
        color = 0x3498DB

    tag_line = " ｜ ".join(setup["tags"]) if setup["tags"] else "標準趨勢突破"
    detail_text = " | ".join([f"**{k}**: `{v}`" for k, v in setup["breakdown"].items()])
    tv_symbol = setup["symbol"].replace("/", "")

    embed = {
        "title": f"{grade} | {setup['symbol']} 【{setup['timeframe']} 時區】",
        "description": (
            f"🏷️ **訊號特徵**：**{tag_line}**\n"
            f"🕒 **出訊號時間（台灣）**：`{setup['signal_time_tw']}`\n"
            f"🎯 **綜合評分**：`{score} / 100 分` ｜ 📈 **預估勝率**：`{setup['win_rate_str']}` ｜ 💸 **資金費率**：{fr_str}\n"
            f"⚖️ **盈虧比 (R:R)**：TP1 `1 : {setup['rr_tp1']}` ／ TP2 `1 : {setup['rr_tp2']}`\n"
            f"📊 [點此開啟 TradingView 圖表確認](https://www.tradingview.com/chart/?symbol=OKX:{tv_symbol})"
        ),
        "color": color,
        "fields": [
            {
                "name": "💲 現在價格 / 建議進場",
                "value": f"現價: `{format_price(setup['current_price'])}`\n進場: `{format_price(setup['entry'])}`",
                "inline": True,
            },
            {
                "name": "⚡ 建議槓桿與複利倉位",
                "value": f"槓桿: **`{setup['leverage']}x`** (逐倉)\n保證金: `{setup['margin_usdt']:.1f} U` (倉位 `{setup['notional_usdt']:.0f} U`)",
                "inline": True,
            },
            {
                "name": f"🛑 止損價 (SL -{setup['sl_pct']:.2f}%)",
                "value": f"`{format_price(setup['sl'])}`\n預計虧損: **`-{setup['est_loss_sl']:.2f} U`**",
                "inline": True,
            },
            {
                "name": f"🎯 止盈 1 (TP1 +{setup['tp1_pct']:.2f}%)",
                "value": f"`{format_price(setup['tp1'])}`\n預計全平賺: **`+{setup['est_profit_tp1']:.2f} U`**",
                "inline": True,
            },
            {
                "name": f"🚀 止盈 2 (TP2 +{setup['tp2_pct']:.2f}%)",
                "value": f"`{format_price(setup['tp2'])}`\n預計全平賺: **`+{setup['est_profit_tp2']:.2f} U`**",
                "inline": True,
            },
            {
                "name": "🧠 多流派評分拆解",
                "value": detail_text,
                "inline": False,
            },
        ],
        "image": {"url": "attachment://chart.png"},
        "footer": {"text": f"{get_stats_footer(state)} • 已開啟自動追蹤 TP/SL"},
    }

    payload = {"username": "多流派翻倉雷達 (OKX版)", "embeds": [embed]}

    try:
        chart_buf = generate_chart_image(df, setup)
        files = {"file": ("chart.png", chart_buf, "image/png")}
        data = {"payload_json": json.dumps(payload)}
        resp = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files, timeout=15)
        if resp.status_code in (200, 204):
            print(f"✅ 已發送新訊號圖文: {setup['symbol']} [{setup['timeframe']}] ({score}分)")
        else:
            print(f"❌ Discord 發送失敗 ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"❌ Discord 連線或繪圖異常: {e}")


def send_outcome_alert(sig: dict, event_type: str, state: dict):
    """發送 TP1 / TP2 / SL 觸價回報並更新累計實戰戰績"""
    if not DISCORD_WEBHOOK_URL:
        return

    entry = sig["entry"]
    tv_symbol = sig["symbol"].replace("/", "")
    hit_time_tw = now_tw_str()
    elapsed_min = int((time.time() - sig["created_at"]) / 60)
    lev = sig.get("leverage", 5)
    stats = state["stats"]

    if event_type == "TP1":
        pct = (sig["tp1"] - entry) / entry * 100
        est_u = sig.get("est_profit_tp1", 0)
        title = f"🎯 止盈 1 達標 (TP1) | {sig['symbol']} 【{sig['timeframe']}】"
        color = 0x1ABC9C
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格已觸及第一止盈目標 **`{format_price(sig['tp1'])}`**（現貨 `+{pct:.2f}%` ｜ `{lev}x` 約 **`+{est_u:.2f} U`**）！\n"
            f"💡 **建議操作**：先平半倉鎖利，剩餘倉位止損移至進場價 **`{format_price(entry)}`** 保本，繼續追蹤 **TP2**。"
        )
    elif event_type == "TP2":
        pct = (sig["tp2"] - entry) / entry * 100
        est_u = sig.get("est_profit_tp2", 0)
        stats["wins"] += 1
        stats["net_pnl_u"] = round(stats["net_pnl_u"] + est_u, 2)
        title = f"🚀 止盈 2 完美達標 (TP2) | {sig['symbol']} 【{sig['timeframe']}】"
        color = 0x2ECC71
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格強勢衝破第二止盈目標 **`{format_price(sig['tp2'])}`**（現貨 `+{pct:.2f}%` ｜ `{lev}x` 約 **`+{est_u:.2f} U`**）！\n"
            f"🏁 本筆訊號全數達標並結束追蹤。"
        )
    else:  # SL
        pct = (sig["sl"] - entry) / entry * 100
        est_u = sig.get("est_loss_sl", 0)
        if sig.get("tp1_hit"):
            # 若已過 TP1 並推保本，記為小賺保本出場
            half_profit = round(sig.get("est_profit_tp1", 0) * 0.5, 2)
            stats["wins"] += 1
            stats["net_pnl_u"] = round(stats["net_pnl_u"] + half_profit, 2)
            tp1_note = f"\n*(註：此單先前已過 TP1，按紀律平半倉推保本實賺約 `+{half_profit:.2f} U`)*"
        else:
            stats["losses"] += 1
            stats["net_pnl_u"] = round(stats["net_pnl_u"] - est_u, 2)
            tp1_note = ""

        title = f"🛑 觸及止損出場 (SL) | {sig['symbol']} 【{sig['timeframe']}】"
        color = 0xE74C3C
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格回落觸及止損位 **`{format_price(sig['sl'])}`**（現貨 `{pct:.2f}%` ｜ `{lev}x` 約 **`-{est_u:.2f} U`**）。{tp1_note}\n"
            f"🛡️ 嚴格執行風控，本筆訊號已結束追蹤。"
        )

    embed = {
        "title": title,
        "description": f"{desc}\n📊 [開啟 TradingView 查看走勢](https://www.tradingview.com/chart/?symbol=OKX:{tv_symbol})",
        "color": color,
        "fields": [
            {"name": "📍 原進場價 (Entry)", "value": f"`{format_price(entry)}`", "inline": True},
            {"name": "🎯 TP1 / 🚀 TP2", "value": f"`{format_price(sig['tp1'])}` / `{format_price(sig['tp2'])}`", "inline": True},
            {"name": "🛑 原止損價 (SL)", "value": f"`{format_price(sig['sl'])}`", "inline": True},
        ],
        "footer": {"text": f"{get_stats_footer(state)} • 原出訊號：{sig.get('signal_time_tw', '')}"},
    }

    payload = {"username": "多流派翻倉雷達 (戰報追蹤)", "embeds": [embed]}
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            print(f"📣 已發送 [{event_type}] 到價通知: {sig['symbol']} [{sig['timeframe']}]")
    except Exception as e:
        print(f"❌ 發送到價通知失敗: {e}")


def send_heartbeat(top_candidates: list, btc_score: int, active_count: int, state: dict):
    """每 3 小時無訊號時發送一次存活報平安與潛力觀察名單"""
    if not DISCORD_WEBHOOK_URL or not top_candidates:
        return

    lines = []
    for s, _ in top_candidates[:3]:
        tag_str = f" ({s['tags'][0]})" if s["tags"] else ""
        lines.append(f"• **{s['symbol']}** `[{s['timeframe']}]`：**{s['score']}分**{tag_str} ｜ 現價 `{format_price(s['current_price'])}`")

    embed = {
        "title": "💓 雲端雷達定時巡檢報平安（目前盤整蓄勢中）",
        "description": (
            f"🕒 **台灣時間**：`{now_tw_str()}`\n"
            f"目前全市場暫無超過 `{MIN_SCORE_TO_ALERT} 分` 的突破單（BTC 大盤分：`{btc_score}/15` ｜ 追蹤中訂單：`{active_count}` 筆）。\n\n"
            f"👀 **最接近啟動的前 3 名觀察名單**：\n" + "\n".join(lines)
        ),
        "color": 0x95A5A6,
        "footer": {"text": f"{get_stats_footer(state)} • 系統 24H 正常運行中"},
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "多流派翻倉雷達 (巡檢)", "embeds": [embed]}, timeout=10)
        print("💓 已發送 3 小時存活心跳通知")
    except Exception as e:
        print(f"[心跳發送失敗] {e}")


# ==========================================
# 9. 追蹤既有訊號是否觸及 TP1 / TP2 / SL
# ==========================================
def check_active_signals(exchange, state: dict):
    active_signals = state["active_signals"]
    if not active_signals:
        return

    print(f"🔎 正在檢查 {len(active_signals)} 筆追蹤中訊號的 TP/SL 狀態...")
    now = time.time()

    for key in list(active_signals.keys()):
        sig = active_signals[key]
        if now - sig["created_at"] > MAX_TRACK_HOURS * 3600:
            del active_signals[key]
            continue

        try:
            since_ms = int(sig.get("last_checked_ms", sig["created_at"] * 1000))
            candles = exchange.fetch_ohlcv(sig["symbol"], "5m", since=since_ms, limit=200)
            if not candles:
                continue

            closed_or_done = False
            for c in candles:
                c_ts, c_open, c_high, c_low, c_close, _ = c

                if c_low <= sig["sl"]:
                    send_outcome_alert(sig, "SL", state)
                    del active_signals[key]
                    closed_or_done = True
                    break

                if not sig.get("tp1_hit", False) and c_high >= sig["tp1"]:
                    sig["tp1_hit"] = True
                    if c_high >= sig["tp2"]:
                        send_outcome_alert(sig, "TP2", state)
                        del active_signals[key]
                        closed_or_done = True
                        break
                    else:
                        send_outcome_alert(sig, "TP1", state)

                elif sig.get("tp1_hit", False) and c_high >= sig["tp2"]:
                    send_outcome_alert(sig, "TP2", state)
                    del active_signals[key]
                    closed_or_done = True
                    break

            if not closed_or_done and key in active_signals:
                active_signals[key]["last_checked_ms"] = candles[-1][0]

        except Exception as e:
            print(f"[追蹤檢查異常] {key}: {e}")


# ==========================================
# 10. 單次全市場掃描主流程
# ==========================================
def run_scan():
    exchange = ccxt.okx({"enableRateLimit": True})
    state = load_state()
    cooldowns = state["cooldowns"]
    active_signals = state["active_signals"]
    oi_snapshot = state["oi_snapshot"]
    current_capital = INITIAL_CAPITAL + state["stats"].get("net_pnl_u", 0.0)

    # 1. 優先檢查既有訂單有沒有打到 TP1 / TP2 / SL
    check_active_signals(exchange, state)

    # 2. 一次抓取全市場永續合約持倉量 (OI)，比對異常增倉
    current_oi_map = fetch_okx_oi_map()
    oi_change_map = {}
    for sym, cur_oi in current_oi_map.items():
        prev_rec = oi_snapshot.get(sym)
        if prev_rec and prev_rec.get("oi", 0) > 0:
            oi_change_map[sym] = (cur_oi - prev_rec["oi"]) / prev_rec["oi"] * 100
        # 更新最新持倉量快照
        oi_snapshot[sym] = {"oi": cur_oi, "ts": time.time()}

    # 3. 組合掃描名單：「前 25 大成交額主流幣」＋「24H 漲幅榜前 25 名異動山寨幣」
    tickers = exchange.fetch_tickers()
    active_usdt = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") or sym in BLACKLIST or ":" in sym:
            continue
        q_vol = float(t.get("quoteVolume") or 0)
        if q_vol == 0:
            q_vol = float(t.get("baseVolume") or 0) * float(t.get("last") or 0)
        pct_24h = float(t.get("percentage") or 0)
        if q_vol > 3_000_000:  # 至少 300 萬 U 成交額，過濾死幣
            active_usdt.append((sym, q_vol, pct_24h))

    top_vol = [x[0] for x in sorted(active_usdt, key=lambda x: x[1], reverse=True)[:25]]
    top_gainers = [x[0] for x in sorted(active_usdt, key=lambda x: x[2], reverse=True)[:25]]
    symbols = list(dict.fromkeys(top_vol + top_gainers))  # 去重合併約 40~45 檔

    # 4. 評估 BTC 大盤環境分
    try:
        ohlcv_btc = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=210)
        df_btc = compute_indicators(pd.DataFrame(ohlcv_btc, columns=["ts", "open", "high", "low", "close", "volume"]))
        last_btc = df_btc.iloc[-1]
        pct_1h = (last_btc["close"] - df_btc.iloc[-2]["close"]) / df_btc.iloc[-2]["close"]
        if pct_1h < -0.012:
            btc_score = 0
        elif last_btc["close"] > last_btc["ema50"] and last_btc["ema20"] > last_btc["ema50"]:
            btc_score = 13
        elif last_btc["close"] > last_btc["ema200"]:
            btc_score = 9
        else:
            btc_score = 5
    except Exception:
        btc_score = 7

    all_candidates = []
    now = time.time()
    sent_any_this_round = False

    def register_and_alert(setup: dict, df_tf: pd.DataFrame):
        nonlocal sent_any_this_round
        state_key = f"{setup['symbol']}_{setup['timeframe']}"
        send_discord_alert(setup, df_tf, state)
        cooldowns[state_key] = now
        active_signals[state_key] = {
            "symbol": setup["symbol"],
            "timeframe": setup["timeframe"],
            "score": setup["score"],
            "signal_time_tw": setup["signal_time_tw"],
            "entry": setup["entry"],
            "sl": setup["sl"],
            "tp1": setup["tp1"],
            "tp2": setup["tp2"],
            "leverage": setup["leverage"],
            "est_loss_sl": setup["est_loss_sl"],
            "est_profit_tp1": setup["est_profit_tp1"],
            "est_profit_tp2": setup["est_profit_tp2"],
            "tp1_hit": False,
            "created_at": now,
            "last_checked_ms": int(now * 1000),
        }
        sent_any_this_round = True

    for sym in symbols:
        try:
            ohlcv_4h = exchange.fetch_ohlcv(sym, "4h", limit=220)
            if len(ohlcv_4h) < 205:
                continue
            df_4h = compute_indicators(pd.DataFrame(ohlcv_4h, columns=["ts", "open", "high", "low", "close", "volume"]))
            macro_bullish_4h = bool(df_4h.iloc[-1]["close"] > df_4h.iloc[-1]["ema200"])
            oi_chg = oi_change_map.get(sym)

            for tf in TIMEFRAMES:
                if tf == "4h":
                    df_tf = df_4h
                else:
                    ohlcv_tf = exchange.fetch_ohlcv(sym, tf, limit=220)
                    if len(ohlcv_tf) < 205:
                        continue
                    df_tf = compute_indicators(pd.DataFrame(ohlcv_tf, columns=["ts", "open", "high", "low", "close", "volume"]))

                setup = evaluate_setup(sym, tf, df_tf, macro_bullish_4h, btc_score, oi_chg, current_capital)
                all_candidates.append((setup, df_tf))

                state_key = f"{sym}_{tf}"
                last_sent = cooldowns.get(state_key, 0)
                if (
                    setup["score"] >= MIN_SCORE_TO_ALERT
                    and (now - last_sent > COOLDOWN_HOURS * 3600)
                    and state_key not in active_signals
                ):
                    register_and_alert(setup, df_tf)

        except Exception as e:
            print(f"[跳過] {sym}: {e}")

    all_candidates.sort(key=lambda x: x[0]["score"], reverse=True)
    print(f"📊 [{now_tw_str()}] 掃描 {len(symbols)} 檔完成（BTC分: {btc_score}/13 | 追蹤中: {len(active_signals)} 筆），最高分 Top 5：")
    for s, _ in all_candidates[:5]:
        print(f"   • {s['symbol']:<12} [{s['timeframe']:<3}] 評分: {s['score']}分 | 現價: {format_price(s['current_price'])}")

    if FORCE_TEST and all_candidates:
        top_setup, top_df = all_candidates[0]
        print("🧪 偵測到手動測試執行，強制發送最高分標的並納入 TP/SL 追蹤...")
        register_and_alert(top_setup, top_df)

    # 若本輪有發送新單，重置心跳計時；否則超過 3 小時沒訊號就發一次存活心跳
    if sent_any_this_round:
        state["last_heartbeat"] = now
    elif now - state.get("last_heartbeat", 0) >= HEARTBEAT_HOURS * 3600:
        send_heartbeat(all_candidates, btc_score, len(active_signals), state)
        state["last_heartbeat"] = now

    save_state(state)


if __name__ == "__main__":
    run_scan()
