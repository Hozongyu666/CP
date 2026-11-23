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
# 1. 核心設定區（從 GitHub Secrets 讀取敏感資訊）
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_TEST = os.environ.get("FORCE_TEST", "false").lower() == "true"

TW_TZ = timezone(timedelta(hours=8))  # 台灣時區 (UTC+8)

TIMEFRAMES = ["15m", "1h", "4h"]     # 同時追蹤的三個時區
TOP_N_COINS = 30                     # 掃描 24H 成交額前 30 大幣種
MIN_SCORE_TO_ALERT = 70              # 綜合評分 >= 70 分才發送新訊號通知
COOLDOWN_HOURS = 4                   # 同一幣種＋時區 4 小時內不重複發新單
MAX_TRACK_HOURS = 72                 # 單筆訊號最長追蹤 72 小時
STATE_FILE = "state.json"            # 保存冷卻紀錄與追蹤中的訊號

# 100U 合約資金管理參數（5x ~ 10x 槓桿）
TOTAL_CAPITAL = 100.0                # 總本金 100U
TARGET_RISK_USDT = 3.5               # 單筆打損目標風險上限：約 3.5U（本金 3.5%）
MIN_MARGIN_USDT = 10.0               # 單筆最低保證金：10U
MAX_MARGIN_USDT = 25.0               # 單筆最高保證金：25U（可同時扛 2~3 單）

BLACKLIST = {"USDC/USDT", "FDUSD/USDT", "TUSD/USDT", "USDP/USDT", "EUR/USDT", "USD1/USDT"}


# ==========================================
# 2. 時間與價格精度格式化工具
# ==========================================
def now_tw_str() -> str:
    return datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")


def format_price(p: float) -> str:
    """根據幣價大小動態調整小數位數，徹底解決 PEPE/SHIB 等低價幣顯示成 0.000004 的問題"""
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
# 3. 狀態讀寫（冷卻紀錄 + 追蹤中訂單）
# ==========================================
def load_state() -> dict:
    default_state = {"cooldowns": {}, "active_signals": {}}
    if not os.path.exists(STATE_FILE):
        return default_state
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "cooldowns" not in data and "active_signals" not in data:
            return {"cooldowns": data, "active_signals": {}}
        data.setdefault("cooldowns", {})
        data.setdefault("active_signals", {})
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


# ==========================================
# 4. 技術指標計算引擎
# ==========================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    df["donchian_high"] = high.rolling(20).max().shift(1)
    df["swing_low"] = low.rolling(10).min().shift(1)

    df["vol_ma20"] = vol.rolling(20).mean().shift(1)
    df["vol_ratio"] = vol / df["vol_ma20"].replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

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
# 5. 0~100 分綜合評分、槓桿與預期損益計算
# ==========================================
def evaluate_setup(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    macro_bullish_4h: bool,
    btc_score: int
) -> dict:
    bar = df.iloc[-2]
    live = df.iloc[-1]
    price = float(live["close"])
    breakdown = {}

    # 1. 趨勢結構 (25分)
    s_trend = 0
    if bar["ema20"] > bar["ema50"] > bar["ema200"] and price > bar["ema20"]:
        s_trend += 15
    elif bar["ema20"] > bar["ema50"] and price > bar["ema20"]:
        s_trend += 8
    if macro_bullish_4h:
        s_trend += 10
    breakdown["趨勢結構"] = f"{s_trend}/25"

    # 2. 動能突破 (25分)
    s_mom = 0
    if bar["close"] > bar["donchian_high"] or price > bar["donchian_high"]:
        s_mom += 15
    elif price > bar["donchian_high"] * 0.992:
        s_mom += 7

    if bar["macd_hist"] > 0 and bar["macd_hist"] > df.iloc[-3]["macd_hist"]:
        s_mom += 10
    elif bar["macd_hist"] > 0:
        s_mom += 5
    breakdown["動能突破"] = f"{s_mom}/25"

    # 3. 量能爆發 (20分)
    vol_r = max(float(bar["vol_ratio"]), float(live["vol_ratio"]))
    if vol_r >= 2.5:
        s_vol = 20
    elif vol_r >= 1.8:
        s_vol = 15
    elif vol_r >= 1.3:
        s_vol = 10
    else:
        s_vol = 0
    breakdown["量能爆發"] = f"{s_vol}/20 ({vol_r:.1f}x)"

    # 4. 趨勢強度與 RSI 位階 (15分)
    s_strength = 0
    adx_val = float(bar["adx"])
    rsi_val = float(live["rsi"])
    if adx_val >= 25:
        s_strength += 8
    elif adx_val >= 20:
        s_strength += 4

    if 55 <= rsi_val <= 72:
        s_strength += 7
    elif 50 <= rsi_val < 55 or 72 < rsi_val <= 78:
        s_strength += 4
    elif rsi_val > 82:
        s_strength -= 5
    s_strength = max(0, s_strength)
    breakdown["強度/RSI"] = f"{s_strength}/15 (ADX:{adx_val:.0f} RSI:{rsi_val:.0f})"

    # 5. BTC 大盤環境 (15分)
    breakdown["BTC大盤"] = f"{btc_score}/15"

    total_score = int(np.clip(s_trend + s_mom + s_vol + s_strength + btc_score, 0, 100))

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

    # --- 動態槓桿建議 (5x ~ 10x) 與 100U 預計損益計算 ---
    # 根據止損距離決定安全槓桿倍數，確保逐倉爆倉線遠在止損線之外
    if sl_pct <= 0.022 and total_score >= 80:
        leverage = 10
    elif sl_pct <= 0.028:
        leverage = 8
    elif sl_pct <= 0.040:
        leverage = 6
    else:
        leverage = 5

    # 根據單筆容忍虧損 (3.5U) 反推保證金，並限制在 10U ~ 25U 之間
    raw_margin = TARGET_RISK_USDT / (sl_pct * leverage)
    margin_usdt = round(float(np.clip(raw_margin, MIN_MARGIN_USDT, MAX_MARGIN_USDT)), 1)
    notional_usdt = margin_usdt * leverage  # 開槓桿後的實際倉位價值

    # 預估合約雙邊手續費約 0.1%（Taker 0.05% x 2）
    est_fee = notional_usdt * 0.001
    est_loss_sl = notional_usdt * sl_pct + est_fee
    est_profit_tp1 = notional_usdt * tp1_pct - est_fee
    est_profit_tp2 = notional_usdt * tp2_pct - est_fee

    # --- 模型預估勝率區間（依綜合評分、大盤環境與趨勢強度推算）---
    if total_score >= 85:
        win_rate_str = "64% ~ 70%"
    elif total_score >= 78:
        win_rate_str = "58% ~ 63%"
    elif total_score >= 70:
        win_rate_str = "52% ~ 57%"
    else:
        win_rate_str = "45% ~ 50%"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "score": total_score,
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
# 6. mplfinance K 線圖表生成（高精度價格標籤）
# ==========================================
def generate_chart_image(df: pd.DataFrame, setup: dict) -> io.BytesIO:
    sub = df.iloc[-60:].copy()
    # 圖表 X 軸時間也轉換為台灣時間 (UTC+8)
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
        mpf.make_addplot(sub["ema200"], color="#a55eea", width=1.3, label="EMA 200"),
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

    # 使用 format_price() 確保低價幣顯示完整 8 位小數
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
# 7. Discord Webhook 推播（新訊號 + TP/SL 到價通知）
# ==========================================
def send_discord_alert(setup: dict, df: pd.DataFrame):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 未設定 DISCORD_WEBHOOK_URL 環境變數，跳過發送")
        return

    score = setup["score"]
    if score >= 85:
        grade = "🔥 極強共振訊號（S級）"
        color = 0x2ECC71
    elif score >= 70:
        grade = "⚡ 強勢突破訊號（A級）"
        color = 0xF1C40F
    else:
        grade = "🧪 手動測試預覽（已納入 TP/SL 追蹤）"
        color = 0x3498DB

    detail_text = " | ".join([f"**{k}**: `{v}`" for k, v in setup["breakdown"].items()])
    tv_symbol = setup["symbol"].replace("/", "")

    embed = {
        "title": f"{grade} | {setup['symbol']} 【{setup['timeframe']} 時區】",
        "description": (
            f"🕒 **出訊號時間（台灣）**：`{setup['signal_time_tw']}`\n"
            f"🎯 **綜合評分**：`{score} / 100 分` ｜ 📈 **預估勝率**：`{setup['win_rate_str']}`\n"
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
                "name": "⚡ 建議槓桿與倉位 (100U本金)",
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
                "name": "🧠 多指標評分拆解",
                "value": detail_text,
                "inline": False,
            },
        ],
        "image": {"url": "attachment://chart.png"},
        "footer": {"text": f"台灣時間 {setup['signal_time_tw']} (UTC+8) • 已開啟自動追蹤 TP/SL"},
    }

    payload = {"username": "量化訊號雷達 (100U戰隊)", "embeds": [embed]}

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


def send_outcome_alert(sig: dict, event_type: str):
    """發送 TP1 / TP2 / SL 觸價回報卡片（附台灣時間與預計U損益）"""
    if not DISCORD_WEBHOOK_URL:
        return

    entry = sig["entry"]
    tv_symbol = sig["symbol"].replace("/", "")
    hit_time_tw = now_tw_str()
    elapsed_min = int((time.time() - sig["created_at"]) / 60)
    lev = sig.get("leverage", 5)

    if event_type == "TP1":
        pct = (sig["tp1"] - entry) / entry * 100
        est_u = sig.get("est_profit_tp1", 0)
        title = f"🎯 止盈 1 達標 (TP1) | {sig['symbol']} 【{sig['timeframe']}】"
        color = 0x1ABC9C
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格已觸及第一止盈目標 **`{format_price(sig['tp1'])}`**（現貨漲幅 `+{pct:.2f}%` ｜ `{lev}x` 槓桿約 **`+{est_u:.2f} U`**）！\n"
            f"💡 **建議操作**：可先平一半鎖利，並將剩餘倉位止損移至進場價 **`{format_price(entry)}`** 保本，系統將繼續追蹤 **TP2**。"
        )
    elif event_type == "TP2":
        pct = (sig["tp2"] - entry) / entry * 100
        est_u = sig.get("est_profit_tp2", 0)
        title = f"🚀 止盈 2 完美達標 (TP2) | {sig['symbol']} 【{sig['timeframe']}】"
        color = 0x2ECC71
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格強勢衝破第二止盈目標 **`{format_price(sig['tp2'])}`**（現貨漲幅 `+{pct:.2f}%` ｜ `{lev}x` 槓桿約 **`+{est_u:.2f} U`**）！\n"
            f"🏁 本筆訊號已全數達標並結束追蹤。"
        )
    else:  # SL
        pct = (sig["sl"] - entry) / entry * 100
        est_u = sig.get("est_loss_sl", 0)
        title = f"🛑 觸及止損出場 (SL) | {sig['symbol']} 【{sig['timeframe']}】"
        color = 0xE74C3C
        tp1_note = "\n*(註：此單先前已觸及過 TP1，若已平半倉推保本則實際為小賺/平手出場)*" if sig.get("tp1_hit") else ""
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格回落觸及止損位 **`{format_price(sig['sl'])}`**（現貨 `{pct:.2f}%` ｜ `{lev}x` 槓桿約 **`-{est_u:.2f} U`**）。{tp1_note}\n"
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
        "footer": {"text": f"原出訊號時間（台灣）：{sig.get('signal_time_tw', '未知')}"},
    }

    payload = {"username": "量化訊號雷達 (戰報追蹤)", "embeds": [embed]}
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            print(f"📣 已發送 [{event_type}] 到價通知: {sig['symbol']} [{sig['timeframe']}]")
    except Exception as e:
        print(f"❌ 發送到價通知失敗: {e}")


# ==========================================
# 8. 追蹤既有訊號是否觸及 TP1 / TP2 / SL
# ==========================================
def check_active_signals(exchange, active_signals: dict):
    if not active_signals:
        return

    print(f"🔎 正在檢查 {len(active_signals)} 筆追蹤中訊號的 TP/SL 狀態...")
    now = time.time()

    for key in list(active_signals.keys()):
        sig = active_signals[key]

        if now - sig["created_at"] > MAX_TRACK_HOURS * 3600:
            print(f"⌛ {key} 追蹤已滿 {MAX_TRACK_HOURS} 小時，自動結束追蹤")
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

                # 1. 保守原則：優先檢查是否觸及止損 (SL)
                if c_low <= sig["sl"]:
                    send_outcome_alert(sig, "SL")
                    del active_signals[key]
                    closed_or_done = True
                    break

                # 2. 檢查是否觸及 TP1
                if not sig.get("tp1_hit", False) and c_high >= sig["tp1"]:
                    sig["tp1_hit"] = True
                    if c_high >= sig["tp2"]:
                        send_outcome_alert(sig, "TP2")
                        del active_signals[key]
                        closed_or_done = True
                        break
                    else:
                        send_outcome_alert(sig, "TP1")

                # 3. 已經過 TP1 後，檢查是否觸及 TP2
                elif sig.get("tp1_hit", False) and c_high >= sig["tp2"]:
                    send_outcome_alert(sig, "TP2")
                    del active_signals[key]
                    closed_or_done = True
                    break

            if not closed_or_done and key in active_signals:
                active_signals[key]["last_checked_ms"] = candles[-1][0]

        except Exception as e:
            print(f"[追蹤檢查異常] {key}: {e}")


# ==========================================
# 9. 單次全市場掃描主流程
# ==========================================
def run_scan():
    exchange = ccxt.okx({"enableRateLimit": True})
    state = load_state()
    cooldowns = state["cooldowns"]
    active_signals = state["active_signals"]

    # 第一步：優先檢查既有訂單有沒有打到 TP1 / TP2 / SL
    check_active_signals(exchange, active_signals)

    # 第二步：掃描全市場前 N 大活躍 USDT 現貨尋找新機會
    tickers = exchange.fetch_tickers()
    usdt_pairs = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") or sym in BLACKLIST or ":" in sym:
            continue
        q_vol = float(t.get("quoteVolume") or 0)
        if q_vol == 0:
            q_vol = float(t.get("baseVolume") or 0) * float(t.get("last") or 0)
        if q_vol > 10_000_000:
            usdt_pairs.append((sym, q_vol))

    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    symbols = [sym for sym, _ in usdt_pairs[:TOP_N_COINS]]

    # 評估 BTC 大盤環境分
    try:
        ohlcv_btc = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=210)
        df_btc = compute_indicators(pd.DataFrame(ohlcv_btc, columns=["ts", "open", "high", "low", "close", "volume"]))
        last_btc = df_btc.iloc[-1]
        pct_1h = (last_btc["close"] - df_btc.iloc[-2]["close"]) / df_btc.iloc[-2]["close"]
        if pct_1h < -0.012:
            btc_score = 0
        elif last_btc["close"] > last_btc["ema50"] and last_btc["ema20"] > last_btc["ema50"]:
            btc_score = 15
        elif last_btc["close"] > last_btc["ema200"]:
            btc_score = 10
        else:
            btc_score = 5
    except Exception:
        btc_score = 8

    all_candidates = []
    now = time.time()

    def register_and_alert(setup: dict, df_tf: pd.DataFrame):
        state_key = f"{setup['symbol']}_{setup['timeframe']}"
        send_discord_alert(setup, df_tf)
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

    for sym in symbols:
        try:
            ohlcv_4h = exchange.fetch_ohlcv(sym, "4h", limit=220)
            if len(ohlcv_4h) < 205:
                continue
            df_4h = compute_indicators(pd.DataFrame(ohlcv_4h, columns=["ts", "open", "high", "low", "close", "volume"]))
            macro_bullish_4h = bool(df_4h.iloc[-1]["close"] > df_4h.iloc[-1]["ema200"])

            for tf in TIMEFRAMES:
                if tf == "4h":
                    df_tf = df_4h
                else:
                    ohlcv_tf = exchange.fetch_ohlcv(sym, tf, limit=220)
                    if len(ohlcv_tf) < 205:
                        continue
                    df_tf = compute_indicators(pd.DataFrame(ohlcv_tf, columns=["ts", "open", "high", "low", "close", "volume"]))

                setup = evaluate_setup(sym, tf, df_tf, macro_bullish_4h, btc_score)
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
    print(f"📊 本輪掃描完成（BTC 環境分: {btc_score}/15 | 追蹤中訂單: {len(active_signals)} 筆），最高分 Top 5：")
    for s, _ in all_candidates[:5]:
        print(f"   • {s['symbol']:<12} [{s['timeframe']:<3}] 評分: {s['score']}分 | 現價: {format_price(s['current_price'])}")

    if FORCE_TEST and all_candidates:
        top_setup, top_df = all_candidates[0]
        print("🧪 偵測到手動測試執行，強制發送最高分標的並納入 TP/SL 追蹤...")
        register_and_alert(top_setup, top_df)

    save_state(state)


if __name__ == "__main__":
    run_scan()
