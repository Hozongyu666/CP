import io
import json
import os
import time
from datetime import datetime

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

TIMEFRAMES = ["15m", "1h", "4h"]     # 同時追蹤的三個時區
TOP_N_COINS = 30                     # 掃描 24H 成交額前 30 大幣種
MIN_SCORE_TO_ALERT = 75              # 綜合評分 >= 75 分才發送通知
COOLDOWN_HOURS = 4                   # 同一幣種＋時區 4 小時內不重複通知
STATE_FILE = "state.json"            # 用於跨次執行保存冷卻紀錄

# 100U 資金管理參數
TOTAL_CAPITAL = 100.0
RISK_PER_TRADE_USDT = 3.0            # 單筆最大止損容忍：3U（3%）
MAX_SINGLE_POS_USDT = 50.0           # 單筆現貨最大配置上限：50U

BLACKLIST = {"USDC/USDT", "FDUSD/USDT", "TUSD/USDT", "USDP/USDT", "EUR/USDT", "USD1/USDT"}


# ==========================================
# 2. 狀態讀寫（防止 GitHub Actions 重複發訊）
# ==========================================
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict):
    # 清理超過 24 小時的過期紀錄，保持檔案精簡
    now = time.time()
    cleaned = {k: v for k, v in state.items() if now - v < 86400}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)


# ==========================================
# 3. 技術指標計算引擎
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
# 4. 0~100 分綜合評分與 TP/SL 計算
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

    # 計算 Entry / SL / TP1 / TP2
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

    raw_pos_usdt = RISK_PER_TRADE_USDT / sl_pct
    suggested_usdt = min(raw_pos_usdt, MAX_SINGLE_POS_USDT)
    actual_risk_usdt = suggested_usdt * sl_pct

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "score": total_score,
        "entry": price,
        "sl": sl,
        "sl_pct": sl_pct * 100,
        "tp1": tp1,
        "tp1_pct": (tp1 - price) / price * 100,
        "tp2": tp2,
        "tp2_pct": (tp2 - price) / price * 100,
        "suggested_usdt": suggested_usdt,
        "actual_risk_usdt": actual_risk_usdt,
        "breakdown": breakdown,
    }


# ==========================================
# 5. mplfinance K 線圖表生成
# ==========================================
def generate_chart_image(df: pd.DataFrame, setup: dict) -> io.BytesIO:
    sub = df.iloc[-60:].copy()
    sub.index = pd.to_datetime(sub["ts"], unit="ms")
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
        title=f"\n{setup['symbol']} [{setup['timeframe']}] | Score: {setup['score']}/100",
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

    p = setup["entry"]
    fmt = ".6f" if p < 0.1 else (".4f" if p < 10 else ".2f")
    tags = [
        (setup["tp2"], f" TP2 {setup['tp2']:{fmt}} ", "#00c853", "black"),
        (setup["tp1"], f" TP1 {setup['tp1']:{fmt}} ", "#2ecc71", "black"),
        (setup["entry"], f" ENTRY {setup['entry']:{fmt}} ", "#3498db", "white"),
        (setup["sl"], f" SL {setup['sl']:{fmt}} ", "#ff5252", "white"),
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
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, edgecolor="none", alpha=0.9),
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
# 6. Discord Webhook 卡片＋圖片上傳
# ==========================================
def send_discord_alert(setup: dict, df: pd.DataFrame):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 未設定 DISCORD_WEBHOOK_URL 環境變數，跳過發送")
        return

    score = setup["score"]
    if score >= 85:
        grade = "🔥 極強共振訊號（S級）"
        color = 0x2ECC71
    elif score >= 75:
        grade = "⚡ 強勢突破訊號（A級）"
        color = 0xF1C40F
    else:
        grade = "🧪 手動觸發測試預覽"
        color = 0x3498DB

    p = setup["entry"]
    fmt = ".6f" if p < 0.1 else (".4f" if p < 10 else ".2f")
    detail_text = " | ".join([f"**{k}**: `{v}`" for k, v in setup["breakdown"].items()])
    tv_symbol = setup["symbol"].replace("/", "")

    embed = {
        "title": f"{grade} | {setup['symbol']} 【{setup['timeframe']} 時區】",
        "description": (
            f"### 🎯 做單綜合評分：**`{score} / 100 分`**\n"
            f"📊 [點此開啟 TradingView 圖表確認](https://www.tradingview.com/chart/?symbol=OKX:{tv_symbol})"
        ),
        "color": color,
        "fields": [
            {"name": "📍 建議進場 (Entry)", "value": f"`{setup['entry']:{fmt}}`", "inline": True},
            {"name": f"🛑 止損價 (SL -{setup['sl_pct']:.2f}%)", "value": f"`{setup['sl']:{fmt}}`", "inline": True},
            {"name": "💰 100U 建議下單", "value": f"`{setup['suggested_usdt']:.1f} U` (損 `-{setup['actual_risk_usdt']:.2f} U`)", "inline": True},
            {"name": f"🎯 止盈 1 (TP1 +{setup['tp1_pct']:.2f}%)", "value": f"`{setup['tp1']:{fmt}}`", "inline": True},
            {"name": f"🚀 止盈 2 (TP2 +{setup['tp2_pct']:.2f}%)", "value": f"`{setup['tp2']:{fmt}}`", "inline": True},
            {"name": "🧠 多指標評分拆解", "value": detail_text, "inline": False},
        ],
        "image": {"url": "attachment://chart.png"},
        "footer": {"text": f"GitHub Actions 雲端雷達 • UTC {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"},
    }

    payload = {"username": "量化訊號雷達 (100U戰隊)", "embeds": [embed]}

    try:
        chart_buf = generate_chart_image(df, setup)
        files = {"file": ("chart.png", chart_buf, "image/png")}
        data = {"payload_json": json.dumps(payload)}
        resp = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files, timeout=15)
        if resp.status_code in (200, 204):
            print(f"✅ 已發送 Discord 圖文通知: {setup['symbol']} [{setup['timeframe']}] ({score}分)")
        else:
            print(f"❌ Discord 發送失敗 ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"❌ Discord 連線或繪圖異常: {e}")


# ==========================================
# 7. 單次全市場掃描入口
# ==========================================
def run_scan():
    # 使用 OKX 公開行情介面（不被 GitHub Actions 美國 Azure 機房 IP 封鎖）
    exchange = ccxt.okx({"enableRateLimit": True})
    alert_history = load_state()

    # 1. 取得全市場前 N 大活躍 USDT 現貨
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

    # 2. 評估 BTC 大盤環境分
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
                last_sent = alert_history.get(state_key, 0)
                if setup["score"] >= MIN_SCORE_TO_ALERT and (now - last_sent > COOLDOWN_HOURS * 3600):
                    send_discord_alert(setup, df_tf)
                    alert_history[state_key] = now

        except Exception as e:
            print(f"[跳過] {sym}: {e}")

    all_candidates.sort(key=lambda x: x[0]["score"], reverse=True)
    print(f"📊 本輪掃描完成（BTC 環境分: {btc_score}/15），最高分 Top 5：")
    for s, _ in all_candidates[:5]:
        print(f"   • {s['symbol']:<12} [{s['timeframe']:<3}] 評分: {s['score']}分")

    # 若是在 GitHub Actions 網頁手動點擊「Run workflow」，強制推播最高分一張測試
    if FORCE_TEST and all_candidates:
        print("🧪 偵測到手動測試執行，強制發送最高分標的至 Discord...")
        send_discord_alert(all_candidates[0][0], all_candidates[0][1])

    save_state(alert_history)


if __name__ == "__main__":
    run_scan()
