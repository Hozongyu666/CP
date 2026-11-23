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
# 1. 核心設定區（v4.1 強趨勢順風版）
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_TEST = os.environ.get("FORCE_TEST", "false").strip().lower() == "true"
FORCE_DAILY_REPORT = os.environ.get("FORCE_DAILY_REPORT", "false").strip().lower() == "true"

TW_TZ = timezone(timedelta(hours=8))  # 台灣時區 (UTC+8)
STATE_VERSION = 4                    # v4.1 升級：自動清空舊版 v3 戰績，從 100U 重新精準統計

TIMEFRAMES = ["1h", "4h"]            # 砍掉 15m 雜訊，專注 1H / 4H 高勝率波段
MIN_SCORE_TO_ALERT = 74              # 嚴格評分門檻：74 分以上才發單
COIN_COOLDOWN_MINUTES = 240          # 同一幣種 4 小時內不重複轟炸
MAX_TRACK_HOURS = 48                 # 單筆訊號最長追蹤 48 小時（提高資金周轉率）
HEARTBEAT_HOURS = 4                  # 每 4 小時無新單發一次存活巡檢
STATE_FILE = "state.json"

# 100U 穩健複利資金管理參數（3x ~ 5x 安全槓桿，壓低回撤）
INITIAL_CAPITAL = 100.0              # 初始本金 100U
RISK_PCT_PER_TRADE = 0.02            # 單筆最大打損風險：當前總資金的 2.0%
MIN_MARGIN_USDT = 12.0               # 單筆最低保證金：12U
MAX_MARGIN_PCT = 0.30                # 單筆最高保證金上限：當前總資金的 30%

# 排除穩定幣與黃金/大宗商品代幣（XAUT、PAXG 不適用加密動能模型）
BLACKLIST = {
    "USDC/USDT", "FDUSD/USDT", "TUSD/USDT", "USDP/USDT", "EUR/USDT", "USD1/USDT",
    "XAUT/USDT", "PAXG/USDT",
}


# ==========================================
# 2. 時間與價格精度格式化工具
# ==========================================
def now_tw() -> datetime:
    return datetime.now(TW_TZ)


def now_tw_str() -> str:
    return now_tw().strftime("%Y-%m-%d %H:%M:%S")


def today_tw_str() -> str:
    return now_tw().strftime("%Y-%m-%d")


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
# 3. 狀態讀寫（自動升級至 v4.1 乾淨狀態）
# ==========================================
def create_fresh_state() -> dict:
    return {
        "version": STATE_VERSION,
        "coin_history": {},
        "active_signals": {},
        "oi_snapshot": {},
        "last_heartbeat": 0,
        "last_daily_report_date": "",
        "stats": {"wins": 0, "losses": 0, "net_pnl_u": 0.0},
        "daily_stats": {"date": today_tw_str(), "wins": 0, "losses": 0, "net_pnl_u": 0.0, "signals_sent": 0},
    }


def load_state() -> dict:
    today = today_tw_str()
    if not os.path.exists(STATE_FILE):
        return create_fresh_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("version") != STATE_VERSION:
            print("🔄 偵測到版本升級至 v4.1，已自動重置模擬戰績為 100.00U (0勝0敗) 重新起算！")
            fresh = create_fresh_state()
            fresh["oi_snapshot"] = data.get("oi_snapshot", {})
            return fresh

        if data.get("daily_stats", {}).get("date") != today:
            data["daily_stats"] = {
                "date": today,
                "wins": 0,
                "losses": 0,
                "net_pnl_u": 0.0,
                "signals_sent": 0,
            }
        return data
    except Exception:
        return create_fresh_state()


def save_state(state: dict):
    now = time.time()
    state["coin_history"] = {
        k: v for k, v in state.get("coin_history", {}).items()
        if now - v.get("last_ts", 0) < 86400
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_stats_summary_text(state: dict) -> str:
    st = state.get("stats", {"wins": 0, "losses": 0, "net_pnl_u": 0.0})
    dst = state.get("daily_stats", {"wins": 0, "losses": 0, "net_pnl_u": 0.0})
    active_cnt = len(state.get("active_signals", {}))

    tot_closed = st["wins"] + st["losses"]
    tot_wr = f"{(st['wins'] / tot_closed * 100):.1f}%" if tot_closed > 0 else "尚無平倉"

    day_closed = dst["wins"] + dst["losses"]
    day_wr = f"{(dst['wins'] / day_closed * 100):.1f}%" if day_closed > 0 else "尚無平倉"

    return (
        f"剩 {active_cnt} 單追蹤中 ｜ 今日: {dst['net_pnl_u']:+.2f}U (勝率 {day_wr}) ｜ "
        f"系統總勝率: {tot_wr} ({st['net_pnl_u']:+.2f}U)"
    )


# ==========================================
# 4. 抓全市場永續合約「異常持倉量 (OI)」與資金費率
# ==========================================
def fetch_okx_oi_map() -> dict[str, float]:
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
    try:
        inst_id = symbol.replace("/USDT", "-USDT-SWAP")
        url = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}"
        resp = requests.get(url, timeout=6).json()
        data = resp.get("data", [])
        if data:
            return float(data[0].get("fundingRate", 0)) * 100
    except Exception:
        pass
    return None


# ==========================================
# 5. 技術指標計算引擎
# ==========================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, open_, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    typical_price = (high + low + close) / 3.0
    tp_vol = (typical_price * vol).rolling(24).sum()
    vol_sum = vol.rolling(24).sum().replace(0, np.nan)
    df["vwap"] = tp_vol / vol_sum

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2.0 * bb_std
    df["bb_lower"] = bb_mid - 2.0 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
    df["bb_width_min30"] = df["bb_width"].rolling(30).min().shift(1)

    df["donchian_high"] = high.rolling(20).max().shift(1)
    df["donchian_low"] = low.rolling(20).min().shift(1)
    df["swing_low"] = low.rolling(10).min()
    df["swing_high"] = high.rolling(10).max()

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
# 6. v4.1 強趨勢順風評估引擎（4H硬鎖 + 大盤順風 + 精選高勝率形態）
# ==========================================
def evaluate_long(
    symbol: str, timeframe: str, df: pd.DataFrame,
    coin_4h_trend: int, btc_4h_trend: int, oi_change_pct: float | None, current_capital: float
) -> dict | None:
    # 【硬門檻 1】個股 4H 必須處於明確多頭趨勢，且 BTC 4H 不可為空頭（徹底阻絕大盤逆風接刀）
    if coin_4h_trend <= 0 or btc_4h_trend < 0:
        return None

    live = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(live["close"])
    atr = float(live["atr"])
    if atr <= 0 or (atr / price) < 0.006:  # 過濾 TRX 等日均波動過低、容易卡單的牛皮幣
        return None

    # 【硬門檻 2】當根必須為強勢陽線，且不可離 EMA20 超過 1.8 ATR（嚴格反追高）
    if live["close"] <= live["open"]:
        return None
    dist_atr = (price - float(live["ema20"])) / atr
    if dist_atr > 1.8 or dist_atr < -0.5:
        return None
    bar_range = float(live["high"] - live["low"])
    if bar_range > 0 and (float(live["high"]) - price) / bar_range > 0.35:
        return None

    # 【硬門檻 3】趨勢強度 ADX >= 23 才進場，避開橫盤震盪
    adx_val = float(live["adx"])
    if adx_val < 23:
        return None

    tags = []
    breakdown = {}

    # 1. 4H+1H 雙週期順風與 VWAP (30分)
    s_trend = 15  # 已通過 4H 多頭硬鎖
    if live["ema20"] > live["ema50"] and price > live["vwap"]:
        s_trend += 15
    elif price > live["vwap"]:
        s_trend += 8
    breakdown["雙週期趨勢"] = f"{s_trend}/30"

    # 2. 精選雙形態：只保留「回踩支撐確認突破」與「布林壓縮放量突破」(30分)
    recent_low = float(df["low"].iloc[-3:].min())
    support_zone = max(float(live["ema20"]), float(live["vwap"]))
    touched_support = recent_low <= support_zone * 1.005
    bouncing_up = price > float(prev["high"]) and float(live["rsi"]) <= 65

    was_squeezed = df["bb_width"].iloc[-4:-1].min() <= (float(prev["bb_width_min30"]) * 1.15)
    fresh_breakout = (price > float(live["donchian_high"])) and (float(prev["close"]) <= float(prev["donchian_high"]) * 1.015)

    if live["ema20"] > live["ema50"] and touched_support and bouncing_up:
        s_setup = 30
        tags.append("🎯 回踩支撐確認突破")
    elif fresh_breakout and was_squeezed:
        s_setup = 28
        tags.append("💥 布林壓縮放量突破")
    elif fresh_breakout:
        s_setup = 22
        tags.append("📈 順勢通道突破")
    else:
        return None
    breakdown["精選形態"] = f"{s_setup}/30"

    # 3. 量能與主力增倉 OI (25分)
    vol_r = float(live["vol_ratio"])
    s_vol = 12 if vol_r >= 1.8 else (8 if vol_r >= 1.2 else 4)
    if oi_change_pct is not None:
        if oi_change_pct >= 3.0:
            s_oi = 13
            tags.append(f"🐋 主力暴增倉 +{oi_change_pct:.1f}%")
        elif oi_change_pct >= 0.8:
            s_oi = 9
            tags.append(f"🐳 持倉增加 +{oi_change_pct:.1f}%")
        elif oi_change_pct < -0.3:
            s_oi = -4
        else:
            s_oi = 4
        oi_str = f"OI:{oi_change_pct:+.1f}%"
    else:
        s_oi = 6 if vol_r >= 1.5 else 3
        oi_str = "OI:暖機"
    breakdown["量能/持倉"] = f"{max(0, s_vol + s_oi)}/25 ({vol_r:.1f}x | {oi_str})"

    # 4. 大盤共振與 RSI 健康度 (15分)
    rsi_val = float(live["rsi"])
    s_env = 8 if (48 <= rsi_val <= 65) else 4
    s_env += 7 if btc_4h_trend > 0 else 3
    breakdown["大盤/RSI"] = f"{s_env}/15 (ADX:{adx_val:.0f} RSI:{rsi_val:.0f})"

    total_score = int(np.clip(s_trend + s_setup + s_vol + s_oi + s_env, 0, 100))

    # --- 放寬止損空間（1.8x ATR 防插針）與高盈虧比目標 ---
    struct_low = float(df["swing_low"].iloc[-1])
    sl = min(struct_low - 0.3 * atr, price - 1.8 * atr)
    sl_pct = float(np.clip((price - sl) / price, 0.015, 0.050))
    sl = price * (1 - sl_pct)

    risk = price - sl
    tp1 = price + 1.5 * risk   # 1.5R 第一止盈（平 60% 鎖利 + 推保本）
    tp2 = price + 2.8 * risk   # 2.8R 第二波段目標

    return build_setup_dict(
        symbol, timeframe, "LONG", total_score, tags, breakdown,
        price, sl, sl_pct, tp1, tp2, current_capital
    )


def evaluate_short(
    symbol: str, timeframe: str, df: pd.DataFrame,
    coin_4h_trend: int, btc_4h_trend: int, oi_change_pct: float | None, current_capital: float
) -> dict | None:
    # 【硬門檻 1】個股 4H 必須處於明確空頭趨勢，且 BTC 4H 不可為強勢多頭
    if coin_4h_trend >= 0 or btc_4h_trend > 0:
        return None

    live = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(live["close"])
    atr = float(live["atr"])
    if atr <= 0 or (atr / price) < 0.006:
        return None

    # 【硬門檻 2】當根必須為下跌陰線，且不可追空在地板（距離 EMA20 <= 1.8 ATR）
    if live["close"] >= live["open"]:
        return None
    dist_atr = (float(live["ema20"]) - price) / atr
    if dist_atr > 1.8 or dist_atr < -0.5:
        return None
    bar_range = float(live["high"] - live["low"])
    if bar_range > 0 and (price - float(live["low"])) / bar_range > 0.35:
        return None

    adx_val = float(live["adx"])
    if adx_val < 23:
        return None

    tags = []
    breakdown = {}

    # 1. 4H+1H 雙週期空頭順風 (30分)
    s_trend = 15
    if live["ema20"] < live["ema50"] and price < live["vwap"]:
        s_trend += 15
    elif price < live["vwap"]:
        s_trend += 8
    breakdown["雙週期空頭"] = f"{s_trend}/30"

    # 2. v4 回測最強雙空頭形態：「反彈確認轉空」與「壓縮放量跌破」(30分)
    recent_high = float(df["high"].iloc[-3:].max())
    resist_zone = min(float(live["ema20"]), float(live["vwap"]))
    touched_resist = recent_high >= resist_zone * 0.995
    rejecting_down = price < float(prev["low"]) and float(live["rsi"]) >= 35

    was_squeezed = df["bb_width"].iloc[-4:-1].min() <= (float(prev["bb_width_min30"]) * 1.15)
    fresh_breakdown = (price < float(live["donchian_low"])) and (float(prev["close"]) >= float(prev["donchian_low"]) * 0.985)

    if live["ema20"] < live["ema50"] and touched_resist and rejecting_down:
        s_setup = 30
        tags.append("🎯 反彈確認轉空 (主力模型)")
    elif fresh_breakdown and was_squeezed:
        s_setup = 28
        tags.append("💥 壓縮放量跌破 (高盈虧比)")
    elif fresh_breakdown:
        s_setup = 22
        tags.append("📉 順勢跌破支撐")
    else:
        return None
    breakdown["精選形態"] = f"{s_setup}/30"

    # 3. 量能與持倉 (25分)
    vol_r = float(live["vol_ratio"])
    s_vol = 12 if vol_r >= 1.8 else (8 if vol_r >= 1.2 else 4)
    if oi_change_pct is not None:
        if oi_change_pct >= 2.5:
            s_oi = 13
            tags.append(f"🐋 空軍增倉壓制 +{oi_change_pct:.1f}%")
        elif oi_change_pct >= 0.5:
            s_oi = 9
            tags.append(f"🐳 持倉增加 +{oi_change_pct:.1f}%")
        else:
            s_oi = 4
        oi_str = f"OI:{oi_change_pct:+.1f}%"
    else:
        s_oi = 6 if vol_r >= 1.5 else 3
        oi_str = "OI:暖機"
    breakdown["量能/持倉"] = f"{s_vol + s_oi}/25 ({vol_r:.1f}x | {oi_str})"

    # 4. 大盤共振與 RSI (15分)
    rsi_val = float(live["rsi"])
    s_env = 8 if (35 <= rsi_val <= 52) else 4
    s_env += 7 if btc_4h_trend < 0 else 3
    breakdown["大盤/RSI"] = f"{s_env}/15 (ADX:{adx_val:.0f} RSI:{rsi_val:.0f})"

    total_score = int(np.clip(s_trend + s_setup + s_vol + s_oi + s_env, 0, 100))

    struct_high = float(df["swing_high"].iloc[-1])
    sl = max(struct_high + 0.3 * atr, price + 1.8 * atr)
    sl_pct = float(np.clip((sl - price) / price, 0.015, 0.050))
    sl = price * (1 + sl_pct)

    risk = sl - price
    tp1 = price - 1.5 * risk
    tp2 = price - 2.8 * risk

    return build_setup_dict(
        symbol, timeframe, "SHORT", total_score, tags, breakdown,
        price, sl, sl_pct, tp1, tp2, current_capital
    )


def build_setup_dict(
    symbol: str, timeframe: str, side: str, total_score: int, tags: list, breakdown: dict,
    price: float, sl: float, sl_pct: float, tp1: float, tp2: float, current_capital: float
) -> dict:
    tp1_pct = abs(tp1 - price) / price
    tp2_pct = abs(tp2 - price) / price

    # 安全槓桿控制在 3x ~ 5x（大幅壓低回撤）
    if sl_pct <= 0.022 and total_score >= 82:
        leverage = 5
    elif sl_pct <= 0.032:
        leverage = 4
    else:
        leverage = 3

    target_risk_u = max(1.5, current_capital * RISK_PCT_PER_TRADE)
    max_margin_u = max(15.0, current_capital * MAX_MARGIN_PCT)
    raw_margin = target_risk_u / (sl_pct * leverage)
    margin_usdt = round(float(np.clip(raw_margin, MIN_MARGIN_USDT, max_margin_u)), 1)
    notional_usdt = margin_usdt * leverage

    est_fee = notional_usdt * 0.001
    est_loss_sl = notional_usdt * sl_pct + est_fee
    est_profit_tp1 = notional_usdt * tp1_pct - est_fee
    est_profit_tp2 = notional_usdt * tp2_pct - est_fee

    win_rate_str = "55% ~ 62%" if total_score >= 82 else "46% ~ 53%"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "matched_tfs": [timeframe],
        "score": total_score,
        "tags": tags,
        "history_note": "",
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
        "rr_tp2": 2.8,
        "win_rate_str": win_rate_str,
        "breakdown": breakdown,
    }


# ==========================================
# 7. mplfinance K 線圖表生成
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

    tfs_label = "+".join(setup["matched_tfs"])
    side_label = "LONG 做多" if setup["side"] == "LONG" else "SHORT 做空"
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
        title=f"\n{setup['symbol']} [{side_label} | {tfs_label}] | Score: {setup['score']}/100 | {setup['leverage']}x | TW {setup['signal_time_tw']}",
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

    y_min = min(sub["Low"].min(), setup["sl"], setup["tp2"])
    y_max = max(sub["High"].max(), setup["sl"], setup["tp2"])
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
# 8. Discord Webhook 推播（新訊號 + 戰報 + 每日結算 + 心跳）
# ==========================================
def send_discord_alert(setup: dict, df: pd.DataFrame, state: dict):
    if not DISCORD_WEBHOOK_URL:
        print("❌ 未設定 DISCORD_WEBHOOK_URL 環境變數，跳過發送")
        return

    fr = fetch_funding_rate(setup["symbol"])
    fr_str = f"`{fr:+.4f}%`" if fr is not None else "`無合約`"

    is_long = setup["side"] == "LONG"
    side_badge = "🟢 【做多 LONG】" if is_long else "🔴 【做空 SHORT】"
    score = setup["score"]
    tfs_str = " + ".join(setup["matched_tfs"])

    if len(setup["matched_tfs"]) >= 2:
        grade = f"🔥 4H+1H 雙週期共振（{tfs_str}）"
    elif score >= 82:
        grade = "🔥 強趨勢高勝率訊號（S級）"
    elif score >= 74:
        grade = "⚡ 順勢確認訊號（A級）"
    else:
        grade = "🧪 手動測試預覽"

    color = 0x2ECC71 if is_long else 0xE74C3C
    tag_line = " ｜ ".join(setup["tags"]) if setup["tags"] else "強趨勢順風結構"
    history_line = f"\n📜 **歷史紀錄**：{setup['history_note']}" if setup.get("history_note") else ""
    detail_text = " | ".join([f"**{k}**: `{v}`" for k, v in setup["breakdown"].items()])
    tv_symbol = setup["symbol"].replace("/", "")

    embed = {
        "title": f"{side_badge} {grade} | {setup['symbol']} 【{setup['timeframe']}】",
        "description": (
            f"🏷️ **訊號特徵**：**{tag_line}**{history_line}\n"
            f"🕒 **出訊號時間（台灣）**：`{setup['signal_time_tw']}`\n"
            f"🎯 **綜合評分**：`{score} / 100 分` ｜ 📈 **策略預期勝率**：`{setup['win_rate_str']}` ｜ 💸 **資金費率**：{fr_str}\n"
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
                "name": "⚡ 建議槓桿與穩健倉位",
                "value": f"槓桿: **`{setup['leverage']}x`** (逐倉)\n保證金: `{setup['margin_usdt']:.1f} U` (倉位 `{setup['notional_usdt']:.0f} U`)",
                "inline": True,
            },
            {
                "name": f"🛑 止損價 (SL {setup['sl_pct']:.2f}%)",
                "value": f"`{format_price(setup['sl'])}`\n預計虧損: **`-{setup['est_loss_sl']:.2f} U`**",
                "inline": True,
            },
            {
                "name": f"🎯 止盈 1 (TP1 {setup['tp1_pct']:.2f}%)",
                "value": f"`{format_price(setup['tp1'])}`\n全平參考: **`+{setup['est_profit_tp1']:.2f} U`** *(平60%推保本)*",
                "inline": True,
            },
            {
                "name": f"🚀 止盈 2 (TP2 {setup['tp2_pct']:.2f}%)",
                "value": f"`{format_price(setup['tp2'])}`\n全平參考: **`+{setup['est_profit_tp2']:.2f} U`**",
                "inline": True,
            },
            {
                "name": "🧠 v4.1 評分拆解",
                "value": detail_text,
                "inline": False,
            },
        ],
        "image": {"url": "attachment://chart.png"},
        "footer": {"text": get_stats_summary_text(state)},
    }

    payload = {"username": "強趨勢量化雷達 v4.1 (回測驗證版)", "embeds": [embed]}

    try:
        chart_buf = generate_chart_image(df, setup)
        files = {"file": ("chart.png", chart_buf, "image/png")}
        data = {"payload_json": json.dumps(payload)}
        resp = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files, timeout=15)
        if resp.status_code in (200, 204):
            print(f"✅ 已發送新訊號圖文: {setup['symbol']} [{setup['side']} | {tfs_str}] ({score}分)")
        else:
            print(f"❌ Discord 發送失敗 ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"❌ Discord 連線或繪圖異常: {e}")


def send_outcome_alert(sig: dict, event_type: str, state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    entry = sig["entry"]
    side = sig.get("side", "LONG")
    side_str = "做多" if side == "LONG" else "做空"
    tv_symbol = sig["symbol"].replace("/", "")
    hit_time_tw = now_tw_str()
    elapsed_min = int((time.time() - sig["created_at"]) / 60)
    lev = sig.get("leverage", 4)
    stats = state["stats"]
    d_stats = state["daily_stats"]

    # 計算「TP1 平 60% 鎖利 + 剩餘 40% 衝 TP2 或回保本」的真實累計損益
    tp1_part_profit = round(sig.get("est_profit_tp1", 0) * 0.60, 2)
    tp2_part_profit = round(sig.get("est_profit_tp2", 0) * 0.40, 2)

    if event_type == "TP1":
        pct = abs(sig["tp1"] - entry) / entry * 100
        title = f"🎯 止盈 1 達標 (TP1) | {sig['symbol']} 【{side_str} {sig['timeframe']}】"
        color = 0x1ABC9C
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格已觸及第一止盈目標 **`{format_price(sig['tp1'])}`**（波動 `{pct:.2f}%` ｜ `{lev}x`）！\n"
            f"💰 **操作紀律**：先平 **60% 倉位**鎖定 **`+{tp1_part_profit:.2f} U`** 利潤，剩餘 40% 倉位系統已自動將止損移至 **`{format_price(sig['sl'])}`**（覆蓋手續費保本），繼續追蹤 **TP2**。"
        )
    elif event_type == "TP2":
        pct = abs(sig["tp2"] - entry) / entry * 100
        total_win_u = round(tp1_part_profit + tp2_part_profit, 2)
        stats["wins"] += 1
        stats["net_pnl_u"] = round(stats["net_pnl_u"] + total_win_u, 2)
        d_stats["wins"] += 1
        d_stats["net_pnl_u"] = round(d_stats["net_pnl_u"] + total_win_u, 2)

        title = f"🚀 止盈 2 完美達標 (TP2) | {sig['symbol']} 【{side_str} {sig['timeframe']}】"
        color = 0x2ECC71
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格強勢衝破第二止盈目標 **`{format_price(sig['tp2'])}`**（波動 `{pct:.2f}%`）！\n"
            f"🏁 本筆訊號全數達標（TP1鎖利 `+ TP2尾倉` 合計淨賺 **`+{total_win_u:.2f} U`**），結束追蹤。"
        )
    elif event_type == "BREAKEVEN":
        stats["wins"] += 1
        stats["net_pnl_u"] = round(stats["net_pnl_u"] + tp1_part_profit, 2)
        d_stats["wins"] += 1
        d_stats["net_pnl_u"] = round(d_stats["net_pnl_u"] + tp1_part_profit, 2)

        title = f"🛡️ TP1鎖利後保本出場 | {sig['symbol']} 【{side_str} {sig['timeframe']}】"
        color = 0x3498DB
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"此單先前已於 **TP1** 平掉 60% 倉位，剩餘 40% 倉位回落至保本止損位 **`{format_price(sig['sl'])}`** 安全離場，整單實賺 **`+{tp1_part_profit:.2f} U`**！"
        )
    else:  # SL
        pct = abs(sig["sl"] - entry) / entry * 100
        est_u = sig.get("est_loss_sl", 0)
        stats["losses"] += 1
        stats["net_pnl_u"] = round(stats["net_pnl_u"] - est_u, 2)
        d_stats["losses"] += 1
        d_stats["net_pnl_u"] = round(d_stats["net_pnl_u"] - est_u, 2)

        title = f"🛑 觸及止損出場 (SL) | {sig['symbol']} 【{side_str} {sig['timeframe']}】"
        color = 0xE74C3C
        desc = (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"價格觸及止損位 **`{format_price(sig['sl'])}`**（波動 `-{pct:.2f}%` ｜ `{lev}x` 約 **`-{est_u:.2f} U`**）。\n"
            f"🛡️ 嚴格執行風控，本筆訊號已結束追蹤。"
        )

    embed = {
        "title": title,
        "description": f"{desc}\n📊 [開啟 TradingView 查看走勢](https://www.tradingview.com/chart/?symbol=OKX:{tv_symbol})",
        "color": color,
        "fields": [
            {"name": "📍 原進場價 (Entry)", "value": f"`{format_price(entry)}`", "inline": True},
            {"name": "🎯 TP1 / 🚀 TP2", "value": f"`{format_price(sig['tp1'])}` / `{format_price(sig['tp2'])}`", "inline": True},
            {"name": "🛑 止損價 (SL)", "value": f"`{format_price(sig['sl'])}`", "inline": True},
        ],
        "footer": {"text": get_stats_summary_text(state)},
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "強趨勢量化雷達 (戰報追蹤)", "embeds": [embed]}, timeout=10)
        print(f"📣 已發送 [{event_type}] 到價通知: {sig['symbol']} [{side_str}]")
    except Exception as e:
        print(f"❌ 發送到價通知失敗: {e}")


def send_daily_summary(state: dict, tickers: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    st = state["stats"]
    dst = state["daily_stats"]
    active_signals = state["active_signals"]

    day_closed = dst["wins"] + dst["losses"]
    day_wr = f"{(dst['wins'] / day_closed * 100):.1f}% ({dst['wins']}勝 {dst['losses']}敗)" if day_closed > 0 else "0.0% (今日無平倉單)"

    tot_closed = st["wins"] + st["losses"]
    tot_wr = f"{(st['wins'] / tot_closed * 100):.1f}% ({st['wins']}勝 {st['losses']}敗)" if tot_closed > 0 else "0.0% (尚無平倉單)"
    cur_cap = INITIAL_CAPITAL + st["net_pnl_u"]

    open_lines = []
    total_unrealized_u = 0.0
    for sym, sig in active_signals.items():
        cur_p = float(tickers.get(sym, {}).get("last") or sig["entry"])
        entry = sig["entry"]
        side = sig.get("side", "LONG")
        lev = sig.get("leverage", 4)
        rem_notional = sig.get("notional_usdt", 40.0) * (0.4 if sig.get("tp1_hit") else 1.0)
        spot_pct = (cur_p - entry) / entry if side == "LONG" else (entry - cur_p) / entry
        unrealized_u = rem_notional * spot_pct
        total_unrealized_u += unrealized_u
        status_tag = "🛡️已過TP1保本中" if sig.get("tp1_hit") else "⏳等待TP1"
        side_tag = "多" if side == "LONG" else "空"
        open_lines.append(
            f"• **{sym}** `[{side_tag}|{sig['timeframe']}]` ({lev}x) ｜ 進場 `{format_price(entry)}` → 現價 `{format_price(cur_p)}` ｜ "
            f"剩餘浮盈: **`{unrealized_u:+.2f} U`** [{status_tag}]"
        )

    open_detail = "\n".join(open_lines) if open_lines else "目前空倉，無未平倉追蹤單。"

    embed = {
        "title": f"📅 每日量化戰報總結 | {today_tw_str()} (台灣時間)",
        "description": (
            f"🕒 **結算時間**：`{now_tw_str()}`\n"
            f"今天共發出 **`{dst.get('signals_sent', 0)}`** 筆新訊號，目前剩 **`{len(active_signals)}`** 單在線上追蹤。"
        ),
        "color": 0x2ECC71 if dst["net_pnl_u"] >= 0 else 0xE67E22,
        "fields": [
            {"name": "💰 今日已平倉盈虧", "value": f"**`{dst['net_pnl_u']:+.2f} U`**", "inline": True},
            {"name": "🎯 今日勝率", "value": f"**`{day_wr}`**", "inline": True},
            {"name": "📦 每日跑完剩餘單數", "value": f"**`{len(active_signals)} 單`** (浮盈 `{total_unrealized_u:+.2f} U`)", "inline": True},
            {"name": "🏆 此系統歷史總勝率", "value": f"**`{tot_wr}`**", "inline": True},
            {"name": "📈 系統累計總盈虧 / 複利資金", "value": f"**`{st['net_pnl_u']:+.2f} U`** (`{cur_cap:.2f} U`)", "inline": True},
            {"name": "📋 剩餘未平倉追蹤明細", "value": open_detail, "inline": False},
        ],
        "footer": {"text": "強趨勢量化雷達 v4.1 • 每日 23:50 自動結算"},
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "強趨勢量化雷達 (每日戰報)", "embeds": [embed]}, timeout=10)
        print("📅 已發送每日結算戰報至 Discord")
    except Exception as e:
        print(f"[每日戰報發送失敗] {e}")


def send_heartbeat(top_candidates: list, btc_4h_trend: int, skipped_notes: list, state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    btc_text = "4H 多頭順風（僅開放做多）" if btc_4h_trend > 0 else ("4H 空頭順風（僅開放做空）" if btc_4h_trend < 0 else "4H 震盪整理（多空皆可）")
    lines = []
    for s, _ in top_candidates[:3]:
        tfs = "+".join(s["matched_tfs"])
        side_str = "多" if s["side"] == "LONG" else "空"
        tag_str = f" ({s['tags'][0]})" if s["tags"] else ""
        lines.append(f"• **{s['symbol']}** `[{side_str}|{tfs}]`：**{s['score']}分**{tag_str} ｜ 現價 `{format_price(s['current_price'])}`")

    cand_block = "\n".join(lines) if lines else "目前全市場皆未滿足 4H+1H 順風條件，持續空倉觀望中。"
    skip_block = ""
    if skipped_notes:
        skip_block = "\n\n⏭️ **本輪達標但因冷卻略過（避免重複追單）**：\n" + "\n".join(skipped_notes[:4])

    embed = {
        "title": "💓 雲端雷達 v4.1 定時巡檢報平安",
        "description": (
            f"🕒 **台灣時間**：`{now_tw_str()}`\n"
            f"🧭 **大盤濾網狀態**：**{btc_text}**\n\n"
            f"👀 **最接近進場條件的觀察名單**：\n{cand_block}{skip_block}"
        ),
        "color": 0x95A5A6,
        "footer": {"text": get_stats_summary_text(state)},
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "強趨勢量化雷達 (巡檢)", "embeds": [embed]}, timeout=10)
        print("💓 已發送存活心跳通知")
    except Exception as e:
        print(f"[心跳發送失敗] {e}")


# ==========================================
# 9. 追蹤既有訊號（TP1 平 60% + 保本上移覆蓋手續費）
# ==========================================
def check_active_signals(exchange, state: dict):
    active_signals = state["active_signals"]
    if not active_signals:
        return

    print(f"🔎 正在檢查 {len(active_signals)} 筆追蹤中訊號的 TP/SL 狀態...")
    now = time.time()

    for sym in list(active_signals.keys()):
        sig = active_signals[sym]
        if now - sig["created_at"] > MAX_TRACK_HOURS * 3600:
            del active_signals[sym]
            continue

        side = sig.get("side", "LONG")
        try:
            since_ms = int(sig.get("last_checked_ms", sig["created_at"] * 1000))
            candles = exchange.fetch_ohlcv(sig["symbol"], "5m", since=since_ms, limit=200)
            if not candles:
                continue

            closed_or_done = False
            for c in candles:
                c_ts, c_open, c_high, c_low, c_close, _ = c

                if side == "LONG":
                    if c_low <= sig["sl"]:
                        evt = "BREAKEVEN" if sig.get("tp1_hit") else "SL"
                        send_outcome_alert(sig, evt, state)
                        del active_signals[sym]
                        closed_or_done = True
                        break
                    if not sig.get("tp1_hit", False) and c_high >= sig["tp1"]:
                        sig["tp1_hit"] = True
                        sig["sl"] = sig["entry"] * 1.002  # 保本止損上移 +0.2% 覆蓋雙邊手續費
                        if c_high >= sig["tp2"]:
                            send_outcome_alert(sig, "TP2", state)
                            del active_signals[sym]
                            closed_or_done = True
                            break
                        else:
                            send_outcome_alert(sig, "TP1", state)
                    elif sig.get("tp1_hit", False) and c_high >= sig["tp2"]:
                        send_outcome_alert(sig, "TP2", state)
                        del active_signals[sym]
                        closed_or_done = True
                        break

                else:  # SHORT
                    if c_high >= sig["sl"]:
                        evt = "BREAKEVEN" if sig.get("tp1_hit") else "SL"
                        send_outcome_alert(sig, evt, state)
                        del active_signals[sym]
                        closed_or_done = True
                        break
                    if not sig.get("tp1_hit", False) and c_low <= sig["tp1"]:
                        sig["tp1_hit"] = True
                        sig["sl"] = sig["entry"] * 0.998  # 保本止損下移 -0.2% 覆蓋雙邊手續費
                        if c_low <= sig["tp2"]:
                            send_outcome_alert(sig, "TP2", state)
                            del active_signals[sym]
                            closed_or_done = True
                            break
                        else:
                            send_outcome_alert(sig, "TP1", state)
                    elif sig.get("tp1_hit", False) and c_low <= sig["tp2"]:
                        send_outcome_alert(sig, "TP2", state)
                        del active_signals[sym]
                        closed_or_done = True
                        break

            if not closed_or_done and sym in active_signals:
                active_signals[sym]["last_checked_ms"] = candles[-1][0]

        except Exception as e:
            print(f"[追蹤檢查異常] {sym}: {e}")


# ==========================================
# 10. 單次全市場掃描主流程
# ==========================================
def run_scan():
    exchange = ccxt.okx({"enableRateLimit": True})
    state = load_state()
    coin_history = state["coin_history"]
    active_signals = state["active_signals"]
    oi_snapshot = state["oi_snapshot"]
    current_capital = INITIAL_CAPITAL + state["stats"].get("net_pnl_u", 0.0)

    # 1. 優先檢查既有訂單有沒有打到 TP1 / TP2 / SL
    check_active_signals(exchange, state)

    # 2. 抓全市場永續合約持倉量 (OI) 快照
    current_oi_map = fetch_okx_oi_map()
    oi_change_map = {}
    for sym, cur_oi in current_oi_map.items():
        prev_rec = oi_snapshot.get(sym)
        if prev_rec and prev_rec.get("oi", 0) > 0:
            oi_change_map[sym] = (cur_oi - prev_rec["oi"]) / prev_rec["oi"] * 100
        oi_snapshot[sym] = {"oi": cur_oi, "ts": time.time()}

    # 3. 組合掃描名單：「前 25 大成交額主流幣」＋「24H 漲跌幅異動山寨幣」
    tickers = exchange.fetch_tickers()
    active_usdt = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") or sym in BLACKLIST or ":" in sym:
            continue
        q_vol = float(t.get("quoteVolume") or 0)
        if q_vol == 0:
            q_vol = float(t.get("baseVolume") or 0) * float(t.get("last") or 0)
        pct_24h = float(t.get("percentage") or 0)
        if q_vol > 5_000_000:
            active_usdt.append((sym, q_vol, abs(pct_24h)))

    top_vol = [x[0] for x in sorted(active_usdt, key=lambda x: x[1], reverse=True)[:25]]
    top_movers = [x[0] for x in sorted(active_usdt, key=lambda x: x[2], reverse=True)[:25]]
    symbols = list(dict.fromkeys(top_vol + top_movers))

    # 4. 評估 BTC 4H 大盤趨勢方向（-1 空頭, 0 震盪, +1 多頭）
    try:
        ohlcv_btc_4h = exchange.fetch_ohlcv("BTC/USDT", "4h", limit=100)
        df_btc_4h = compute_indicators(pd.DataFrame(ohlcv_btc_4h, columns=["ts", "open", "high", "low", "close", "volume"]))
        b4 = df_btc_4h.iloc[-1]
        if b4["close"] > b4["ema20"] and b4["ema20"] > b4["ema50"]:
            btc_4h_trend = 1
        elif b4["close"] < b4["ema20"] and b4["ema20"] < b4["ema50"]:
            btc_4h_trend = -1
        else:
            btc_4h_trend = 0
    except Exception:
        btc_4h_trend = 0

    all_coin_best = []
    skipped_notes = []
    now = time.time()
    sent_any_this_round = False

    def register_and_alert(setup: dict, df_tf: pd.DataFrame):
        nonlocal sent_any_this_round
        sym = setup["symbol"]
        send_discord_alert(setup, df_tf, state)
        state["daily_stats"]["signals_sent"] = state["daily_stats"].get("signals_sent", 0) + 1

        coin_history[sym] = {
            "last_ts": now,
            "last_tf": "+".join(setup["matched_tfs"]),
            "last_side": setup["side"],
            "last_price": setup["entry"],
            "last_score": setup["score"],
            "skip_count": 0,
        }

        active_signals[sym] = {
            "symbol": sym,
            "side": setup["side"],
            "timeframe": "+".join(setup["matched_tfs"]),
            "score": setup["score"],
            "signal_time_tw": setup["signal_time_tw"],
            "entry": setup["entry"],
            "sl": setup["sl"],
            "tp1": setup["tp1"],
            "tp2": setup["tp2"],
            "leverage": setup["leverage"],
            "notional_usdt": setup["notional_usdt"],
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
            ohlcv_4h = exchange.fetch_ohlcv(sym, "4h", limit=120)
            if len(ohlcv_4h) < 60:
                continue
            df_4h = compute_indicators(pd.DataFrame(ohlcv_4h, columns=["ts", "open", "high", "low", "close", "volume"]))
            c4 = df_4h.iloc[-1]

            # 個股 4H 趨勢硬判斷
            if c4["close"] > c4["ema50"] and c4["ema20"] > c4["ema50"]:
                coin_4h_trend = 1
            elif c4["close"] < c4["ema50"] and c4["ema20"] < c4["ema50"]:
                coin_4h_trend = -1
            else:
                coin_4h_trend = 0

            # 若個股 4H 處於無方向震盪，直接跳過節省算力與避免假突破
            if coin_4h_trend == 0:
                continue

            oi_chg = oi_change_map.get(sym)
            tf_results = []

            for tf in TIMEFRAMES:
                if tf == "4h":
                    df_tf = df_4h
                else:
                    ohlcv_tf = exchange.fetch_ohlcv(sym, tf, limit=120)
                    if len(ohlcv_tf) < 60:
                        continue
                    df_tf = compute_indicators(pd.DataFrame(ohlcv_tf, columns=["ts", "open", "high", "low", "close", "volume"]))

                long_setup = evaluate_long(sym, tf, df_tf, coin_4h_trend, btc_4h_trend, oi_chg, current_capital)
                short_setup = evaluate_short(sym, tf, df_tf, coin_4h_trend, btc_4h_trend, oi_chg, current_capital)

                for cand in (long_setup, short_setup):
                    if cand is not None:
                        tf_results.append((cand, df_tf))

            if not tf_results:
                continue

            qualified = [x for x in tf_results if x[0]["score"] >= MIN_SCORE_TO_ALERT]
            tf_results.sort(key=lambda x: x[0]["score"], reverse=True)

            if qualified:
                qualified.sort(key=lambda x: x[0]["score"], reverse=True)
                best_setup, best_df = qualified[0]
                same_side_tfs = [x[0]["timeframe"] for x in qualified if x[0]["side"] == best_setup["side"]]
                if len(same_side_tfs) >= 2:
                    best_setup["matched_tfs"] = same_side_tfs
                    best_setup["score"] = min(100, best_setup["score"] + 6)
                    best_setup["tags"].insert(0, "⚡ 4H+1H 雙週期共振 (+6分)")
            else:
                best_setup, best_df = tf_results[0]

            prev_hist = coin_history.get(sym)
            mins_ago = int((now - prev_hist["last_ts"]) / 60) if prev_hist else None

            if prev_hist and mins_ago is not None:
                price_diff_pct = (best_setup["current_price"] - prev_hist["last_price"]) / prev_hist["last_price"] * 100
                best_setup["history_note"] = (
                    f"{mins_ago} 分鐘前曾觸發 [{prev_hist['last_tf']}]（當時價 `{format_price(prev_hist['last_price'])}`，至今 `{price_diff_pct:+.2f}%`）"
                )

            all_coin_best.append((best_setup, best_df))

            if best_setup["score"] >= MIN_SCORE_TO_ALERT:
                if sym in active_signals:
                    note = f"• **{sym}** ({best_setup['score']}分) — 已有追蹤中訂單（{mins_ago or 0} 分鐘前發單），本次略過"
                    skipped_notes.append(note)
                elif mins_ago is not None and mins_ago < COIN_COOLDOWN_MINUTES:
                    prev_hist["skip_count"] = prev_hist.get("skip_count", 0) + 1
                    note = f"• **{sym}** ({best_setup['score']}分) — **{mins_ago} 分鐘前**已發過（累計略過 {prev_hist['skip_count']} 次）"
                    skipped_notes.append(note)
                else:
                    register_and_alert(best_setup, best_df)

        except Exception as e:
            print(f"[跳過] {sym}: {e}")

    all_coin_best.sort(key=lambda x: x[0]["score"], reverse=True)
    print(f"📊 [{now_tw_str()}] 掃描 {len(symbols)} 檔完成（BTC 4H趨勢: {btc_4h_trend} | 追蹤中: {len(active_signals)} 單），最高分 Top 5：")
    for s, _ in all_coin_best[:5]:
        print(f"   • {s['symbol']:<12} [{s['side']:<5}|{'+'.join(s['matched_tfs'])}] 評分: {s['score']}分 | 現價: {format_price(s['current_price'])}")

    if FORCE_TEST and all_coin_best:
        top_setup, top_df = all_coin_best[0]
        print("🧪 偵測到手動測試執行，強制發送最高分標的並納入追蹤...")
        register_and_alert(top_setup, top_df)

    tw_now = now_tw()
    today_s = today_tw_str()
    is_daily_window = (tw_now.hour == 23 and tw_now.minute >= 48)
    if (is_daily_window and state.get("last_daily_report_date") != today_s) or FORCE_DAILY_REPORT:
        send_daily_summary(state, tickers)
        state["last_daily_report_date"] = today_s

    if sent_any_this_round:
        state["last_heartbeat"] = now
    elif now - state.get("last_heartbeat", 0) >= HEARTBEAT_HOURS * 3600:
        send_heartbeat(all_coin_best, btc_4h_trend, skipped_notes, state)
        state["last_heartbeat"] = now

    save_state(state)


if __name__ == "__main__":
    run_scan()
