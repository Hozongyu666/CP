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
# 1. 核心設定區（v5.1 結構支撐校準版）
# ==========================================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE_TEST = os.environ.get("FORCE_TEST", "false").strip().lower() == "true"
FORCE_DAILY_REPORT = os.environ.get("FORCE_DAILY_REPORT", "false").strip().lower() == "true"

TW_TZ = timezone(timedelta(hours=8))  # 台灣時區 (UTC+8)
STATE_VERSION = 6                    # v5.1 結構支撐校準版
STATE_FILE = "state.json"

# --- 系統一（穩健主趨勢）參數：與 v4.2 回測 100% 對齊 ---
MAIN_MIN_SCORE = 74
MAIN_COOLDOWN_MIN = 240
MAIN_MAX_HOURS = 48
INITIAL_CAPITAL = 100.0
MAIN_RISK_PCT = 0.02

# --- 系統二（高R暴漲妖幣狙擊）參數 ---
MOON_MIN_SCORE = 72
MOON_COOLDOWN_MIN = 360
MOON_MAX_HOURS = 72
MOON_MARGIN_USDT = 15.0

HEARTBEAT_HOURS = 4

BLACKLIST_BASE = {
    "USDC", "FDUSD", "TUSD", "USDP", "EUR", "USD1", "DAI",
    "XAUT", "PAXG",
}
VALID_X_CRYPTO = {"XRP", "XLM", "XTZ", "XMR", "XEC", "XDC", "XAI", "XCH", "XRD", "XVS"}


def is_excluded_base(base: str) -> bool:
    if base in BLACKLIST_BASE:
        return True
    if base.startswith("X") and base not in VALID_X_CRYPTO:
        return True
    return False


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
# 3. 雙系統獨立狀態讀寫
# ==========================================
def create_fresh_state() -> dict:
    today = today_tw_str()
    return {
        "version": STATE_VERSION,
        "coin_history": {},
        "active_main": {},
        "active_moon": {},
        "last_heartbeat": 0,
        "last_daily_report_date": "",
        "stats_main": {"wins": 0, "losses": 0, "net_pnl_u": 0.0},
        "stats_moon": {"wins": 0, "losses": 0, "net_pnl_u": 0.0},
        "daily_stats": {
            "date": today,
            "main_sent": 0,
            "moon_sent": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl_u": 0.0,
        },
    }


def load_state() -> dict:
    today = today_tw_str()
    if not os.path.exists(STATE_FILE):
        return create_fresh_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != STATE_VERSION:
            print("🔄 升級至 v5.1 結構支撐校準版，已初始化乾淨帳本！")
            return create_fresh_state()
        if data.get("daily_stats", {}).get("date") != today:
            data["daily_stats"] = {
                "date": today,
                "main_sent": 0,
                "moon_sent": 0,
                "wins": 0,
                "losses": 0,
                "net_pnl_u": 0.0,
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


def get_footer_text(state: dict, system_type: str) -> str:
    sm = state["stats_main"]
    smoon = state["stats_moon"]
    m_tot = sm["wins"] + sm["losses"]
    m_wr = f"{sm['wins']/m_tot*100:.0f}%" if m_tot > 0 else "無平倉"
    moon_tot = smoon["wins"] + smoon["losses"]
    moon_wr = f"{smoon['wins']/moon_tot*100:.0f}%" if moon_tot > 0 else "無平倉"

    if system_type == "MOONSHOT":
        return f"妖幣狙擊戰績: {smoon['wins']}勝{smoon['losses']}敗 ({moon_wr} | {smoon['net_pnl_u']:+.2f}U) ｜ 主系統: {sm['net_pnl_u']:+.2f}U"
    return f"主系統戰績: {sm['wins']}勝{sm['losses']}敗 ({m_wr} | {sm['net_pnl_u']:+.2f}U) ｜ 妖幣狙擊: {smoon['net_pnl_u']:+.2f}U"


# ==========================================
# 4. OKX 永續合約微結構接口
# ==========================================
def fetch_swap_oi_and_funding(inst_id: str) -> tuple[float | None, float | None, float | None]:
    oi_1h_pct = None
    oi_3h_pct = None
    funding_pct = None

    try:
        r_oi = requests.get(
            "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history",
            params={"instId": inst_id, "period": "1H", "limit": "5"},
            timeout=6,
        ).json()
        rows = r_oi.get("data", [])
        if len(rows) >= 2:
            cur_oi, prev_1h = float(rows[0][2]), float(rows[1][2])
            if prev_1h > 0:
                oi_1h_pct = (cur_oi - prev_1h) / prev_1h * 100
        if len(rows) >= 4:
            cur_oi, prev_3h = float(rows[0][2]), float(rows[3][2])
            if prev_3h > 0:
                oi_3h_pct = (cur_oi - prev_3h) / prev_3h * 100
    except Exception:
        pass

    try:
        r_fr = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": inst_id},
            timeout=6,
        ).json()
        fr_data = r_fr.get("data", [])
        if fr_data:
            funding_pct = float(fr_data[0].get("fundingRate", 0)) * 100
    except Exception:
        pass

    return oi_1h_pct, oi_3h_pct, funding_pct


# ==========================================
# 5. 技術指標計算引擎
# ==========================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    typical_price = (high + low + close) / 3.0
    df["vwap"] = (typical_price * vol).rolling(24).sum() / vol.rolling(24).sum().replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2.0 * bb_std
    df["bb_lower"] = bb_mid - 2.0 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
    df["bb_width_min30"] = df["bb_width"].rolling(30).min().shift(1)
    df["bb_width_q25"] = df["bb_width"].rolling(48).quantile(0.25).shift(1)

    df["donchian_high"] = high.rolling(20).max().shift(1)
    df["donchian_low"] = low.rolling(20).min().shift(1)
    df["swing_low"] = low.rolling(10).min()
    df["swing_high"] = high.rolling(10).max()

    df["vol_ma20"] = vol.rolling(20).mean().shift(1)
    df["vol_ratio"] = vol / df["vol_ma20"].replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

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
# 6. 【系統二】高R暴漲妖幣狙擊引擎
# ==========================================
def evaluate_moonshot(symbol: str, inst_id: str, df_1h: pd.DataFrame) -> dict | None:
    if len(df_1h) < 60:
        return None

    live = df_1h.iloc[-1]
    prev = df_1h.iloc[-2]
    price = float(live["close"])
    open_p = float(live["open"])
    high_p = float(live["high"])
    low_p = float(live["low"])
    atr = float(live["atr"])
    vol_r = float(live["vol_ratio"])

    if np.isnan(atr) or atr <= 0 or np.isnan(vol_r):
        return None

    if price <= open_p or price <= float(live["donchian_high"]) or vol_r < 2.5:
        return None

    bar_range = high_p - low_p
    if bar_range > 0 and (high_p - price) / bar_range > 0.35:
        return None

    price_12h_ago = float(df_1h["close"].iloc[-13])
    if (price - price_12h_ago) / price_12h_ago > 0.28:
        return None

    oi_1h, oi_3h, fr_pct = fetch_swap_oi_and_funding(inst_id)

    tags = []
    breakdown = {}

    recent_min_bb = float(df_1h["bb_width"].iloc[-5:-1].min())
    q25_bb = float(prev["bb_width_q25"]) if not np.isnan(prev["bb_width_q25"]) else recent_min_bb
    if recent_min_bb <= q25_bb * 1.15:
        s_struct = 30
        tags.append("🌀 底部極致吸籌壓縮")
    else:
        s_struct = 20
        tags.append("🚀 強勢箱體突破")
    breakdown["吸籌壓縮"] = f"{s_struct}/30"

    if vol_r >= 5.0:
        s_vol = 25
        tags.append(f"💣 核爆級巨量 ({vol_r:.1f}x)")
    elif vol_r >= 3.5:
        s_vol = 20
        tags.append(f"🔥 異常爆量 ({vol_r:.1f}x)")
    else:
        s_vol = 14
        tags.append(f"📈 放量啟動 ({vol_r:.1f}x)")
    breakdown["量能核爆"] = f"{s_vol}/25 ({vol_r:.1f}x)"

    oi_max_chg = max([x for x in (oi_1h, oi_3h) if x is not None], default=None)
    if oi_max_chg is not None:
        if oi_max_chg >= 10.0:
            s_oi = 25
            tags.append(f"🐋 主力狂開倉 OI {oi_max_chg:+.1f}%")
        elif oi_max_chg >= 5.0:
            s_oi = 18
            tags.append(f"🐳 持倉暴增 OI {oi_max_chg:+.1f}%")
        elif oi_max_chg >= 1.5:
            s_oi = 10
        else:
            s_oi = 0
        oi_str = f"1H:{oi_1h:+.1f}% / 3H:{oi_3h:+.1f}%" if (oi_1h is not None and oi_3h is not None) else f"{oi_max_chg:+.1f}%"
    else:
        s_oi = 10
        oi_str = "OI查詢逾時"
    breakdown["異常增倉"] = f"{s_oi}/25 ({oi_str})"

    if fr_pct is not None:
        if fr_pct <= -0.05:
            s_fr = 20
            tags.append(f"⚡ 極端負費率軋空 ({fr_pct:+.3f}%)")
        elif fr_pct <= -0.015:
            s_fr = 14
            tags.append(f"🔥 負費率空軍燃料 ({fr_pct:+.3f}%)")
        elif fr_pct <= 0.01:
            s_fr = 6
        else:
            s_fr = 0
        fr_str = f"{fr_pct:+.4f}%"
    else:
        s_fr = 5
        fr_str = "未知"
    breakdown["軋空燃料"] = f"{s_fr}/20 (費率:{fr_str})"

    total_score = int(np.clip(s_struct + s_vol + s_oi + s_fr, 0, 100))

    sl_raw = min(low_p * 0.992, price - 2.0 * atr)
    sl_pct = float(np.clip((price - sl_raw) / price, 0.035, 0.075))
    sl = price * (1 - sl_pct)

    leverage = 10 if sl_pct <= 0.052 else 8
    margin_usdt = MOON_MARGIN_USDT
    notional_usdt = margin_usdt * leverage

    tp1 = price * 1.15
    tp2 = price * 1.35
    tp3 = price * 1.80

    est_fee = notional_usdt * 0.001
    est_jackpot_u = (
        notional_usdt * 0.30 * 0.15
        + notional_usdt * 0.30 * 0.35
        + notional_usdt * 0.20 * 0.80
        + notional_usdt * 0.20 * 1.20
        - est_fee
    )

    return {
        "system": "MOONSHOT",
        "symbol": symbol,
        "inst_id": inst_id,
        "timeframe": "1h暴漲起爆",
        "side": "LONG",
        "matched_tfs": ["1h"],
        "score": total_score,
        "tags": tags,
        "history_note": "",
        "signal_time_tw": now_tw_str(),
        "current_price": price,
        "entry": price,
        "sl": sl,
        "sl_pct": sl_pct * 100,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "leverage": leverage,
        "margin_usdt": margin_usdt,
        "notional_usdt": notional_usdt,
        "est_loss_sl": min(margin_usdt, notional_usdt * sl_pct + est_fee),
        "est_profit_tp1": notional_usdt * 0.15 - est_fee,
        "est_profit_tp2": notional_usdt * 0.35 - est_fee,
        "est_jackpot_u": est_jackpot_u,
        "rr_max": round(est_jackpot_u / margin_usdt, 1),
        "breakdown": breakdown,
    }


# ==========================================
# 7. 【系統一】主趨勢穩健評估引擎（1H 進場 + 7日結構支撐/壓力牆校準）
# ==========================================
def evaluate_main_trend(
    symbol: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
    coin_4h_trend: int, btc_4h_trend: int, current_capital: float
) -> dict | None:
    if coin_4h_trend == 0:
        return None

    live = df_1h.iloc[-1]
    prev = df_1h.iloc[-2]
    price = float(live["close"])
    atr = float(live["atr"])
    if np.isnan(atr) or atr <= 0 or (atr / price) < 0.006:
        return None

    adx_val = float(live["adx"])
    if np.isnan(adx_val) or adx_val < 24:
        return None

    bar_range = float(live["high"] - live["low"])
    vol_r = float(live["vol_ratio"])

    # 取得過去 7 天（42 根 4H K 線）的結構高低點牆，避免止盈畫到無底洞
    major_7d_high = float(df_4h["high"].iloc[-42:].max())
    major_7d_low = float(df_4h["low"].iloc[-42:].min())

    candidates = []

    # ---------------- 做多（4H 與 BTC 4H 同步偏多，1H 進場） ----------------
    if coin_4h_trend > 0 and btc_4h_trend > 0:
        dist_atr = (price - float(live["ema20"])) / atr
        wick_ok = True if bar_range == 0 else ((float(live["high"]) - price) / bar_range <= 0.35)
        if live["close"] > live["open"] and (-0.5 <= dist_atr <= 1.8) and wick_ok:
            s_trend = 15 + (15 if (live["ema20"] > live["ema50"] and price > live["vwap"]) else (8 if price > live["vwap"] else 0))
            recent_low = float(df_1h["low"].iloc[-3:].min())
            touched_support = recent_low <= max(float(live["ema20"]), float(live["vwap"])) * 1.005
            bouncing_up = price > float(prev["high"]) and float(live["rsi"]) <= 64 and vol_r >= 1.25 and float(live["macd_hist"]) > float(prev["macd_hist"])
            was_squeezed = df_1h["bb_width"].iloc[-4:-1].min() <= (float(prev["bb_width_min30"]) * 1.15)
            fresh_break = (price > float(live["donchian_high"])) and (float(prev["close"]) <= float(prev["donchian_high"]) * 1.015)

            s_setup = 0
            tags = []
            if fresh_break and was_squeezed and vol_r >= 1.3:
                s_setup = 32
                tags.append("💥 壓縮放量突破")
            elif fresh_break and vol_r >= 1.2:
                s_setup = 28
                tags.append("📈 順勢通道突破")
            elif live["ema20"] > live["ema50"] and touched_support and bouncing_up:
                s_setup = 26
                tags.append("🎯 回踩確認突破")

            if s_setup > 0:
                # 計算真實 1H 結構止損；若真實止損超過 3.8% 則直接放棄，不硬截斷
                sl = min(float(df_1h["swing_low"].iloc[-1]) - 0.3 * atr, price - 1.6 * atr)
                sl_pct = (price - sl) / price
                if sl_pct > 0.038:
                    return None
                if sl_pct < 0.014:
                    sl_pct = 0.014
                    sl = price * (1 - sl_pct)

                risk = price - sl
                # 對比上方 7 日大壓力牆：若未達 1.3R 就先撞牆，直接放棄
                resist_ceiling = major_7d_high * 0.997
                if (resist_ceiling - price) < 1.3 * risk:
                    return None

                tp1 = min(price + 1.5 * risk, resist_ceiling)
                # TP2 最多看到略為突破 7 日高點 1.5%，不畫天邊目標
                tp2 = min(price + 2.6 * risk, major_7d_high * 1.015)

                s_vol = 18 if vol_r >= 1.8 else (13 if vol_r >= 1.25 else 7)
                s_env = (8 if 48 <= float(live["rsi"]) <= 64 else 4) + 7
                score = int(np.clip(s_trend + s_setup + s_vol + s_env, 0, 100))
                if score >= MAIN_MIN_SCORE:
                    candidates.append(build_main_dict(
                        symbol, "1h", "LONG", score, tags,
                        {"4H+1H順風": f"{s_trend}/30", "順風形態": f"{s_setup}/32", "量能確認": f"{s_vol}/18", "大盤/RSI": f"{s_env}/20"},
                        price, sl, sl_pct, tp1, tp2, current_capital
                    ))

    # ---------------- 做空（4H 與 BTC 4H 同步偏空，1H 進場） ----------------
    if coin_4h_trend < 0 and btc_4h_trend < 0:
        dist_atr = (float(live["ema20"]) - price) / atr
        wick_ok = True if bar_range == 0 else ((price - float(live["low"])) / bar_range <= 0.35)
        if live["close"] < live["open"] and (-0.5 <= dist_atr <= 1.8) and wick_ok:
            s_trend = 15 + (15 if (live["ema20"] < live["ema50"] and price < live["vwap"]) else (8 if price < live["vwap"] else 0))
            recent_high = float(df_1h["high"].iloc[-3:].max())
            touched_resist = recent_high >= min(float(live["ema20"]), float(live["vwap"])) * 0.995
            rejecting_down = price < float(prev["low"]) and float(live["rsi"]) >= 36 and vol_r >= 1.25 and float(live["macd_hist"]) < float(prev["macd_hist"])
            was_squeezed = df_1h["bb_width"].iloc[-4:-1].min() <= (float(prev["bb_width_min30"]) * 1.15)
            fresh_down = (price < float(live["donchian_low"])) and (float(prev["close"]) >= float(prev["donchian_low"]) * 0.985)

            s_setup = 0
            tags = []
            if fresh_down and was_squeezed and vol_r >= 1.3:
                s_setup = 32
                tags.append("💥 壓縮放量跌破")
            elif fresh_down and vol_r >= 1.2:
                s_setup = 28
                tags.append("📉 順勢跌破支撐")
            elif live["ema20"] < live["ema50"] and touched_resist and rejecting_down:
                s_setup = 26
                tags.append("🎯 反彈確認轉空")

            if s_setup > 0:
                # 計算真實 1H 結構止損；若止損超過 3.8% 直接放棄，避免像 DOGE 一樣把 SL 硬塞進壓力區中間
                sl = max(float(df_1h["swing_high"].iloc[-1]) + 0.3 * atr, price + 1.6 * atr)
                sl_pct = (sl - price) / price
                if sl_pct > 0.038:
                    return None
                if sl_pct < 0.014:
                    sl_pct = 0.014
                    sl = price * (1 + sl_pct)

                risk = sl - price
                # 對比下方 7 日大支撐牆：若未達 1.3R 就先撞上前低支撐，直接放棄
                support_floor = major_7d_low * 1.003
                if (price - support_floor) < 1.3 * risk:
                    return None

                tp1 = max(price - 1.5 * risk, support_floor)
                # TP2 最多看到略為跌破 7 日前低 1.5%，絕不畫無底洞目標
                tp2 = max(price - 2.6 * risk, major_7d_low * 0.985)

                s_vol = 18 if vol_r >= 1.8 else (13 if vol_r >= 1.25 else 7)
                s_env = (8 if 36 <= float(live["rsi"]) <= 52 else 4) + 7
                score = int(np.clip(s_trend + s_setup + s_vol + s_env, 0, 100))
                if score >= MAIN_MIN_SCORE:
                    candidates.append(build_main_dict(
                        symbol, "1h", "SHORT", score, tags,
                        {"4H+1H空頭": f"{s_trend}/30", "順風形態": f"{s_setup}/32", "量能確認": f"{s_vol}/18", "大盤/RSI": f"{s_env}/20"},
                        price, sl, sl_pct, tp1, tp2, current_capital
                    ))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


def build_main_dict(
    symbol: str, timeframe: str, side: str, score: int, tags: list, breakdown: dict,
    price: float, sl: float, sl_pct: float, tp1: float, tp2: float, current_capital: float
) -> dict:
    tp1_pct = abs(tp1 - price) / price
    tp2_pct = abs(tp2 - price) / price
    rr_tp1 = round(tp1_pct / sl_pct, 2)
    rr_tp2 = round(tp2_pct / sl_pct, 2)
    leverage = 5 if (sl_pct <= 0.022 and score >= 82) else (4 if sl_pct <= 0.030 else 3)

    target_risk_u = max(1.5, current_capital * MAIN_RISK_PCT)
    max_margin_u = max(15.0, current_capital * 0.30)
    margin_usdt = round(float(np.clip(target_risk_u / (sl_pct * leverage), 12.0, max_margin_u)), 1)
    notional_usdt = margin_usdt * leverage
    est_fee = notional_usdt * 0.001

    return {
        "system": "MAIN",
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "matched_tfs": ["4h順風", "1h進場"],
        "score": score,
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
        "est_loss_sl": notional_usdt * sl_pct + est_fee,
        "est_profit_tp1": notional_usdt * tp1_pct - est_fee,
        "est_profit_tp2": notional_usdt * tp2_pct - est_fee,
        "rr_tp1": rr_tp1,
        "rr_tp2": rr_tp2,
        "breakdown": breakdown,
    }


# ==========================================
# 8. K 線圖表生成
# ==========================================
def generate_chart_image(df: pd.DataFrame, setup: dict) -> io.BytesIO:
    sub = df.iloc[-60:].copy()
    sub.index = pd.to_datetime(sub["ts"], unit="ms") + pd.Timedelta(hours=8)
    sub = sub.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350", edge="inherit", wick="inherit",
        volume={"up": "#26a69a", "down": "#ef5350"},
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds", marketcolors=mc,
        facecolor="#1e1f22", figcolor="#1e1f22", gridcolor="#2b2d31", gridstyle="--",
        y_on_right=True, rc={"axes.labelcolor": "#b5bac1", "xtick.color": "#b5bac1", "ytick.color": "#b5bac1"},
    )

    apds = [
        mpf.make_addplot(sub["ema20"], color="#f1c40f", width=1.2, label="EMA 20"),
        mpf.make_addplot(sub["vwap"], color="#00d2d3", linestyle="--", width=1.2, label="VWAP"),
    ]

    is_moon = setup["system"] == "MOONSHOT"
    levels = [setup["tp2"], setup["tp1"], setup["entry"], setup["sl"]]
    if is_moon:
        level_colors = ["#e056fd", "#00e676", "#3498db", "#ff5252"]
        title_str = f"\n[MOONSHOT SNIPER] {setup['symbol']} | Score: {setup['score']}/100 | {setup['leverage']}x | TW {setup['signal_time_tw']}"
    else:
        level_colors = ["#00e676", "#2ecc71", "#3498db", "#ff5252"]
        side_label = "LONG" if setup["side"] == "LONG" else "SHORT"
        title_str = f"\n[MAIN TREND] {setup['symbol']} [{side_label} | 1H] | Score: {setup['score']}/100 | TW {setup['signal_time_tw']}"

    fig, axes = mpf.plot(
        sub, type="candle", style=style, volume=True, addplot=apds,
        hlines=dict(hlines=levels, colors=level_colors, linestyle=["--", "--", "-", "--"], linewidths=[1.2, 1.2, 1.5, 1.5]),
        title=title_str, ylabel="Price (USDT)", ylabel_lower="Volume",
        figsize=(11, 6.2), panel_ratios=(3.2, 1), tight_layout=True, returnfig=True,
    )

    ax_main = axes[0]
    ax_main.legend(loc="upper left", facecolor="#2b2d31", edgecolor="#3f4147", labelcolor="white", fontsize=8.5)

    tags = [
        (setup["tp2"], f" TP2 {format_price(setup['tp2'])} ", "#e056fd" if is_moon else "#00c853", "black"),
        (setup["tp1"], f" TP1 {format_price(setup['tp1'])} ", "#2ecc71", "black"),
        (setup["entry"], f" ENTRY {format_price(setup['entry'])} ", "#3498db", "white"),
        (setup["sl"], f" SL {format_price(setup['sl'])} ", "#ff5252", "white"),
    ]
    x_pos = len(sub) - 0.5
    for y_val, text, bg_color, fg_color in tags:
        ax_main.text(
            x_pos, y_val, text, color=fg_color, fontsize=8, fontweight="bold", va="center", ha="right",
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
# 9. Discord 雙系統獨立推播引擎
# ==========================================
def send_discord_alert(setup: dict, df: pd.DataFrame, state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    is_moon = setup["system"] == "MOONSHOT"
    tag_line = " ｜ ".join(setup["tags"]) if setup["tags"] else "結構確認"
    detail_text = " | ".join([f"**{k}**: `{v}`" for k, v in setup["breakdown"].items()])
    tv_symbol = setup["symbol"].replace("/", "")

    if is_moon:
        embed = {
            "title": f"🚀🚀 【系統二｜暴漲妖幣狙擊】{setup['symbol']} 異常起爆！",
            "description": (
                f"🔥 **盤口異常特徵**：**{tag_line}**\n"
                f"🕒 **起爆時間（台灣）**：`{setup['signal_time_tw']}`\n"
                f"🎯 **妖幣爆發評分**：`{setup['score']} / 100 分` ｜ 🎰 **吃滿波段潛力**：**`+{setup['est_jackpot_u']:.0f} U` (`{setup['rr_max']}R`)**\n"
                f"⚠️ **紀律提醒**：務必使用 **逐倉 `{setup['leverage']}x`**、固定小額 **`{setup['margin_usdt']:.0f} U`** 博取高倍暴利！\n"
                f"📊 [立即開啟 TradingView 確認](https://www.tradingview.com/chart/?symbol=OKX:{tv_symbol}-PERP)"
            ),
            "color": 0x9B59B6,
            "fields": [
                {
                    "name": "💲 現價 / 突破進場",
                    "value": f"現價: `{format_price(setup['current_price'])}`\n進場: `{format_price(setup['entry'])}`",
                    "inline": True,
                },
                {
                    "name": "🎰 彩票逐倉配置",
                    "value": f"槓桿: **`{setup['leverage']}x` (逐倉)**\n保證金: **`{setup['margin_usdt']:.0f} U`** (倉位 `{setup['notional_usdt']:.0f} U`)",
                    "inline": True,
                },
                {
                    "name": f"🛑 止損位 (SL -{setup['sl_pct']:.2f}%)",
                    "value": f"`{format_price(setup['sl'])}`\n最大虧損鎖死: **`-{setup['est_loss_sl']:.2f} U`**",
                    "inline": True,
                },
                {
                    "name": "🎯 TP1 (+15% 抽回本金)",
                    "value": f"`{format_price(setup['tp1'])}`\n平30%鎖利＋推保本",
                    "inline": True,
                },
                {
                    "name": "🔥 TP2 (+35% 波段大肉)",
                    "value": f"`{format_price(setup['tp2'])}`\n全倉參考: **`+{setup['est_profit_tp2']:.1f} U`**",
                    "inline": True,
                },
                {
                    "name": "👑 TP3 (+80% 超級妖幣)",
                    "value": f"`{format_price(setup['tp3'])}`\n吃滿波段: **`+{setup['est_jackpot_u']:.1f} U`**",
                    "inline": True,
                },
                {
                    "name": "🧠 妖幣微結構拆解",
                    "value": detail_text,
                    "inline": False,
                },
            ],
            "image": {"url": "attachment://chart.png"},
            "footer": {"text": get_footer_text(state, "MOONSHOT")},
        }
        bot_name = "妖幣狙擊雷達【系統二：高R暴漲】"
    else:
        is_long = setup["side"] == "LONG"
        side_badge = "🟢 做多 LONG" if is_long else "🔴 做空 SHORT"
        embed = {
            "title": f"🛡️ 【系統一｜主趨勢穩健】{side_badge} | {setup['symbol']} 【1H進場】",
            "description": (
                f"🏷️ **結構特徵**：**{tag_line}**\n"
                f"🕒 **出訊號時間（台灣）**：`{setup['signal_time_tw']}`\n"
                f"🎯 **綜合評分**：`{setup['score']} / 100 分` ｜ ⚖️ **實質盈虧比**：TP1 `1:{setup['rr_tp1']}` ／ TP2 `1:{setup['rr_tp2']}`（已校準7日支撐牆）\n"
                f"📊 [點此開啟 TradingView 圖表確認](https://www.tradingview.com/chart/?symbol=OKX:{tv_symbol})"
            ),
            "color": 0x2ECC71 if is_long else 0x3498DB,
            "fields": [
                {
                    "name": "💲 現價 / 建議進場",
                    "value": f"現價: `{format_price(setup['current_price'])}`\n進場: `{format_price(setup['entry'])}`",
                    "inline": True,
                },
                {
                    "name": "⚡ 穩健槓桿與倉位",
                    "value": f"槓桿: **`{setup['leverage']}x`** (逐倉)\n保證金: `{setup['margin_usdt']:.1f} U` (倉位 `{setup['notional_usdt']:.0f} U`)",
                    "inline": True,
                },
                {
                    "name": f"🛑 結構止損 (SL {setup['sl_pct']:.2f}%)",
                    "value": f"`{format_price(setup['sl'])}`\n預計虧損: **`-{setup['est_loss_sl']:.2f} U`**",
                    "inline": True,
                },
                {
                    "name": f"🎯 止盈 1 (TP1 {setup['tp1_pct']:.2f}%)",
                    "value": f"`{format_price(setup['tp1'])}`\n平60%推保本 (`+{setup['est_profit_tp1']*0.6:.2f}U`)",
                    "inline": True,
                },
                {
                    "name": f"🚀 止盈 2 (TP2 {setup['tp2_pct']:.2f}%)",
                    "value": f"`{format_price(setup['tp2'])}`\n全平參考: **`+{setup['est_profit_tp2']:.2f} U`**",
                    "inline": True,
                },
                {
                    "name": "🧠 主趨勢評分拆解",
                    "value": detail_text,
                    "inline": False,
                },
            ],
            "image": {"url": "attachment://chart.png"},
            "footer": {"text": get_footer_text(state, "MAIN")},
        }
        bot_name = "量化趨勢雷達【系統一：穩健主單】"

    payload = {"username": bot_name, "embeds": [embed]}
    try:
        chart_buf = generate_chart_image(df, setup)
        files = {"file": ("chart.png", chart_buf, "image/png")}
        resp = requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files=files, timeout=15)
        if resp.status_code in (200, 204):
            print(f"✅ 已發送 [{setup['system']}] 訊號: {setup['symbol']} ({setup['score']}分)")
    except Exception as e:
        print(f"❌ Discord 發送異常: {e}")


def send_outcome_alert(sig: dict, event_type: str, pnl_u: float, state: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    is_moon = sig.get("system") == "MOONSHOT"
    sys_tag = "🚀【系統二｜暴漲妖幣】" if is_moon else "🛡️【系統一｜穩健主單】"
    hit_time_tw = now_tw_str()
    elapsed_min = int((time.time() - sig["created_at"]) / 60)

    title_map = {
        "TP1": f"🎯 止盈 1 達標（抽回本金/推保本）| {sig['symbol']}",
        "TP2": f"🔥 止盈 2 波段達標 | {sig['symbol']}",
        "TP3": f"👑 暴漲 TP3 (+80%) 超級達標！| {sig['symbol']}",
        "TRAILING_WIN": f"💰 吊燈追蹤鎖利出場 | {sig['symbol']}",
        "BREAKEVEN": f"🛡️ 保本安全離場 | {sig['symbol']}",
        "SL": f"🛑 觸及止損出場 | {sig['symbol']}",
    }
    color_map = {
        "TP1": 0x1ABC9C,
        "TP2": 0x2ECC71,
        "TP3": 0xF1C40F,
        "TRAILING_WIN": 0x2ECC71,
        "BREAKEVEN": 0x3498DB,
        "SL": 0xE74C3C,
    }

    embed = {
        "title": f"{sys_tag} {title_map.get(event_type, event_type)}",
        "description": (
            f"🕒 **到價時間（台灣）**：`{hit_time_tw}`（發單後 {elapsed_min} 分鐘）\n"
            f"💵 **本單已實現結算損益**：**`{pnl_u:+.2f} U`**（原進場價 `{format_price(sig['entry'])}`）"
        ),
        "color": color_map.get(event_type, 0x95A5A6),
        "footer": {"text": get_footer_text(state, sig.get("system", "MAIN"))},
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "雙引擎戰報追蹤", "embeds": [embed]}, timeout=10)
    except Exception:
        pass


def send_daily_summary(state: dict, spot_tickers: dict, swap_tickers: dict):
    if not DISCORD_WEBHOOK_URL:
        return

    sm = state["stats_main"]
    smoon = state["stats_moon"]
    dst = state["daily_stats"]
    act_main = state["active_main"]
    act_moon = state["active_moon"]

    d_tot = dst["wins"] + dst["losses"]
    d_wr = f"{dst['wins']/d_tot*100:.1f}%" if d_tot > 0 else "尚無平倉"

    m_tot = sm["wins"] + sm["losses"]
    m_wr = f"{sm['wins']/m_tot*100:.1f}%" if m_tot > 0 else "尚無平倉"

    moon_tot = smoon["wins"] + smoon["losses"]
    moon_wr = f"{smoon['wins']/moon_tot*100:.1f}%" if moon_tot > 0 else "尚無平倉"

    lines = []
    for sym, sig in act_main.items():
        cur_p = float(spot_tickers.get(sym, {}).get("last") or sig["entry"])
        pct = (cur_p - sig["entry"]) / sig["entry"] * (1 if sig["side"] == "LONG" else -1) * 100
        lines.append(f"• 🛡️ **[主趨勢] {sym}** ({sig['side']}) ｜ 浮動現貨 `{pct:+.2f}%`")

    for sym, sig in act_moon.items():
        cur_p = float(swap_tickers.get(sym, {}).get("last") or sig["entry"])
        pct = (cur_p - sig["entry"]) / sig["entry"] * 100
        lines.append(f"• 🚀 **[妖幣狙擊] {sym}** (10x) ｜ 浮動現貨 **`{pct:+.2f}%`**")

    detail = "\n".join(lines) if lines else "目前雙系統皆空倉觀望中。"

    embed = {
        "title": f"📅 雙引擎每日量化戰報 | {today_tw_str()} (台灣時間)",
        "description": (
            f"🕒 **結算時間**：`{now_tw_str()}`\n"
            f"今日發出：穩健主單 **`{dst.get('main_sent', 0)}`** 筆 ｜ 妖幣狙擊 **`{dst.get('moon_sent', 0)}`** 筆\n"
            f"目前線上追蹤中：共 **`{len(act_main) + len(act_moon)}`** 單"
        ),
        "color": 0x2ECC71 if dst["net_pnl_u"] >= 0 else 0xE67E22,
        "fields": [
            {"name": "💰 今日雙系統總盈虧", "value": f"**`{dst['net_pnl_u']:+.2f} U`** (勝率 `{d_wr}`)", "inline": False},
            {"name": "🛡️ 系統一（穩健主趨勢）累計", "value": f"勝率 `{m_wr}` ｜ 淨利 **`{sm['net_pnl_u']:+.2f} U`**", "inline": True},
            {"name": "🚀 系統二（暴漲妖幣）累計", "value": f"勝率 `{moon_wr}` ｜ 淨利 **`{smoon['net_pnl_u']:+.2f} U`**", "inline": True},
            {"name": "📋 剩餘未平倉追蹤明細", "value": detail, "inline": False},
        ],
        "footer": {"text": "雙引擎量化雷達 v5.1 • 每日 23:50 自動結算"},
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "雙引擎戰報總結", "embeds": [embed]}, timeout=10)
    except Exception:
        pass


# ==========================================
# 10. 雙系統訂單即時追蹤
# ==========================================
def check_all_active_orders(exchange, state: dict):
    now = time.time()

    for sym in list(state["active_main"].keys()):
        sig = state["active_main"][sym]
        if now - sig["created_at"] > MAIN_MAX_HOURS * 3600:
            del state["active_main"][sym]
            continue
        try:
            candles = exchange.fetch_ohlcv(sym, "5m", since=int(sig.get("last_checked_ms", sig["created_at"] * 1000)), limit=200)
            if not candles:
                continue
            done = False
            tp1_part = round(sig["est_profit_tp1"] * 0.60, 2)
            tp2_part = round(sig["est_profit_tp2"] * 0.40, 2)

            for c in candles:
                _, _, h, l, _, _ = c
                if sig["side"] == "LONG":
                    if l <= sig["sl"]:
                        pnl = tp1_part if sig.get("tp1_hit") else -sig["est_loss_sl"]
                        evt = "BREAKEVEN" if sig.get("tp1_hit") else "SL"
                        record_pnl(state, "MAIN", pnl, pnl > 0)
                        send_outcome_alert(sig, evt, pnl, state)
                        del state["active_main"][sym]
                        done = True
                        break
                    if not sig.get("tp1_hit") and h >= sig["tp1"]:
                        sig["tp1_hit"] = True
                        sig["sl"] = sig["entry"] * 1.002
                        send_outcome_alert(sig, "TP1", tp1_part, state)
                    if sig.get("tp1_hit") and h >= sig["tp2"]:
                        total_win = round(tp1_part + tp2_part, 2)
                        record_pnl(state, "MAIN", total_win, True)
                        send_outcome_alert(sig, "TP2", total_win, state)
                        del state["active_main"][sym]
                        done = True
                        break
                else:
                    if h >= sig["sl"]:
                        pnl = tp1_part if sig.get("tp1_hit") else -sig["est_loss_sl"]
                        evt = "BREAKEVEN" if sig.get("tp1_hit") else "SL"
                        record_pnl(state, "MAIN", pnl, pnl > 0)
                        send_outcome_alert(sig, evt, pnl, state)
                        del state["active_main"][sym]
                        done = True
                        break
                    if not sig.get("tp1_hit") and l <= sig["tp1"]:
                        sig["tp1_hit"] = True
                        sig["sl"] = sig["entry"] * 0.998
                        send_outcome_alert(sig, "TP1", tp1_part, state)
                    if sig.get("tp1_hit") and l <= sig["tp2"]:
                        total_win = round(tp1_part + tp2_part, 2)
                        record_pnl(state, "MAIN", total_win, True)
                        send_outcome_alert(sig, "TP2", total_win, state)
                        del state["active_main"][sym]
                        done = True
                        break
            if not done and sym in state["active_main"]:
                state["active_main"][sym]["last_checked_ms"] = candles[-1][0]
        except Exception as e:
            print(f"[主系統追蹤異常] {sym}: {e}")

    for sym in list(state["active_moon"].keys()):
        sig = state["active_moon"][sym]
        if now - sig["created_at"] > MOON_MAX_HOURS * 3600:
            del state["active_moon"][sym]
            continue
        try:
            candles = exchange.fetch_ohlcv(sym, "5m", since=int(sig.get("last_checked_ms", sig["created_at"] * 1000)), limit=200)
            if not candles:
                continue
            done = False
            notional = sig["notional_usdt"]
            entry = sig["entry"]

            for c in candles:
                _, _, h, l, _, _ = c
                if l <= sig["sl"]:
                    realized = sig.get("realized_u", 0.0)
                    rem_ratio = sig.get("rem_ratio", 1.0)
                    rem_pnl = notional * rem_ratio * ((sig["sl"] - entry) / entry)
                    total_pnl = round(realized + rem_pnl - (notional * 0.001), 2)
                    evt = "TRAILING_WIN" if total_pnl > 0 else "SL"
                    record_pnl(state, "MOONSHOT", total_pnl, total_pnl > 0)
                    send_outcome_alert(sig, evt, total_pnl, state)
                    del state["active_moon"][sym]
                    done = True
                    break

                if not sig.get("tp1_hit") and h >= sig["tp1"]:
                    sig["tp1_hit"] = True
                    sig["sl"] = entry * 1.005
                    sig["rem_ratio"] = 0.70
                    sig["realized_u"] = round(sig.get("realized_u", 0.0) + notional * 0.30 * 0.15, 2)
                    send_outcome_alert(sig, "TP1", sig["realized_u"], state)

                if not sig.get("tp2_hit") and h >= sig["tp2"]:
                    sig["tp2_hit"] = True
                    sig["sl"] = max(sig["sl"], entry * 1.15)
                    sig["rem_ratio"] = 0.40
                    sig["realized_u"] = round(sig.get("realized_u", 0.0) + notional * 0.30 * 0.35, 2)
                    send_outcome_alert(sig, "TP2", sig["realized_u"], state)

                if not sig.get("tp3_hit") and h >= sig["tp3"]:
                    sig["tp3_hit"] = True
                    sig["rem_ratio"] = 0.20
                    sig["realized_u"] = round(sig.get("realized_u", 0.0) + notional * 0.20 * 0.80, 2)
                    send_outcome_alert(sig, "TP3", sig["realized_u"], state)

                if sig.get("tp2_hit"):
                    peak = max(sig.get("peak_price", entry), h)
                    sig["peak_price"] = peak
                    sig["sl"] = max(sig["sl"], peak * 0.88)

            if not done and sym in state["active_moon"]:
                state["active_moon"][sym]["last_checked_ms"] = candles[-1][0]
        except Exception as e:
            print(f"[妖幣追蹤異常] {sym}: {e}")


def record_pnl(state: dict, system: str, pnl_u: float, is_win: bool):
    target = state["stats_moon"] if system == "MOONSHOT" else state["stats_main"]
    dst = state["daily_stats"]
    if is_win:
        target["wins"] += 1
        dst["wins"] += 1
    else:
        target["losses"] += 1
        dst["losses"] += 1
    target["net_pnl_u"] = round(target["net_pnl_u"] + pnl_u, 2)
    dst["net_pnl_u"] = round(dst["net_pnl_u"] + pnl_u, 2)


# ==========================================
# 11. 雙系統同步掃描主流程
# ==========================================
def run_scan():
    exchange = ccxt.okx({"enableRateLimit": True})
    state = load_state()
    coin_history = state["coin_history"]
    current_capital = INITIAL_CAPITAL + state["stats_main"].get("net_pnl_u", 0.0)
    now = time.time()

    check_all_active_orders(exchange, state)

    spot_tickers = exchange.fetch_tickers()
    active_spot = []
    for sym, t in spot_tickers.items():
        if not sym.endswith("/USDT") or ":" in sym:
            continue
        base = sym.split("/")[0]
        if is_excluded_base(base):
            continue
        q_vol = float(t.get("quoteVolume") or 0) or (float(t.get("baseVolume") or 0) * float(t.get("last") or 0))
        if q_vol > 5_000_000:
            active_spot.append((sym, q_vol))

    active_spot.sort(key=lambda x: x[1], reverse=True)
    main_symbols = [x[0] for x in active_spot[:30]]

    try:
        ohlcv_btc = exchange.fetch_ohlcv("BTC/USDT", "4h", limit=100)
        df_btc = compute_indicators(pd.DataFrame(ohlcv_btc, columns=["ts", "open", "high", "low", "close", "volume"]))
        b4 = df_btc.iloc[-1]
        btc_4h_trend = 1 if (b4["close"] > b4["ema20"] > b4["ema50"]) else (-1 if (b4["close"] < b4["ema20"] < b4["ema50"]) else 0)
    except Exception:
        btc_4h_trend = 0

    main_candidates = []
    for sym in main_symbols:
        try:
            ohlcv_4h = exchange.fetch_ohlcv(sym, "4h", limit=120)
            if len(ohlcv_4h) < 60:
                continue
            df_4h = compute_indicators(pd.DataFrame(ohlcv_4h, columns=["ts", "open", "high", "low", "close", "volume"]))
            c4 = df_4h.iloc[-1]
            c4_trend = 1 if (c4["close"] > c4["ema50"] and c4["ema20"] > c4["ema50"]) else (-1 if (c4["close"] < c4["ema50"] and c4["ema20"] < c4["ema50"]) else 0)
            if c4_trend == 0:
                continue

            ohlcv_1h = exchange.fetch_ohlcv(sym, "1h", limit=120)
            if len(ohlcv_1h) < 60:
                continue
            df_1h = compute_indicators(pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "volume"]))

            best_s = evaluate_main_trend(sym, df_1h, df_4h, c4_trend, btc_4h_trend, current_capital)
            if best_s:
                main_candidates.append((best_s, df_1h))
                hist_key = f"MAIN_{sym}"
                mins_ago = (now - coin_history[hist_key]["last_ts"]) / 60 if hist_key in coin_history else 9999
                if best_s["score"] >= MAIN_MIN_SCORE and sym not in state["active_main"] and mins_ago >= MAIN_COOLDOWN_MIN:
                    send_discord_alert(best_s, df_1h, state)
                    state["daily_stats"]["main_sent"] += 1
                    coin_history[hist_key] = {"last_ts": now}
                    state["active_main"][sym] = {
                        **best_s,
                        "tp1_hit": False,
                        "created_at": now,
                        "last_checked_ms": int(now * 1000),
                    }
        except Exception:
            continue

    swap_tickers = exchange.fetch_tickers(params={"instType": "SWAP"})
    moon_pool = []
    for sym, t in swap_tickers.items():
        if not sym.endswith("/USDT:USDT"):
            continue
        base = sym.split("/")[0]
        if is_excluded_base(base):
            continue
        info = t.get("info", {})
        inst_id = info.get("instId", f"{base}-USDT-SWAP")
        vol_ccy = float(info.get("volCcy24h") or 0) * float(t.get("last") or 0)
        pct_24h = float(t.get("percentage") or 0)
        if 1_500_000 <= vol_ccy <= 120_000_000 and pct_24h > 3.0:
            moon_pool.append((sym, inst_id, pct_24h))

    moon_pool.sort(key=lambda x: x[2], reverse=True)
    moon_candidates = []

    for sym, inst_id, _ in moon_pool[:35]:
        try:
            ohlcv_1h = exchange.fetch_ohlcv(sym, "1h", limit=100)
            if len(ohlcv_1h) < 60:
                continue
            df_1h = compute_indicators(pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "volume"]))
            m_setup = evaluate_moonshot(sym, inst_id, df_1h)
            if m_setup:
                moon_candidates.append((m_setup, df_1h))
                hist_key = f"MOON_{sym}"
                mins_ago = (now - coin_history[hist_key]["last_ts"]) / 60 if hist_key in coin_history else 9999
                if m_setup["score"] >= MOON_MIN_SCORE and sym not in state["active_moon"] and mins_ago >= MOON_COOLDOWN_MIN:
                    send_discord_alert(m_setup, df_1h, state)
                    state["daily_stats"]["moon_sent"] += 1
                    coin_history[hist_key] = {"last_ts": now}
                    state["active_moon"][sym] = {
                        **m_setup,
                        "tp1_hit": False,
                        "tp2_hit": False,
                        "tp3_hit": False,
                        "rem_ratio": 1.0,
                        "realized_u": 0.0,
                        "peak_price": m_setup["entry"],
                        "created_at": now,
                        "last_checked_ms": int(now * 1000),
                    }
        except Exception:
            continue

    print(f"📊 [{now_tw_str()}] 雙系統掃描完成 | 主趨勢追蹤: {len(state['active_main'])} 單 | 妖幣狙擊追蹤: {len(state['active_moon'])} 單")

    if FORCE_TEST:
        if main_candidates:
            main_candidates.sort(key=lambda x: x[0]["score"], reverse=True)
            send_discord_alert(main_candidates[0][0], main_candidates[0][1], state)
        if moon_candidates:
            moon_candidates.sort(key=lambda x: x[0]["score"], reverse=True)
            send_discord_alert(moon_candidates[0][0], moon_candidates[0][1], state)

    tw_now = now_tw()
    today_s = today_tw_str()
    if (tw_now.hour == 23 and tw_now.minute >= 48 and state.get("last_daily_report_date") != today_s) or FORCE_DAILY_REPORT:
        send_daily_summary(state, spot_tickers, swap_tickers)
        state["last_daily_report_date"] = today_s

    save_state(state)


if __name__ == "__main__":
    run_scan()
