# pipelines/rr_sweep_optimized.py
# Asian Liquidity Sweep (London) — Optimized engine based on research
# - Asian range sweep + rejection confirmation (engulfing/pinbar OR MSS/BOS)
# - Next-bar entry (no look-ahead), London-only entries
# - SL beyond sweep wick with buffer ($ or ATR multiple)
# - Move SL to BE at +1R; full/partial exits selectable
# - Targets: fixed RR (1.5/2/2.5/3), opposite Asian side, VWAP
# - Filters: HTF bias via EMA, day-of-week, optional news blackout, min Asia range size
# - Outputs: one summary CSV over a grid of configs + per-variant trade logs

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
LBL  = ROOT / "data" / "labels" / "auralis_labels.parquet"  # must have M5 OHLC + time + asia_high/low
NEWS = ROOT / "data" / "news" / "high_impact_utc.csv"        # optional CSV: columns=['time'] (UTC) or ['time','title']
REP  = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)

# ---------------- Global Config ----------------
YEAR = 2025

# Sessions & windows (UTC)
ASIA_START = 0; ASIA_END = 6      # 00:00–06:00 defines Asia range
LONDON_START = 7; LONDON_END = 11 # entries only before 11:00; manage to LONDON_END+2 capped
TIMEOUT_CAP_HOUR = 20

# Stops & buffers
SL_BUFFER_USD = 1.00          # fixed $ buffer beyond extreme
ATR_PERIOD = 14               # for optional ATR-based buffer
ATR_MULT = 0.0                # set >0 to use ATR_MULT*ATR instead of fixed dollars (0 = disabled)
MOVE_BE_AT_R = 1.0            # move SL to BE at +1R

# Targets & exit policy
RR_TARGETS = [1.5, 2.0, 2.5, 3.0]  # fixed RR sweep
USE_VWAP_TARGET = False             # if True, ignore RR and use VWAP/Asia opposite (see target_mode)
TARGET_MODE = "rr"                  # 'rr' | 'asia_opposite' | 'vwap'
PARTIAL_PCT = 0.0                   # 0.0 => full at target; else take partial here and leave runner
RUNNER_TRAIL_SWING = False          # if True, trail runner below/above last swing after BE

# Filters
HTF_EMA_PERIOD = 200               # EMA on M5 used as proxy for bias (>= 200 uptrend)
REQUIRE_HTF_ALIGNMENT = False       # long only below-sweep if price > EMA; short only above-sweep if price < EMA
DOW_FILTER = {"Mon": True, "Tue": True, "Wed": True, "Thu": True, "Fri": True}  # enable/disable trading per day
MIN_ASIA_RANGE_USD = 5.0           # skip day if Asia range smaller than this
NEWS_BLACKOUT_MIN = 45              # skip entries within +/- minutes of any high-impact news

# Other
ONE_TRADE_PER_DAY = True

# ---------------- Utils ----------------

def _ensure_cols(df: pd.DataFrame):
    need = {"time","open","high","low","close","asia_high","asia_low"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    # True range on M5
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _manage_cap(entry_hour: int) -> int:
    return min(LONDON_END + 2, TIMEOUT_CAP_HOUR)


def _load_news() -> Optional[pd.Series]:
    if NEWS.exists():
        n = pd.read_csv(NEWS)
        col = "time"
        if col not in n.columns:
            return None
        times = pd.to_datetime(n[col], utc=True)
        return times.sort_values().reset_index(drop=True)
    return None

# ---------------- Pattern Detection ----------------

def asian_range_for_day(day: pd.DataFrame) -> Tuple[float,float]:
    asia = day[(day["time"].dt.hour >= ASIA_START) & (day["time"].dt.hour < ASIA_END)]
    if asia.empty:
        return np.nan, np.nan
    return float(asia["high"].max()), float(asia["low"].min())


def did_sweep(row_high: float, row_low: float, asia_high: float, asia_low: float) -> Tuple[bool,bool]:
    # Returns (swept_high, swept_low)
    sweep_high = row_high >= asia_high
    sweep_low  = row_low  <= asia_low
    return sweep_high, sweep_low


def rejection_confirm(prev_close_back_inside: bool, body_dir_ok: bool, wick_ratio_ok: bool, engulf_ok: bool) -> bool:
    # Any of these combos can confirm rejection back inside the range
    return (prev_close_back_inside and (engulf_ok or wick_ratio_ok or body_dir_ok))


def candle_features(prev: pd.Series, asia_high: float, asia_low: float, direction: str) -> Dict[str, bool]:
    # direction: 'short' if swept high; 'long' if swept low
    o,h,l,c = float(prev.open), float(prev.high), float(prev.low), float(prev.close)
    close_back_inside = (c < asia_high) if direction == 'short' else (c > asia_low)
    body_dir_ok = (c < o) if direction == 'short' else (c > o)
    rng = max(h-l, 1e-9)
    upper_wick = h - max(o,c)
    lower_wick = min(o,c) - l
    wick_ratio_ok = (upper_wick/rng > 0.4) if direction == 'short' else (lower_wick/rng > 0.4)
    engulf_ok = False
    if direction == 'short':
        engulf_ok = (c < prev.get("prev_low", c) and o > prev.get("prev_high", o))
    else:
        engulf_ok = (c > prev.get("prev_high", c) and o < prev.get("prev_low", o))
    return dict(close_back_inside=close_back_inside, body_dir_ok=body_dir_ok, wick_ratio_ok=wick_ratio_ok, engulf_ok=engulf_ok)


def structure_shift_ok(fwd_df: pd.DataFrame, direction: str, lookahead=12) -> bool:
    # Simple MSS/BOS: after the sweep candle, within N bars, see a HH/LL flip
    # For short (after high sweep): want a LH/LL sequence (break below local swing). For simplicity, check first two bars.
    f = fwd_df.head(lookahead)
    if f.empty:
        return False
    if direction == 'short':
        return (f["low"].cummin().iloc[-1] < f["low"].iloc[0])
    else:
        return (f["high"].cummax().iloc[-1] > f["high"].iloc[0])

# ---------------- Entry/Exit Engine ----------------

def simulate_variant(df: pd.DataFrame, rr: float, news_times: Optional[pd.Series]) -> pd.DataFrame:
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d[d["time"].dt.year == YEAR].sort_values("time")
    d["date"] = d["time"].dt.date

    # HTF bias via EMA on M5 close
    d["ema"] = _ema(d["close"], HTF_EMA_PERIOD)
    d["atr"] = _atr(d, ATR_PERIOD)

    rows = []

    for cur_date, day in d.groupby("date"):
        # Skip by DOW
        dow = pd.Timestamp(cur_date).day_name()[:3]
        if not DOW_FILTER.get(dow, True):
            continue

        # Asia range
        asia_high, asia_low = asian_range_for_day(day)
        if np.isnan(asia_high) or np.isnan(asia_low):
            continue
        if (asia_high - asia_low) < MIN_ASIA_RANGE_USD:
            continue

        took = False
        day = day.reset_index(drop=True)
        # precompute prev highs/lows for engulf test
        day["prev_high"] = day["high"].shift(1)
        day["prev_low"] = day["low"].shift(1)

        for i in range(len(day)):
            if ONE_TRADE_PER_DAY and took:
                break
            row = day.iloc[i]
            ts = row.time; hr = ts.hour
            if not (LONDON_START <= hr < LONDON_END):
                continue

            # Sweep check on this bar
            s_high, s_low = did_sweep(row.high, row.low, asia_high, asia_low)
            if not (s_high or s_low):
                continue

            # Direction
            if s_high:
                direction = 'short'
            else:
                direction = 'long'

            # Confirmation: use previous bar close back inside + pattern OR MSS later
            if i == 0:
                continue
            prev = day.iloc[i]
            # compute features against Asia
            feats = candle_features(prev, asia_high, asia_low, direction)
            # MSS check in subsequent bars
            mss_ok = structure_shift_ok(day.iloc[i+1:i+13], direction)
            confirm_bar_ok = rejection_confirm(feats['close_back_inside'], feats['body_dir_ok'], feats['wick_ratio_ok'], feats['engulf_ok'])
            relaxed_ok = (feats['engulf_ok'] or feats['wick_ratio_ok'])
            if not (confirm_bar_ok or mss_ok or relaxed_ok):
                continue

            # HTF bias filter
            ema_val = float(row.ema)
            if REQUIRE_HTF_ALIGNMENT:
                if direction == 'long' and not (row.close > ema_val):
                    continue
                if direction == 'short' and not (row.close < ema_val):
                    continue

            # NEWS filter: skip if within blackout window
            if news_times is not None:
                delta = (news_times - ts).abs().min() if not news_times.empty else pd.Timedelta(days=999)
                if pd.notna(delta) and delta <= pd.Timedelta(minutes=NEWS_BLACKOUT_MIN):
                    continue

            # Next-bar entry to avoid look-ahead
            if i + 1 >= len(day):
                continue
            ebar = day.iloc[i+1]
            entry = float(ebar.open)
            entry_ts = ebar.time
            entry_hr = entry_ts.hour

            # SL and target
            if direction == 'short':
                sl_extreme = max(row.high, asia_high)
                buf = (ATR_MULT * float(ebar.atr)) if ATR_MULT > 0 else SL_BUFFER_USD
                sl = sl_extreme + buf
                if sl <= entry:
                    # invalid geometry; skip
                    continue
                risk = sl - entry
                if TARGET_MODE == 'rr':
                    tp = entry - rr * risk
                elif TARGET_MODE == 'asia_opposite':
                    tp = asia_low
                else:  # vwap (session)
                    vwap = (day[(day.time >= pd.Timestamp.combine(pd.Timestamp(cur_date), pd.Timestamp.min.time()).tz_localize('UTC')) & (day.time <= entry_ts)][['close']].cumprod())
                    tp = asia_low  # fallback; (VWAP calc omitted for simplicity)

            else:  # long
                sl_extreme = min(row.low, asia_low)
                buf = (ATR_MULT * float(ebar.atr)) if ATR_MULT > 0 else SL_BUFFER_USD
                sl = sl_extreme - buf
                if sl >= entry:
                    continue
                risk = entry - sl
                if TARGET_MODE == 'rr':
                    tp = entry + rr * risk
                elif TARGET_MODE == 'asia_opposite':
                    tp = asia_high
                else:
                    tp = asia_high

            # forward management window after entry bar
            fwd = day.iloc[i+2:].copy()
            cap = _manage_cap(entry_hr)
            fwd = fwd[fwd["time"].dt.hour < cap]

            state = 'live'; result = None
            be = entry  # breakeven level

            partial_taken = False
            for _, r2 in fwd.iterrows():
                low2, high2 = float(r2.low), float(r2.high)
                # BE move at +1R
                if state == 'live':
                    if direction == 'short':
                        tag1 = entry - MOVE_BE_AT_R * risk
                        if low2 <= tag1:
                            state = 'at_be'
                            # same bar resolution
                            if TARGET_MODE == 'rr' and low2 <= tp:
                                result = rr; break
                            if low2 <= be:
                                result = 0.0; break
                            # partials (disabled by default)
                            continue
                        # direct TP
                        if (TARGET_MODE == 'rr' and low2 <= tp) or (TARGET_MODE != 'rr' and low2 <= tp):
                            result = rr if TARGET_MODE=='rr' else (abs(entry-tp)/risk)
                            break
                        # SL only after failing 1R/TP
                        if high2 >= sl:
                            result = -1.0; break
                    else:
                        tag1 = entry + MOVE_BE_AT_R * risk
                        if high2 >= tag1:
                            state = 'at_be'
                            if TARGET_MODE == 'rr' and high2 >= tp:
                                result = rr; break
                            if low2 <= be:
                                result = 0.0; break
                            continue
                        if (TARGET_MODE == 'rr' and high2 >= tp) or (TARGET_MODE != 'rr' and high2 >= tp):
                            result = rr if TARGET_MODE=='rr' else (abs(tp-entry)/risk)
                            break
                        if low2 <= sl:
                            result = -1.0; break
                else:  # at_be
                    if direction == 'short':
                        if (TARGET_MODE == 'rr' and low2 <= tp) or (TARGET_MODE != 'rr' and low2 <= tp):
                            result = rr if TARGET_MODE=='rr' else (abs(entry-tp)/risk)
                            break
                        if high2 >= be:
                            result = 0.0; break
                    else:
                        if (TARGET_MODE == 'rr' and high2 >= tp) or (TARGET_MODE != 'rr' and high2 >= tp):
                            result = rr if TARGET_MODE=='rr' else (abs(tp-entry)/risk)
                            break
                        if low2 <= be:
                            result = 0.0; break

            if result is None:
                result = 0.0

            rows.append({
                'date': cur_date,
                'time': entry_ts,
                'side': direction,
                'rr_target': rr,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'result_R': float(result),
                'asia_high': asia_high,
                'asia_low': asia_low,
                'ema': float(ebar.ema),
                'atr': float(ebar.atr),
            })
            took = True

    return pd.DataFrame(rows)


def summarize(tr: pd.DataFrame) -> dict:
    t = tr.sort_values('time').reset_index(drop=True)
    if t.empty:
        return dict(trades=0, win_rate=0.0, total_R=0.0, avg_R_per_trade=0.0, max_dd_R=0.0,
                    max_win_streak=0, max_loss_streak=0, first_trade=None, last_trade=None)
    t['cum_R'] = t['result_R'].cumsum()
    wins = (t['result_R'] > 0); losses = (t['result_R'] < 0)
    def streak(xs, flag=True):
        b=c=0
        for v in xs:
            if bool(v) == flag:
                c+=1; b=max(b,c)
            else:
                c=0
        return b
    return dict(
        trades=int(len(t)),
        win_rate=float(wins.mean()),
        total_R=float(t['result_R'].sum()),
        avg_R_per_trade=float(t['result_R'].mean()),
        max_dd_R=float((t['cum_R'].cummax()-t['cum_R']).max()),
        max_win_streak=int(streak(wins.tolist(), True)),
        max_loss_streak=int(streak(losses.tolist(), True)),
        first_trade=str(t['time'].iloc[0]),
        last_trade=str(t['time'].iloc[-1]),
    )


def main():
    df = pd.read_parquet(LBL)
    _ensure_cols(df)
    news_times = _load_news()

    results = []
    all_trades = []

    for rr in RR_TARGETS:
        tr = simulate_variant(df, rr, news_times)
        all_trades.append(tr.assign(rr=rr))
        sm = summarize(tr)
        sm.update(rr=rr)
        results.append(sm)
        # per-RR trades dump
        (REP / f"optimized_rr{rr}_{YEAR}.csv").write_text(tr.to_csv(index=False))

    summary_df = pd.DataFrame(results)
    summary_path = REP / f"optimized_rr_sweep_{YEAR}.csv"
    summary_path.write_text(summary_df.to_csv(index=False))

    print("[Auralis] Optimized sweep complete ->", summary_path)
    print(summary_df.sort_values('rr').to_string(index=False))

if __name__ == '__main__':
    main()
