# pipelines/backtest_baseline_optimized.py
# Baseline backtester — same logic as your original, with safe toggles to optimize without changing defaults.
# Defaults originally preserved your behaviour (London 07–11, entry at close, 80% at 2R, runner to BE, $1 SL buffer).
# This version is tuned for 5-year reality: full history, next-bar entry, EMA bias, Asia-range filters.
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Make sure project root is on sys.path so 'models' can be imported
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.xgb_filter import XGBTradeFilter  # <-- now this will work

LBL = ROOT / "data" / "labels" / "auralis_labels.parquet"
OUT = ROOT / "reports" / "baseline_2018_2025_baseline.csv"

# ======= Config (5-year baseline) =======
LONDON_START = 8   # 07:00 UTC
LONDON_END   = 10  # entries allowed before 11:00
TIMEOUT_CAP_HOUR = 14  # runner managed no later than 20:00 UTC
RR_TARGET = 2.0
PARTIAL_PCT = 0.80
SL_BUFFER = 1.0     # $1.0 beyond wick/asia level
YEAR_FILTER = None  # use full dataset (all years)
P_THRESHOLD = 0.6  # default; can be overridden from outside (run_experiments.py)

# ---- Optional switches (we turn some ON for a realistic baseline) ----
USE_NEXT_BAR_ENTRY = True       # enter on next bar open instead of signal close
USE_RETRACE_ENTRY  = False      # if True, use a limit at 50% of signal candle; waits for fill
RETRACE_PCT        = 0.50       # 50% retrace toward signal candle’s opposite extreme
RETRACE_FILL_MAX_BARS = 6       # give up if not filled within N bars after signal

USE_EMA_BIAS = False            # simple HTF bias: long only if close>EMA; short only if close<EMA
EMA_PERIOD   = 200

ASIA_RANGE_MIN = 3            # skip ultra-tight Asia ranges
ASIA_RANGE_MAX = 7          # skip very wide Asia ranges (wild days)

INTRABAR_PRIORITY = "sl_first" # 'sl_first' (conservative) or 'tp_first' (optimistic) when both hit same bar before partial
ONE_TRADE_PER_DAY = True

# ============================================================

def in_london(ts_series):
    hrs = pd.to_datetime(ts_series, utc=True).dt.hour
    return (hrs >= LONDON_START) & (hrs < LONDON_END)


def asia_range_guard(day, tcol):
    """Return False if Asia range is outside [ASIA_RANGE_MIN, ASIA_RANGE_MAX] when these are set."""
    asia = day[(day[tcol].dt.hour >= 0) & (day[tcol].dt.hour < 6)]
    if asia.empty:
        return False
    a_hi = float(asia["high"].max())
    a_lo = float(asia["low"].min())
    rng = a_hi - a_lo
    if ASIA_RANGE_MIN is not None and rng < ASIA_RANGE_MIN:
        return False
    if ASIA_RANGE_MAX is not None and rng > ASIA_RANGE_MAX:
        return False
    return True


def intrabar_hit(low, high, sl, tp):
    """
    Determine if SL/TP hits in a single bar when both levels lie within [low, high].
    Returns "sl", "tp", or None.
    """
    if sl is None or tp is None:
        return None
    if low <= sl <= high and low <= tp <= high:
        if INTRABAR_PRIORITY == "sl_first":
            return "sl"
        elif INTRABAR_PRIORITY == "tp_first":
            return "tp"
        else:
            return "sl"
    elif low <= sl <= high:
        return "sl"
    elif low <= tp <= high:
        return "tp"
    return None


def add_ema(df, price_col="close", period=EMA_PERIOD):
    col = f"ema_{period}"
    if col not in df.columns:
        df[col] = df[price_col].ewm(span=period, adjust=False).mean()
    return col


def apply_ema_bias(row, side, ema_col="ema_200"):
    if not USE_EMA_BIAS:
        return True
    ema_val = row.get(ema_col, np.nan)
    if pd.isna(ema_val):
        return False
    if side == "long" and row["close"] <= ema_val:
        return False
    if side == "short" and row["close"] >= ema_val:
        return False
    return True


def get_entry_price(row, next_row, side):
    """
    Determine entry price based on config:
    - Next-bar market
    - or retrace limit
    """
    if USE_RETRACE_ENTRY:
        # Limit at RETRACE_PCT of signal candle toward opposite extreme
        hi, lo, op, cl = float(row["high"]), float(row["low"]), float(row["open"]), float(row["close"])
        if side == "long":
            # retrace toward low from close/open
            base = min(op, cl)
            target = base - RETRACE_PCT * (base - lo)
        else:
            base = max(op, cl)
            target = base + RETRACE_PCT * (hi - base)
        return "limit", target
    else:
        if USE_NEXT_BAR_ENTRY and next_row is not None:
            return "market_next", float(next_row["open"])
        else:
            return "market_close", float(row["close"])


def main():
    df = pd.read_parquet(LBL).copy()
    tcol_candidates = [c for c in df.columns if "time" in c.lower() or c.lower() == "timestamp"]
    if not tcol_candidates:
        raise ValueError("No time column found in labels file.")
    tcol = tcol_candidates[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)

    if YEAR_FILTER:
        df = df[df[tcol].dt.year == YEAR_FILTER]

    # Ensure sorted
    df = df.sort_values(tcol).reset_index(drop=True)

    # Add EMA if needed
    ema_col = None
    if USE_EMA_BIAS:
        ema_col = add_ema(df, price_col="close", period=EMA_PERIOD)

    # Extract date column for grouping
    df["date"] = df[tcol].dt.date

    trades = []
    xgb_filter = XGBTradeFilter()

    for date, day in df.groupby("date"):
        day = day.sort_values(tcol).reset_index(drop=True)

        # Asia-range filter (works on full day, including Asia)
        if not asia_range_guard(day, tcol):
            continue

        took_trade_today = False

        # Pre-slice London window
        mask_london = in_london(day[tcol])
        london = day[mask_london].copy()
        if london.empty:
            continue

        for i, row in london.iterrows():
            # Enforce one trade per day if enabled
            if ONE_TRADE_PER_DAY and took_trade_today:
                break

            # Determine side from label signals
            long_sig = bool(row.get("entry_long", False) and row.get("confirm_long", False))
            short_sig = bool(row.get("entry_short", False) and row.get("confirm_short", False))

            if long_sig and short_sig:
                # both signals at same bar → skip to avoid conflict
                continue
            elif long_sig:
                side = "long"
            elif short_sig:
                side = "short"
            else:
                continue

            # EMA bias filter
            if not apply_ema_bias(row, side, ema_col):
                continue
            # ---- XGBoost filter ----
            asia_high = float(row.get("asia_high", row["high"]))
            asia_low  = float(row.get("asia_low",  row["low"]))
            asia_range = asia_high - asia_low
            t_entry = row[tcol]

            feat = {
                "asia_range": asia_range,
                "hour": t_entry.hour,
                "dow": t_entry.weekday(),
                "price_close": float(row["close"]),
            }
            # Optional indicators if you added them to signals
            for col in ["ema_200", "atr_14", "rsi_14"]:
                if col in row.index:
                    val = row[col]
                    feat[col] = float(val) if not pd.isna(val) else 0.0

            p_win = xgb_filter.score(feat)
            if p_win < P_THRESHOLD:
                continue
            # ---- END XGBoost filter ----

            # Determine SL based on Asia levels vs wick
            asia_high = float(row.get("asia_high", row["high"]))
            asia_low  = float(row.get("asia_low",  row["low"]))

            if side == "long":
                wick_low = float(row["low"])
                sl_level = min(wick_low, asia_low) - SL_BUFFER
            else:
                wick_high = float(row["high"])
                sl_level = max(wick_high, asia_high) + SL_BUFFER

            # Choose entry
            next_row = london.iloc[i+1] if (i+1 < len(london)) else None
            entry_mode, entry_candidate = get_entry_price(row, next_row, side)

            # Build forward data (from signal or next bar index depending on mode)
            if entry_mode == "market_close":
                entry_price = entry_candidate
                risk = (entry_price - sl_level) if side == "long" else (sl_level - entry_price)
                if risk <= 0:
                    continue
                tp2 = entry_price + RR_TARGET * risk if side == "long" else entry_price - RR_TARGET * risk
                forward = day.loc[day.index > row.name].copy()
            elif entry_mode == "market_next":
                if next_row is None:
                    continue
                entry_price = entry_candidate
                risk = (entry_price - sl_level) if side == "long" else (sl_level - entry_price)
                if risk <= 0:
                    continue
                tp2 = entry_price + RR_TARGET * risk if side == "long" else entry_price - RR_TARGET * risk
                forward = day.loc[day.index > next_row.name].copy()
            else:  # limit retrace
                # simulate fill on subsequent bars
                forward_all = day.loc[day.index > row.name].copy()
                filled = False
                fill_price = None
                fill_idx = None
                for j, r2 in forward_all.iterrows():
                    low2, high2 = float(r2["low"]), float(r2["high"])
                    if side == "long" and low2 <= entry_candidate <= high2:
                        filled = True; fill_price = entry_candidate; fill_idx = j; break
                    if side == "short" and low2 <= entry_candidate <= high2:
                        filled = True; fill_price = entry_candidate; fill_idx = j; break
                if not filled:
                    continue
                entry_price = float(fill_price)
                if side == "long":
                    risk = entry_price - sl_level
                    tp2  = entry_price + RR_TARGET * risk
                else:
                    risk = sl_level - entry_price
                    tp2  = entry_price - RR_TARGET * risk
                if risk <= 0:
                    continue
                forward = forward_all.loc[forward_all.index > fill_idx].copy()

            # Now simulate forward bar-by-bar for partial at 2R + runner to BE/timeout
            if forward.empty:
                continue

            # Timeout limit
            timeout_hour = min(LONDON_END + 2, TIMEOUT_CAP_HOUR)

            partial_hit = False
            partial_time = None
            final_exit_time = None
            total_R = 0.0

            # TP level for partial
            tp_partial = entry_price + RR_TARGET * risk if side == "long" else entry_price - RR_TARGET * risk
            sl_working = sl_level

            for _, r2 in forward.iterrows():
                t2 = r2[tcol]
                hour2 = t2.hour
                if hour2 >= timeout_hour:
                    # close at market
                    close_px = float(r2["close"])
                    if not partial_hit:
                        # entire position closed at market
                        if side == "long":
                            total_R = (close_px - entry_price) / risk
                        else:
                            total_R = (entry_price - close_px) / risk
                        final_exit_time = t2
                    else:
                        # runner closed at market
                        if side == "long":
                            runner_R = (close_px - entry_price) / risk
                        else:
                            runner_R = (entry_price - close_px) / risk
                        total_R = PARTIAL_PCT * RR_TARGET + (1 - PARTIAL_PCT) * runner_R
                        final_exit_time = t2
                    break

                high2 = float(r2["high"])
                low2  = float(r2["low"])

                if not partial_hit:
                    # Check intrabar hits
                    intrabar = intrabar_hit(low2, high2, sl_working, tp_partial)
                    if intrabar == "sl":
                        # full position stopped
                        total_R = -1.0
                        final_exit_time = t2
                        break
                    elif intrabar == "tp":
                        # partial profit
                        partial_hit = True
                        partial_time = t2
                        # realized on partial
                        realized_R = PARTIAL_PCT * RR_TARGET
                        # move SL to BE for runner
                        sl_working = entry_price
                        # continue for runner
                        continue

                    # If only one side in range
                    if low2 <= sl_working <= high2:
                        total_R = -1.0
                        final_exit_time = t2
                        break
                    if low2 <= tp_partial <= high2:
                        partial_hit = True
                        partial_time = t2
                        realized_R = PARTIAL_PCT * RR_TARGET
                        sl_working = entry_price
                        continue
                else:
                    # Runner only; SL at BE
                    intrabar = intrabar_hit(low2, high2, sl_working, None)
                    if intrabar == "sl":
                        # runner stopped at BE
                        total_R = PARTIAL_PCT * RR_TARGET  # only partial survived
                        final_exit_time = t2
                        break

                    if side == "long":
                        if low2 <= sl_working <= high2:
                            total_R = PARTIAL_PCT * RR_TARGET
                            final_exit_time = t2
                            break
                    else:
                        if low2 <= sl_working <= high2:
                            total_R = PARTIAL_PCT * RR_TARGET
                            final_exit_time = t2
                            break

            # If no explicit exit before loop end, force close at last bar
            if final_exit_time is None:
                last = forward.iloc[-1]
                close_px = float(last["close"])
                if not partial_hit:
                    if side == "long":
                        total_R = (close_px - entry_price) / risk
                    else:
                        total_R = (entry_price - close_px) / risk
                    final_exit_time = last[tcol]
                else:
                    if side == "long":
                        runner_R = (close_px - entry_price) / risk
                    else:
                        runner_R = (entry_price - close_px) / risk
                    total_R = PARTIAL_PCT * RR_TARGET + (1 - PARTIAL_PCT) * runner_R
                    final_exit_time = last[tcol]

            took_trade_today = True

            trades.append({
                "date": date,
                "entry_time": row[tcol],
                "exit_time": final_exit_time,
                "side": side,
                "entry_price": entry_price,
                "sl": sl_level,
                "tp2": tp2,
                "total_R": total_R,
                "asia_high": asia_high,
                "asia_low": asia_low,
            })

    if not trades:
        print("[Auralis] No trades generated.")
        return

    res = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    res["cum_R"] = res["total_R"].cumsum()
    roll_max = res["cum_R"].cummax()
    dd = (roll_max - res["cum_R"]).max()
    res.to_csv(OUT, index=False)
    
    wins = (res["total_R"] > 0).sum()
    print(f"[Auralis] baseline trades: {len(res)}")
    print(f"[Auralis] win rate: {wins/len(res):.1%}")
    print(f"[Auralis] total R: {res['total_R'].sum():.2f}")
    print(f"[Auralis] max DD (R): {dd:.2f}")
    print(f"[Auralis] saved -> {OUT}")

    # Add yearly breakdown
    res["year"] = pd.to_datetime(res["entry_time"]).dt.year

    yearly = res.groupby("year")["total_R"].agg(
        trades="count",
        total_R="sum",
        avg_R="mean",
        winrate=lambda s: (s > 0).mean()
    )

    print("\n[Yearly breakdown]")
    print(yearly)

if __name__ == "__main__":
    main()
