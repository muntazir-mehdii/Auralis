# pipelines/backtest_baseline_optimized.py
# Baseline backtester — same logic as your original, with safe toggles to optimize without changing defaults.
# Defaults preserve your current behaviour (London 07–11, entry at close, 80% at 2R, runner to BE, $1 SL buffer).
# Optional upgrades you can flip on: next‑bar entry (no look‑ahead), 50% retrace limit fill, EMA bias, Asia‑range guards.

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "labels" / "auralis_labels.parquet"
OUT = ROOT / "reports" / "baseline_2025_optimized.csv"

# ======= Config (defaults match your posted baseline) =======
LONDON_START = 7   # 07:00 UTC
LONDON_END   = 11  # entries allowed before 11:00
TIMEOUT_CAP_HOUR = 20  # runner managed no later than 20:00 UTC
RR_TARGET = 2.0
PARTIAL_PCT = 0.80
SL_BUFFER = 1.0     # $1.0 beyond wick/asia level
YEAR_FILTER = 2025

# ---- New optional switches (OFF by default to keep results comparable) ----
USE_NEXT_BAR_ENTRY = False      # enter on next bar open instead of signal close
USE_RETRACE_ENTRY  = False      # if True, use a limit at 50% of signal candle; waits for fill
RETRACE_PCT        = 0.50       # 50% retrace toward signal candle’s opposite extreme
RETRACE_FILL_MAX_BARS = 6       # give up if not filled within N bars after signal

USE_EMA_BIAS = False            # simple HTF bias: long only if close>EMA; short only if close<EMA
EMA_PERIOD   = 200

ASIA_RANGE_MIN = None           # e.g. 5.0 to skip tiny ranges
ASIA_RANGE_MAX = None           # e.g. 30.0 to skip too‑wide ranges

INTRABAR_PRIORITY = "sl_first" # 'sl_first' (conservative) or 'tp_first' (optimistic) when both hit same bar before partial
ONE_TRADE_PER_DAY = True

# ============================================================

def in_london(ts_series):
    hrs = pd.to_datetime(ts_series, utc=True).dt.hour
    return (hrs >= LONDON_START) & (hrs < LONDON_END)


def asia_range_guard(day: pd.DataFrame, tcol: str) -> bool:
    """Return True if day passes optional Asia range guards, else False."""
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


def get_entry_price(row, next_row, side: str):
    """Compute entry price per settings. Defaults to signal close. If next‑bar entry is on, use next bar open.
    If retrace entry is enabled, compute a limit and indicate we must wait for a fill.
    Returns (mode, price_or_limit), where mode in {'market','limit'}.
    """
    if USE_RETRACE_ENTRY:
        # 50% toward the wick extreme on the signal candle
        o, h, l, c = map(float, [row["open"], row["high"], row["low"], row["close"]])
        if side == "long":
            limit = c - RETRACE_PCT * (c - l)
        else:
            limit = c + RETRACE_PCT * (h - c)
        return "limit", limit

    if USE_NEXT_BAR_ENTRY and next_row is not None:
        return "market", float(next_row["open"])  # next bar open

    return "market", float(row["close"])  # signal close (original behaviour)


def main():
    df = pd.read_parquet(LBL).copy()
    tcol = "time" if "time" in df.columns else [c for c in df.columns if "time" in c][0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    if YEAR_FILTER:
        df = df[df[tcol].dt.year == YEAR_FILTER]
    df = df.sort_values(tcol).reset_index(drop=True)
    df["date"] = df[tcol].dt.date

    need = {"open","high","low","close","asia_high","asia_low","entry_long","entry_short","confirm_long","confirm_short"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labels: {missing}")

    df["long_sig"]  = df["entry_long"]  & df["confirm_long"]  & in_london(df[tcol])
    df["short_sig"] = df["entry_short"] & df["confirm_short"] & in_london(df[tcol])

    if USE_EMA_BIAS:
        df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean()

    trades = []

    for d, day in df.groupby("date"):
        day = day.reset_index(drop=True)
        if (ASIA_RANGE_MIN is not None) or (ASIA_RANGE_MAX is not None):
            if not asia_range_guard(day, tcol):
                continue

        took_trade = False
        for i, row in day.iterrows():
            if ONE_TRADE_PER_DAY and took_trade:
                break
            ts = row[tcol]
            if not (LONDON_START <= ts.hour < LONDON_END):
                continue

            # Decide side
            side = None
            if bool(row["long_sig"]) and not bool(row["short_sig"]):
                side = "long"
            elif bool(row["short_sig"]) and not bool(row["long_sig"]):
                side = "short"
            elif bool(row["long_sig"]) and bool(row["short_sig"]):
                # If both fire on same bar, skip (keeps behaviour deterministic)
                continue
            else:
                continue

            # Optional EMA bias
            if USE_EMA_BIAS and not pd.isna(row.get("ema", np.nan)):
                if side == "long" and not (row["close"] > row["ema"]):
                    continue
                if side == "short" and not (row["close"] < row["ema"]):
                    continue

            next_row = day.iloc[i+1] if i+1 < len(day) else None
            mode, entry_candidate = get_entry_price(row, next_row, side)

            # Compute SL & TP based on signal context (same as your baseline)
            if side == "long":
                entry_ref = float(row["close"]) if mode == "market" else float(row["close"])  # reference for geometry checks
                sl = min(float(row["low"]), float(row["asia_low"])) - SL_BUFFER
                if sl >= entry_ref:
                    continue
                risk_at_close = entry_ref - sl
                tp2_at_close  = entry_ref + RR_TARGET * risk_at_close
            else:
                entry_ref = float(row["close"]) if mode == "market" else float(row["close"])  # reference for geometry checks
                sl = max(float(row["high"]), float(row["asia_high"])) + SL_BUFFER
                if sl <= entry_ref:
                    continue
                risk_at_close = sl - entry_ref
                tp2_at_close  = entry_ref - RR_TARGET * risk_at_close

            timeout_hour = min(LONDON_END + 2, TIMEOUT_CAP_HOUR)
            forward_all = day.iloc[i+1:].copy()
            forward_all = forward_all[forward_all[tcol].dt.hour < timeout_hour]
            if forward_all.empty:
                continue

            # ---- Handle retrace limit fill BEFORE we consider SL/TP ----
            if mode == "limit":
                filled = False
                fill_price = None
                fill_idx = None
                # Wait up to RETRACE_FILL_MAX_BARS for a touch of the limit price
                for j, r2 in forward_all.head(RETRACE_FILL_MAX_BARS).iterrows():
                    if side == "long" and float(r2["low"]) <= entry_candidate:
                        filled = True; fill_price = entry_candidate; fill_idx = j; break
                    if side == "short" and float(r2["high"]) >= entry_candidate:
                        filled = True; fill_price = entry_candidate; fill_idx = j; break
                if not filled:
                    continue  # no trade this signal
                entry = float(fill_price)
                # recompute risk/TP from actual fill
                if side == "long":
                    risk = entry - sl
                    tp2  = entry + RR_TARGET * risk
                else:
                    risk = sl - entry
                    tp2  = entry - RR_TARGET * risk
                forward = forward_all.loc[forward_all.index > fill_idx]
            else:
                # market entry (signal close or next‑bar open)
                entry = float(entry_candidate)
                if side == "long":
                    risk = entry - sl
                    tp2  = entry + RR_TARGET * risk
                else:
                    risk = sl - entry
                    tp2  = entry - RR_TARGET * risk
                forward = forward_all

            if risk <= 0:
                continue

            # ---- Walk forward: partial at 2R, move SL to BE, run until BE or timeout ----
            hitSL = None
            hit2  = None
            be    = entry

            for j, r2 in forward.iterrows():
                lo = float(r2["low"]); hi = float(r2["high"]) 
                if side == "long":
                    # Intra‑bar resolution policy
                    if INTRABAR_PRIORITY == "tp_first":
                        if hi >= tp2: hit2 = r2[tcol]
                        if lo <= sl and hit2 is None: hitSL = r2[tcol]
                    else:  # sl_first (conservative)
                        if lo <= sl: hitSL = r2[tcol]
                        if hi >= tp2 and hitSL is None: hit2 = r2[tcol]
                else:  # short
                    if INTRABAR_PRIORITY == "tp_first":
                        if lo <= tp2: hit2 = r2[tcol]
                        if hi >= sl and hit2 is None: hitSL = r2[tcol]
                    else:
                        if hi >= sl: hitSL = r2[tcol]
                        if lo <= tp2 and hitSL is None: hit2 = r2[tcol]

                if hitSL or hit2:
                    break

            if hit2 is not None:
                # partial booked at 2R
                r_closed = PARTIAL_PCT * RR_TARGET
                # after partial, move SL to BE and let runner manage until BE or timeout
                fwd2 = forward.loc[forward.index > j]
                exit_time = None; exit_price = None
                for _, r3 in fwd2.iterrows():
                    if side == "long" and float(r3["low"]) <= be:
                        exit_time = r3[tcol]; exit_price = be; break
                    if side == "short" and float(r3["high"]) >= be:
                        exit_time = r3[tcol]; exit_price = be; break
                if exit_time is None:
                    # timeout at last available bar
                    last = fwd2.iloc[-1] if not fwd2.empty else forward.iloc[-1]
                    exit_time = last[tcol]; exit_price = float(last["close"])

                if side == "long":
                    runner_rr = max((exit_price - entry) / risk, 0.0)
                else:
                    runner_rr = max((entry - exit_price) / risk, 0.0)
                r_runner = (1 - PARTIAL_PCT) * runner_rr
                total_r = r_closed + r_runner
                trades.append({
                    "date": d, "time": ts, "side": side,
                    "entry": entry, "sl": sl, "tp2": tp2,
                    "partial_time": hit2, "runner_exit_time": exit_time,
                    "total_R": round(total_r, 3),
                    "entry_mode": mode
                })
                took_trade = True

            elif hitSL is not None:
                trades.append({
                    "date": d, "time": ts, "side": side,
                    "entry": entry, "sl": sl, "tp2": tp2,
                    "partial_time": None, "runner_exit_time": hitSL,
                    "total_R": -1.0,
                    "entry_mode": mode
                })
                took_trade = True
            # else: no resolution before timeout → no trade recorded (should be rare)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(trades)
    if res.empty:
        print("[Auralis] no trades found — check label conditions.")
        return

    res = res.sort_values("time").reset_index(drop=True)
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

if __name__ == "__main__":
    main()
