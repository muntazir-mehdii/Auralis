"""
Assignment3_Proposal_Backtesting_v2.py
--------------------------------------

Upgraded ML-powered backtester for Auralis Assignment 3.
Uses:
 - signals_v2 features
 - model_proposed_v2 XGB classifier
 - Fibonacci 61.8% retrace-entry logic (Option B2)
 - Next-bar or retrace entry controlled by config
 - Proper ML filtering with probability threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import upgraded model
from models.model_proposed_v2 import AuralisXGBFilterV2

# ----------------------------------------
# GLOBAL CONFIG
# ----------------------------------------

LBL = ROOT / "data" / "labels" / "auralis_labels.parquet"

OUT = ROOT / "results" / "proposed_v2.csv"

# Trading session
LONDON_START = 8       # 08:00 UTC
LONDON_END   = 11      # entry before 11 UTC
TIMEOUT_CAP  = 15      # force exit after 15:00 UTC

# Risk config
RR_TARGET = 2.0
SL_BUFFER = 1.0

# ML config
P_THRESHOLD = 0.60     # dynamically overwritten by run_experiments

# Entry type
USE_RETRACE = True     # retrace entry only
RETRACE_LEVEL = 0.618  # FIBO 61.8%

# One trade per day rule
ONE_TRADE_PER_DAY = True

# ----------------------------------------
# Helper functions
# ----------------------------------------

def get_time_col(df):
    for c in df.columns:
        if "time" in c.lower():
            return c
    raise ValueError("No time/timestamp column found.")


def in_london_window(ts):
    h = ts.dt.hour
    return (h >= LONDON_START) & (h < LONDON_END)


def fibo_retrace_entry(row, side):
    high = float(row["high"])
    low  = float(row["low"])
    if side == "long":
        return low + (high - low) * RETRACE_LEVEL
    else:
        return high - (high - low) * RETRACE_LEVEL


def intrabar_hit(l, h, sl, tp):
    sl_hit = l <= sl <= h
    tp_hit = l <= tp <= h
    if sl_hit and tp_hit:
        return "sl"
    if sl_hit:
        return "sl"
    if tp_hit:
        return "tp"
    return None


# ----------------------------------------
# MAIN BACKTEST FUNCTION
# ----------------------------------------

def main():

    print("[Auralis] Loading labels...")
    df = pd.read_parquet(LBL).copy()
    tcol = get_time_col(df)

    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.sort_values(tcol).reset_index(drop=True)
    df["date"] = df[tcol].dt.date

    # Load ML classifier
    print("[Auralis] Loading ML model V2...")
    xgb = AuralisXGBFilterV2().load_trained()

    trades = []

    for date, day in df.groupby("date"):

        day = day.sort_values(tcol).reset_index(drop=True)

        took_trade_today = False

        london = day[in_london_window(day[tcol])]
        if london.empty:
            continue

        for i, row in london.iterrows():

            if ONE_TRADE_PER_DAY and took_trade_today:
                break

            # --- SIGNAL LOGIC ---
            long_sig  = bool(row.get("entry_long", False) and row.get("confirm_long", False))
            short_sig = bool(row.get("entry_short", False) and row.get("confirm_short", False))

            if not long_sig and not short_sig:
                continue
            if long_sig and short_sig:
                continue

            side = "long" if long_sig else "short"

            # --- BUILD FEATURES FOR ML FILTER ---
            feat = {
                "trend_1h"      : row.get("trend_1h", 0),
                "trend_4h"      : row.get("trend_4h", 0),
                "trend_slope"   : row.get("trend_slope", 0),
                "vol_regime"    : row.get("vol_regime", 0),
                "compression"   : row.get("compression", 0),
                "asia_range"    : row.get("asia_range", 0),
                "dist_asia_high": row.get("dist_asia_high", 0),
                "dist_asia_low" : row.get("dist_asia_low", 0),
                "wick_top"      : row.get("wick_top", 0),
                "wick_bottom"   : row.get("wick_bottom", 0),
                "body_ratio"    : row.get("body_ratio", 0),
                "ret_3"         : row.get("ret_3", 0),
                "ret_6"         : row.get("ret_6", 0),
                "ret_12"        : row.get("ret_12", 0),
                "micro_up"      : row.get("micro_up", 0),
                "micro_down"    : row.get("micro_down", 0),
                "hour"          : row[tcol].hour,
                "dow"           : row[tcol].weekday(),
            }

            p_score = xgb.score_trade(feat)

            if p_score < P_THRESHOLD:
                continue

            # --- SL LOGIC ---
            asia_high = float(row.get("asia_high", row["high"]))
            asia_low  = float(row.get("asia_low",  row["low"]))

            if side == "long":
                sl = min(float(row["low"]), asia_low) - SL_BUFFER
            else:
                sl = max(float(row["high"]), asia_high) + SL_BUFFER

            # --- ENTRY PRICE (61.8% RETRACE) ---
            entry_price = fibo_retrace_entry(row, side)

            risk = (entry_price - sl) if side == "long" else (sl - entry_price)
            if risk <= 0:
                continue

            tp = entry_price + RR_TARGET * risk if side == "long" else entry_price - RR_TARGET * risk

            # --- FORWARD SCAN ---
            forward = day.loc[day.index > i]
            final_exit = None
            total_R = 0.0

            for _, r2 in forward.iterrows():

                ts2 = r2[tcol]
                if ts2.hour >= TIMEOUT_CAP:
                    close_px = float(r2["close"])
                    total_R = ((close_px - entry_price) / risk) if side == "long" else ((entry_price - close_px) / risk)
                    final_exit = ts2
                    break

                L, H = float(r2["low"]), float(r2["high"])
                hit = intrabar_hit(L, H, sl, tp)

                if hit == "sl":
                    total_R = -1.0
                    final_exit = ts2
                    break

                if hit == "tp":
                    total_R = RR_TARGET
                    final_exit = ts2
                    break

            # fallback exit
            if final_exit is None:
                last = forward.iloc[-1]
                close_px = float(last["close"])
                total_R = ((close_px - entry_price) / risk) if side == "long" else ((entry_price - close_px) / risk)
                final_exit = last[tcol]

            trades.append({
                "date": date,
                "entry_time": row[tcol],
                "exit_time": final_exit,
                "side": side,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "ml_score": p_score,
                "total_R": total_R,
            })

            took_trade_today = True

    # ----------------------------------------
    # SAVE RESULTS
    # ----------------------------------------
    if not trades:
        print("[Auralis] No trades generated.")
        return

    res = pd.DataFrame(trades)
    res = res.sort_values("entry_time").reset_index(drop=True)

    res.to_csv(OUT, index=False)

    cumR = res["total_R"].cumsum()
    maxDD = (cumR.cummax() - cumR).max()

    print(f"[Auralis V2] trades: {len(res)}")
    print(f"[Auralis V2] winrate: {(res['total_R'] > 0).mean():.2%}")
    print(f"[Auralis V2] total R: {res['total_R'].sum():.2f}")
    print(f"[Auralis V2] max DD (R): {maxDD:.2f}")
    print(f"[Auralis V2] saved -> {OUT}")


if __name__ == "__main__":
    main()
