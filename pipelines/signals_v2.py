"""
signals_v2.py
-------------
Builds the feature set for Assignment 3 (ML classifier).

We take:
 - auralis_labels.parquet  → contains signals, candle data, Asia levels
 - baseline backtest CSV   → contains entry_time + total_R (actual trade result)

We merge them to create:
   X = features
   y = win/loss (=1 if total_R>0 else 0)

Output:
   data/signals/auralis_signals_v2.parquet
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS = ROOT / "data" / "labels" / "auralis_labels.parquet"
BASELINE = ROOT / "reports" / "baseline_2018_2025_baseline.csv"

OUT = ROOT / "data" / "signals" / "auralis_signals_v2.parquet"


def build_signals():
    print("[signals_v2] Loading label file...")
    df = pd.read_parquet(LABELS)

    print("[signals_v2] Loading baseline backtest result...")
    bt = pd.read_csv(BASELINE)

    # Standardize timestamps
    tcol = None
    for c in df.columns:
        if "time" in c.lower():
            tcol = c
            break

    if tcol is None:
        raise ValueError("Could not find time column in labels file.")

    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    bt["entry_time"] = pd.to_datetime(bt["entry_time"], utc=True)

    # Merge baseline results onto label rows using timestamp
    print("[signals_v2] Merging baseline + labels...")
    merged = pd.merge(
    df,
    bt[["entry_time", "total_R"]],
    left_on=tcol,
    right_on="entry_time",
    how="left"
    )

    # Keep ONLY rows where a trade actually occurred
    merged = merged[merged["total_R"].notna()].copy()

    # Create binary label (win=1, lose=0)
    merged["win"] = (merged["total_R"] > 0).astype(int)

    # ---- Build features ----
    print("[signals_v2] Creating features...")

    merged["trend_slope"] = merged["close"].diff(12).fillna(0)
    merged["asia_range"] = merged["asia_high"] - merged["asia_low"]
    merged["hour"] = merged[tcol].dt.hour
    merged["dow"] = merged[tcol].dt.weekday

    body = (merged["close"] - merged["open"]).abs()
    merged["wick_top"] = merged["high"] - merged[["close", "open"]].max(axis=1)
    merged["wick_bottom"] = merged[["close", "open"]].min(axis=1) - merged["low"]
    merged["body_ratio"] = body / ((merged["high"] - merged["low"]) + 1e-9)

    merged["ret_3"] = merged["close"].pct_change(3).fillna(0)
    merged["ret_6"] = merged["close"].pct_change(6).fillna(0)
    merged["ret_12"] = merged["close"].pct_change(12).fillna(0)

    merged["micro_up"] = (merged["close"] > merged["open"]).astype(int)
    merged["micro_down"] = 1 - merged["micro_up"]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Final clean dataset
    merged.to_parquet(OUT, index=False)

    print(f"[signals_v2] Saved -> {OUT}")
    print(f"[signals_v2] Total rows: {len(merged)}")
    print(f"[signals_v2] Wins: {merged['win'].sum()}")


if __name__ == "__main__":
    build_signals()
