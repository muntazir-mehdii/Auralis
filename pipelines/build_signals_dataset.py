# pipelines/build_signals_dataset.py

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "reports" / "baseline_2018_2025_baseline.csv"
LABELS_PATH = ROOT / "data" / "labels" / "auralis_labels.parquet"
OUT_PATH = ROOT / "data" / "auralis_signals_xgb.parquet"

def compute_features(trades_df: pd.DataFrame, full_df: pd.DataFrame, tcol: str = "time") -> pd.DataFrame:
    trades_df = trades_df.copy()
    full_df = full_df.copy()

    full_df[tcol] = pd.to_datetime(full_df[tcol], utc=True)
    full_df = full_df.sort_values(tcol).set_index(tcol)

    feature_rows = []

    for _, tr in trades_df.iterrows():
        t_entry = pd.to_datetime(tr["entry_time"], utc=True)

        if t_entry not in full_df.index:
            # If exact timestamp missing due to M5 gaps, snap to nearest earlier bar
            idx = full_df.index.get_indexer([t_entry], method="pad")
            if idx[0] == -1:
                continue
            row = full_df.iloc[idx[0]]
        else:
            row = full_df.loc[t_entry]

        asia_high = float(tr.get("asia_high", row.get("asia_high", row["high"])))
        asia_low  = float(tr.get("asia_low",  row.get("asia_low",  row["low"])))
        asia_range = asia_high - asia_low

        hour = t_entry.hour
        dow = t_entry.weekday()

        f = {
            "entry_time": t_entry,
            "year": t_entry.year,
            "side": 1 if tr["side"] == "long" else -1,
            "result_r": float(tr["total_R"]),
            "asia_range": asia_range,
            "hour": hour,
            "dow": dow,
            "price_close": float(row["close"]),
        }

        # Optional indicators if present
        for col in ["ema_200", "atr_14", "rsi_14"]:
            if col in row.index:
                val = row[col]
                f[col] = float(val) if not pd.isna(val) else 0.0

        feature_rows.append(f)

    return pd.DataFrame(feature_rows)

def main():
    trades = pd.read_csv(TRADES_CSV)
    if trades.empty:
        print("[Auralis] No trades in CSV, run backtest first.")
        return

    df_full = pd.read_parquet(LABELS_PATH)

    # Guess time column
    tcols = [c for c in df_full.columns if "time" in c.lower() or "timestamp" in c.lower()]
    if not tcols:
        raise ValueError("No time column found in labels parquet.")
    tcol = tcols[0]

    signals = compute_features(trades, df_full, tcol=tcol)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(OUT_PATH, index=False)
    print(f"[Auralis] Signals dataset saved -> {OUT_PATH}")
    print(signals.head())

if __name__ == "__main__":
    main()
