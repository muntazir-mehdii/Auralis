import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 4)

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT.parent / "reports" / "baseline_2018_2025_baseline.csv"

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def max_drawdown(series):
    roll_max = series.cummax()
    dd = roll_max - series
    return float(dd.max())

def longest_streak(xs, condition=True):
    best = cur = 0
    for v in xs:
        if v == condition:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def main():
    # Read and preprocess the data
    df = pd.read_csv(IN, parse_dates=["entry_time", "exit_time"])
    df.rename(columns={"entry_time": "time", "exit_time": "partial_time"}, inplace=True)
    
    if df.empty:
        print("[Auralis] No trades in baseline_2025.csv")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "time"]).reset_index(drop=True)

    # Core stats
    df["cum_R"] = df["total_R"].cumsum()
    wins = (df["total_R"] > 0)
    losses = (df["total_R"] <= 0)

    summary = {
        "trades": int(len(df)),
        "win_rate": float(wins.mean()),
        "total_R": float(df["total_R"].sum()),
        "max_dd_R": max_drawdown(df["cum_R"]),
        "avg_R_per_trade": float(df["total_R"].mean()),
        "median_R_per_trade": float(df["total_R"].median()),
        "max_win_streak": int(longest_streak(wins.tolist(), True)),
        "max_loss_streak": int(longest_streak(losses.tolist(), True)),
        "first_trade": str(df["time"].iloc[0]),
        "last_trade": str(df["time"].iloc[-1]),
    }

    # Monthly breakdown
    df["month"] = df["time"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["total_R"].agg(["count", "sum", "mean"]).reset_index()
    monthly.rename(columns={"count": "trades", "sum": "total_R", "mean": "avg_R"}, inplace=True)
    monthly_out = OUT_DIR / "baseline_2025_monthly.csv"
    monthly.to_csv(monthly_out, index=False)

    # Equity curve plot
    eq_png = OUT_DIR / "baseline_2025_equity.png"
    plt.figure()
    plt.plot(df["time"], df["cum_R"])
    plt.title("Auralis — Baseline Equity (R)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative R")
    plt.tight_layout()
    plt.savefig(eq_png, dpi=140)
    plt.close()

    # Dump summary
    import json
    with open(OUT_DIR / "baseline_2025_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[Auralis] summary:", summary)
    print(f"[Auralis] monthly -> {monthly_out}")
    print(f"[Auralis] equity  -> {eq_png}")
    print(f"[Auralis] report  -> {OUT_DIR / 'baseline_2025_summary.json'}")

if __name__ == "__main__":
    main()
