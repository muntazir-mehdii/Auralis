# evaluate_results.py
"""
Assignment 3:
Compare baselines (Assignment 2) with the proposed XGB-filtered strategy.
Outputs:
  - results/proposed_summary.csv
  - results/baseline_vs_proposed.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def summarize_backtest(csv_path: Path, name: str) -> dict:
    """
    Compute basic stats from a backtest CSV:
    trades, winrate, total_R, max drawdown in R.
    """
    df = pd.read_csv(csv_path)
    if "total_R" in df.columns:
        r_col = "total_R"
    elif "r" in df.columns:
        r_col = "r"
    else:
        raise ValueError(f"{csv_path}: no R column found")

    wins = (df[r_col] > 0).sum()
    trades = len(df)
    total_R = df[r_col].sum()
    cum_R = df[r_col].cumsum()
    dd = (cum_R.cummax() - cum_R).max()
    winrate = wins / trades if trades > 0 else 0.0

    return {
        "model": name,
        "trades": trades,
        "winrate": winrate,
        "total_R": total_R,
        "max_DD_R": dd,
    }


def load_baseline_summary() -> pd.DataFrame:
    """
    Load baseline summary from Assignment 2 if available,
    otherwise recompute from individual baseline CSVs.
    """
    summary_path = RESULTS_DIR / "baseline_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        return df

    rows = []
    rows.append(
        summarize_backtest(RESULTS_DIR / "baseline_A_2R.csv", "Baseline_A_2R")
    )
    rows.append(
        summarize_backtest(RESULTS_DIR / "baseline_B_3R.csv", "Baseline_B_3R")
    )
    rows.append(
        summarize_backtest(
            RESULTS_DIR / "baseline_C_2R_asiaFilter.csv", "Baseline_C_2R_asiaFilter"
        )
    )
    rows.append(
        summarize_backtest(
            RESULTS_DIR / "baseline_D_2R_retrace.csv", "Baseline_D_2R_retrace"
        )
    )
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    return df


def summarize_proposed() -> pd.DataFrame:
    """
    Summarize all proposed XGB-filtered backtests, one row per threshold.
    Expects files like results/proposed_thr_50.csv etc.
    """
    rows = []
    for thr_tag in [50, 60, 70]:
        csv_path = RESULTS_DIR / f"proposed_thr_{thr_tag}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing proposed result: {csv_path}")
            continue
        thr = thr_tag / 100.0
        name = f"Proposed_XGB_thr_{thr:.2f}"
        row = summarize_backtest(csv_path, name)
        row["threshold"] = thr
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        out = RESULTS_DIR / "proposed_summary.csv"
        df.to_csv(out, index=False)
        print(f"[Auralis] Proposed summary saved -> {out}")
    return df


def main():
    # Baselines
    baseline_df = load_baseline_summary()
    print("\n=== Baseline models ===")
    print(baseline_df)

    # Proposed (XGB-filtered)
    proposed_df = summarize_proposed()
    print("\n=== Proposed XGB-filtered models ===")
    print(proposed_df)

    # Combine into a single comparison table
    if not proposed_df.empty:
        combo = pd.concat([baseline_df, proposed_df], ignore_index=True)
    else:
        combo = baseline_df.copy()

    out = RESULTS_DIR / "baseline_vs_proposed.csv"
    combo.to_csv(out, index=False)
    print(f"\n[Auralis] Baseline vs Proposed comparison saved -> {out}")
    print(combo)


if __name__ == "__main__":
    main()
