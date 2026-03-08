# evaluate_results_v2.py
"""
Assignment 3 (V2):
Compare enhanced ML-filtered strategy (proposed_v2) with
baseline models from Assignment 2.

Inputs:
  - results/baseline_summary.csv (or baseline_* CSVs)
  - results/proposed_v2_thr_XX.csv

Outputs:
  - results/proposed_v2_summary.csv
  - results/baseline_vs_proposed_v2.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def summarize_backtest(csv_path: Path, name: str) -> dict:
    df = pd.read_csv(csv_path)
    if "total_R" in df.columns:
        r_col = "total_R"
    elif "r" in df.columns:
        r_col = "r"
    else:
        raise ValueError(f"{csv_path}: no R column found")

    trades = len(df)
    wins = (df[r_col] > 0).sum()
    winrate = wins / trades if trades > 0 else 0.0
    total_R = df[r_col].sum()
    cumR = df[r_col].cumsum()
    maxDD = (cumR.cummax() - cumR).max()

    return {
        "model": name,
        "trades": trades,
        "winrate": winrate,
        "total_R": total_R,
        "max_DD_R": maxDD,
    }


def load_baseline_summary() -> pd.DataFrame:
    summary_path = RESULTS / "baseline_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    # fallback: recompute from per-baseline CSVs (same names as Assignment 2)
    rows = []
    rows.append(
        summarize_backtest(RESULTS / "baseline_A_2R.csv", "Baseline_A_2R")
    )
    rows.append(
        summarize_backtest(RESULTS / "baseline_B_3R.csv", "Baseline_B_3R")
    )
    rows.append(
        summarize_backtest(RESULTS / "baseline_C_2R_asiaFilter.csv", "Baseline_C_2R_asiaFilter")
    )
    rows.append(
        summarize_backtest(RESULTS / "baseline_D_2R_retrace.csv", "Baseline_D_2R_retrace")
    )

    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    return df


def summarize_proposed_v2() -> pd.DataFrame:
    rows = []
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    for thr in thresholds:
        tag = int(thr * 100)
        csv_path = RESULTS / f"proposed_v2_thr_{tag}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing proposed_v2 result: {csv_path}")
            continue

        name = f"ProposedV2_thr_{thr:.2f}"
        row = summarize_backtest(csv_path, name)
        row["threshold"] = thr
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        out = RESULTS / "proposed_v2_summary.csv"
        df.to_csv(out, index=False)
        print(f"[Auralis V2] Proposed summary saved -> {out}")

    return df


def main():
    baseline_df = load_baseline_summary()
    print("\n=== Baseline models ===")
    print(baseline_df)

    proposed_df = summarize_proposed_v2()
    print("\n=== Proposed V2 models ===")
    print(proposed_df)

    if not proposed_df.empty:
        combo = pd.concat([baseline_df, proposed_df], ignore_index=True)
    else:
        combo = baseline_df.copy()

    out = RESULTS / "baseline_vs_proposed_v2.csv"
    combo.to_csv(out, index=False)
    print(f"\n[Auralis V2] Baseline vs Proposed V2 saved -> {out}")
    print(combo)


if __name__ == "__main__":
    main()
