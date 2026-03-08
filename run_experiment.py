# run_experiments.py
"""
Assignment 3:
1) Train the XGBoost classifier (proposed method).
2) Run filtered backtests for several probability thresholds.
Outputs:
  - results/xgb_classifier_metrics.csv
  - results/proposed_thr_XX.csv for each threshold XX
"""

from pathlib import Path
import sys
import pandas as pd

from model_proposed import AuralisXGBFilter

ROOT = Path(__file__).resolve().parent
PIPELINES_DIR = ROOT / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.append(str(PIPELINES_DIR))

import Assignment3_Proposal_Backtesting as core
  # XGB-filtered backtest engine


def train_classifier():
    """
    Train XGBoost on the signals dataset and save metrics for the report.
    """
    print("[Auralis] Training XGB classifier (proposed method)...")
    xgb_filter = AuralisXGBFilter()
    metrics = xgb_filter.train()

    # Save metrics to CSV for Assignment 3 report
    rows = []
    for split, m in metrics.items():
        rows.append(
            {
                "split": split,
                "accuracy": m["acc"],
                "auc": m["auc"],
            }
        )

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "xgb_classifier_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[Auralis] Classifier metrics saved -> {out}")
    print(df)


def run_filtered_backtest(threshold: float):
    """
    Run the rule-based strategy with XGB filter using a specific probability threshold.
    """
    thr_tag = int(threshold * 100)  # e.g. 0.6 -> 60
    print(f"\n[Auralis] Running proposed backtest with threshold={threshold:.2f} ...")

    # Configure backtest engine for proposed method
    # (you can adjust these if you want slightly different settings)
    core.RR_TARGET = 2.0
    core.USE_RETRACE_ENTRY = False
    core.USE_NEXT_BAR_ENTRY = True
    core.USE_EMA_BIAS = False
    core.ASIA_RANGE_MIN = None
    core.ASIA_RANGE_MAX = None

    # Set probability threshold used inside the XGB filter
    core.P_THRESHOLD = threshold

    # Output path for this experiment
    out_csv = ROOT / "results" / f"proposed_thr_{thr_tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    core.OUT = out_csv

    # Run backtest
    core.main()
    print(f"[Auralis] Proposed results saved -> {out_csv}")


def main():
    # 1) Train classifier once
    train_classifier()

    # 2) Run a few candidate thresholds
    thresholds = [0.50, 0.60, 0.70]
    for thr in thresholds:
        run_filtered_backtest(thr)


if __name__ == "__main__":
    main()
