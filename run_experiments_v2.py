# run_experiments_v2.py
"""
Assignment 3 (V2):
1) Build signals_v2 dataset.
2) Train the upgraded XGBoost classifier (model_proposed_v2).
3) Run ML-filtered backtests with Assignment3_Proposal_Backtesting_v2.py
   for a custom list of probability thresholds.

Outputs:
  - data/signals/auralis_signals_v2.parquet
  - models/xgb_filter_v2.json (+ feature meta)
  - results/xgb_classifier_v2_metrics.csv
  - results/proposed_v2_thr_XX.csv for each threshold
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
PIPELINES = ROOT / "pipelines"
MODELS    = ROOT / "models"

if str(PIPELINES) not in sys.path:
    sys.path.append(str(PIPELINES))
if str(MODELS) not in sys.path:
    sys.path.append(str(MODELS))

# 1) Build signals_v2
from signals_v2 import build_signals
# 2) Train upgraded classifier
from model_proposed_v2 import AuralisXGBFilterV2
# 3) ML-powered backtester v2
import Assignment3_Proposal_Backtesting_v2 as core


def train_classifier_v2():
    print("[Auralis V2] Building signals_v2 dataset...")
    build_signals()

    print("[Auralis V2] Training classifier (model_proposed_v2)...")
    clf = AuralisXGBFilterV2()
    metrics = clf.train()

    rows = []
    for split, m in metrics.items():
        rows.append({
            "split": split,
            "accuracy": m["acc"],
            "auc": m["auc"],
        })

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "xgb_classifier_v2_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"[Auralis V2] Classifier metrics saved -> {out}")
    print(df)


def run_backtest_for_threshold(thr: float):
    tag = int(thr * 100)   # 0.30 -> 30, etc.
    print(f"\n[Auralis V2] Running backtest with P_THRESHOLD={thr:.2f} ...")

    # configure backtester
    core.P_THRESHOLD = thr
    out_csv = ROOT / "results" / f"proposed_v2_thr_{tag}.csv"
    core.OUT = out_csv

    core.main()
    print(f"[Auralis V2] Saved -> {out_csv}")


def main():
    # Step 1: build signals + train classifier
    train_classifier_v2()

    # Step 2: run backtests for custom thresholds (Group 3)
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    # thresholds = [0.05, 0.50, 0.95]

    for thr in thresholds:
        run_backtest_for_threshold(thr)


if __name__ == "__main__":
    main()
