# model_baseline_D.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PIPELINES_DIR = ROOT / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.append(str(PIPELINES_DIR))

import backtest_baseline_optimized as core


def run_baseline_D():
    """
    Baseline D:
    - TP = 2R
    - Retrace entry at 50% of signal candle
    - No Asia range filter
    - No EMA bias
    """
    core.RR_TARGET = 2.0
    core.USE_RETRACE_ENTRY = True
    core.RETRACE_PCT = 0.5
    core.USE_NEXT_BAR_ENTRY = True  # still use next-bar context
    core.USE_EMA_BIAS = False

    core.LONDON_START = 7
    core.LONDON_END = 11
    core.TIMEOUT_CAP_HOUR = 20

    core.ASIA_RANGE_MIN = None
    core.ASIA_RANGE_MAX = None

    out = ROOT / "results" / "baseline_D_2R_retrace.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    core.OUT = out

    core.main()


if __name__ == "__main__":
    run_baseline_D()
