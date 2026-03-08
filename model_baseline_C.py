# model_baseline_C.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PIPELINES_DIR = ROOT / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.append(str(PIPELINES_DIR))

import backtest_baseline_optimized as core


def run_baseline_C():
    """
    Baseline C:
    - TP = 2R
    - Market entry (next-bar)
    - Asia range filter: 3 <= range <= 7 dollars
    - No EMA bias
    """
    core.RR_TARGET = 2.0
    core.USE_RETRACE_ENTRY = False
    core.USE_NEXT_BAR_ENTRY = True
    core.USE_EMA_BIAS = False

    core.LONDON_START = 7
    core.LONDON_END = 11
    core.TIMEOUT_CAP_HOUR = 20

    core.ASIA_RANGE_MIN = 3.0
    core.ASIA_RANGE_MAX = 7.0

    out = ROOT / "results" / "baseline_C_2R_asiaFilter.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    core.OUT = out

    core.main()


if __name__ == "__main__":
    run_baseline_C()
