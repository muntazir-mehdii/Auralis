# train.py
from model_baseline_A import run_baseline_A
from model_baseline_B import run_baseline_B
from model_baseline_C import run_baseline_C
from model_baseline_D import run_baseline_D

if __name__ == "__main__":
    print("[Auralis] Running Baseline A (2R)...")
    run_baseline_A()
    print("[Auralis] Running Baseline B (3R)...")
    run_baseline_B()
    print("[Auralis] Running Baseline C (2R + Asia filter)...")
    run_baseline_C()
    print("[Auralis] Running Baseline D (2R + retrace)...")
    run_baseline_D()
