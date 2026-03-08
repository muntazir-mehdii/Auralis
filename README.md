Auralis — Assignment 1: Problem & Data Understanding

Project focus: Asian Session Liquidity Sweep (ASLS) during London session on XAUUSD (M5).
Scope of A1: data sourcing, preprocessing/labeling, a strict mechanical baseline backtest, and EDA figures to understand behavior and metrics before optimization/ML.

1) What this repository contains
Auralis/
├─ data/
│  ├─ raw/                         # (not included in submission; large)
│  └─ labels/
│     └─ auralis_labels.parquet    # produced by labeler.py (or small sample)
├─ pipelines/
│  ├─ labeler.py                   # builds Asian session range + signals
│  └─ backtest_baseline.py         # strict baseline executor (RR=2/3 variants)
├─ tools/
│  └─ make_backtest_report.py      # exports multi-page PDF + PNG charts
├─ reports/
│  ├─ baseline_2025.csv            # example output (trade log)
│  └─ figures/                     # auto-generated plots (PNG + PDF)
├─ plots/                          # your LaTeX-ready PDFs (full-width figures)
│  ├─ baseline_2025_equity.pdf
│  ├─ DrawDownCurve_Date.pdf
│  ├─ DistributionPerTrade.pdf
│  ├─ PerTrade_RScatter.pdf
│  ├─ WinRate_time.pdf
│  └─ MonthlyPerformace.pdf
├─ main.tex                        # LaTeX Assignment 1 report
├─ README.md
└─ requirements.txt


For grading, raw historical data isn’t required; we provide scripts and instructions to reproduce labels and plots.

2) Problem summary

During 00:00–06:00 UTC (Asian session) XAUUSD often forms a tight range (liquidity pool). After London open (≈07:00–10:00 UTC), price often sweeps one side of that range and reverses. We formalize a fully mechanical approach to:

detect the Asian range,

label sweep + confirmation,

execute a strict risk model (hard SL with buffer; fixed RR targets),

evaluate the behavior with consistent metrics/plots.

3) Data and labeling (A1)

Instrument: XAUUSD (Spot Gold), M5 (some M1 support for verification).

Span: 2018–2025, UTC timestamps.

Labeling (pipelines/labeler.py):

Compute per-day Asia High/Low between 00:00–06:00 UTC.

Sweep: candle breaks outside the Asian box.

Confirmation: re-close back inside within 1–2 bars.

Emit booleans entry_long, entry_short, and confirmations.

Save to data/labels/auralis_labels.parquet (Parquet).

Important: All times are in UTC (not PKT). Misaligned timezones will degrade signals and session filters.

4) Baseline executor (A1)

File: pipelines/backtest_baseline.py

Session: Allow entries only during London (default 07:00–11:00 UTC).

Stops/Targets:

Hard SL at wick/Asia boundary with configurable buffer (e.g., $0.5).

Options tested: full exit @ 2R or full exit @ 3R; hybrid partials are supported in other variants.

Management window: manage trade up to min(LONDON_END+2h, 20:00 UTC).

Outputs: CSV trade log + cumulative results (R-multiples).

5) EDA and figures (A1)

File: tools/make_backtest_report.py
Reads one or more backtest CSVs and produces:

A multi-page PDF (reports/figures/backtest_figures.pdf)

Individual PNG images for LaTeX (reports/figures/*.png)

Figures we use in the LaTeX report (you also exported PDFs to plots/):

Equity: baseline_2025_equity.pdf

Drawdown: DrawDownCurve_Date.pdf

Per-trade distribution: DistributionPerTrade.pdf

Per-trade scatter: PerTrade_RScatter.pdf

Win rate vs hour: WinRate_time.pdf

Monthly performance: MonthlyPerformace.pdf

These confirm: positive expectancy with asymmetric payoffs (few 2R–3R winners drive gains), drawdown ≈ 11R in the 2025 snapshot, and strongest behavior near 08:00–09:00 UTC.

6) How to run (Windows PowerShell examples)
6.1 Install environment
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

6.2 Generate labels (creates data/labels/auralis_labels.parquet)
python pipelines\labeler.py

6.3 Run the baseline backtest (creates reports\baseline_2025.csv)
python pipelines\backtest_baseline.py

6.4 Export report figures (multi-page PDF + PNGs)
python tools\make_backtest_report.py `
  --inputs "reports\baseline_2025.csv" `
  --outdir "reports\figures" `
  --title "Auralis — Baseline 2025"


(Use backticks for line breaks in PowerShell; use ^ in CMD or run on one line.)

2) Summary of Improvements in A2

Assignment 2 focuses on building a robust baseline across 2018–2025.
Main additions include:

• Asia Range Filters

Skip low-quality days where:

Asia range is too tight or too wide

Chop-zones around major holidays

Unfavorable volatility regimes

• Retracement Entry Model (Best Variant in A2)

After the sweep:

Wait for a strong BOS/MSS candle

Enter only on retracement back into a fair price area

SL outside the sweep wick

TP: 80% at 2R, runner to BE (mirrors real SMC logic)

This produced:

Higher expectancy

Cleaner entries

Reduced drawdowns

Best overall performance in A2
Saved as:
reports/baseline_D_2R_retrace.csv

• 5-Year Backtest

pipelines/backtest_baseline_optimized.py
Runs a full 2018–2025 historical backtest using:

Next-bar execution

EMA bias

Asia filter

Retracement logic

3) How to Run Assignment 2
(1) Activate environment
. .\.venv\Scripts\Activate.ps1

(2) Run the optimized 5-year baseline
python pipelines\backtest_baseline_optimized.py


Outputs:

reports\baseline_2018_2025_baseline.csv

(3) Run individual variants
python pipelines\rr2_fulltp.py
python pipelines\rr3_fulltp.py
python pipelines\retrace_entry.py

(4) Generate A2 figures
python tools\make_backtest_report.py `
  --inputs "reports\baseline_D_2R_retrace.csv" `
  --outdir "reports\figures_optimized" `
  --title "Auralis — Optimized Baseline Variants"

Assignment 3 — Machine Learning Filter (XGBoost)

Assignment 3 introduces Machine Learning classification to filter out low-probability trades and improve overall profitability.

This stage contains two ML models:

Model 1 → model_proposed
(Basic ASLS feature set, Version 1)

Model 2 → model_proposed_v2
(Improved feature engineering, regime/momentum/microstructure — final model)

Model 2 is the best performer and forms the final Assignment-3 submission.

1) What Assignment 3 Adds
Auralis/
├─ pipelines/
│  ├─ signals_v1.py                      # build feature dataset (V1)
│  ├─ signals_v2.py                      # improved feature set (V2)
│  └─ build_signals_dataset.py           # helper/diagnostic builder
├─ models/
│  ├─ model_proposed.py                  # ML Model V1
│  ├─ model_proposed_v2.py               # ML Model V2 (final)
│  └─ xgb_filter_v2.json                 # saved trained classifier
├─ results/
│  ├─ proposed_v2_thr_50.csv             # backtest (θ=0.50 optimal)
│  ├─ baseline_vs_proposed.csv           # threshold sweep summary
│  ├─ confusion_matrix_v2.csv            # 2x2 matrix output
│  └─ proposed_v1_*.csv                  # Model 1 results (negative)
└─ tools/
   └─ make_assignment3_figures.py        # auto-generates all plots

2) Assignment 3 Workflow
Step 1 — Build ML Signals Dataset (Features + Labels)

For Model 1:

python pipelines\signals_v1.py


For Model 2:

python pipelines\signals_v2.py


This produces:

data/signals/auralis_signals_v2.parquet

Step 2 — Train ML Model
Model 1 (Version 1)
python run_experiment.py

Model 2 (Version 2 — Final)
python run_experiments_v2.py


This saves:

models/xgb_filter_v2.json
results/confusion_matrix_v2.csv
results/proposed_v2_thr_50.csv

Step 3 — Backtest with Probability Thresholds

Run threshold sweeps (0.05 → 0.95):

python run_experiments_v2.py


This produces the comparison table:

results/baseline_vs_proposed.csv

Step 4 — Generate Assignment 3 Figures (Automatic)

Just press F5 on:

tools/make_assignment3_figures.py


Or run:

python tools\make_assignment3_figures.py


Outputs include:

plots/proposed_v2/equity_curve.png
plots/proposed_v2/threshold_sweep.png
plots/proposed_v2/feature_importance.png
results/confusion_matrix_v2.png


These are referenced in the LaTeX Assignment 3 report.

3) Key Outcomes in Assignment 3
Model 1 (V1)

Return: –8.25 R (negative)

Win rate: ~32–35%

Didn’t outperform baseline

Reason: Weak feature set + low expressiveness

Model 2 (V2 — Final Model)

Return: +94.92 R (best across entire project)

Win rate: 36.23%

Drawdown: Significantly reduced vs baseline

Trades stable across thresholds (0.3–0.8)

Strong feature engineering + tuning

Baseline Best (A2)

Baseline_D_2R_retrace.csv

Return: 15.58 R

Win rate: ~34%

Model V2 outperformed baseline by 6× in total R.

4) How to Run Assignment 3 from Zero
(1) Activate environment
. .\.venv\Scripts\Activate.ps1

(2) Build signals (features)
python pipelines\signals_v2.py

(3) Train ML model
python run_experiments_v2.py

(4) Generate all A3 figures for report
python tools\make_assignment3_figures.py

(5) Compile LaTeX

Your report will reference the PNG/PDF outputs.

5) Summary

Assignment 1 → Data, labeling, pure baseline

Assignment 2 → Optimization, retracement model, 5-year backtest

Assignment 3 → Full ML pipeline (XGBoost), model comparison, huge performance gain

Final Result → Model V2 achieves +94.92 R, highest across all Auralis tests.