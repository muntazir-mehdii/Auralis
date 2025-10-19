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