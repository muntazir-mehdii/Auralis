# Auralis — AI-Powered Prediction & Automation Pipeline

> Quantitative research project analyzing the **Asian Session Liquidity Sweep (ASLS)** in the London session for **XAUUSD**. Includes data labeling, rule-based backtesting, strategy optimization, and an XGBoost ML filter — tested across 2018–2025 market data.

---

## Final Results

| Stage | Model | Return | Win Rate |
|-------|-------|--------|----------|
| A1 | Mechanical Baseline | ~8 R | ~33% |
| A2 | Optimized Baseline (Retracement) | 15.58 R | ~34% |
| A3 | XGBoost Model V1 | -8.25 R | ~32% |
| **A3** | **XGBoost Model V2 (Final)** | **+94.92 R** | **36.23%** |

**Model V2 outperformed the rule-based baseline by 6×.**

---

## Repository Structure

```
Auralis/
├── data/
│   ├── raw/                          # Raw OHLCV data (not included; large)
│   └── labels/
│       └── auralis_labels.parquet    # Produced by labeler.py
├── pipelines/
│   ├── labeler.py                    # Asian session range + signal builder
│   ├── backtest_baseline.py          # Baseline executor (RR=2/3 variants)
│   ├── backtest_baseline_optimized.py
│   ├── signals_v1.py                 # ML feature dataset (V1)
│   ├── signals_v2.py                 # ML feature dataset (V2, final)
│   └── retrace_entry.py
├── models/
│   ├── model_proposed.py             # XGBoost V1
│   ├── model_proposed_v2.py          # XGBoost V2 (final)
│   └── xgb_filter_v2.json            # Saved trained classifier
├── results/
│   ├── proposed_v2_thr_50.csv        # Best backtest (θ=0.50)
│   ├── baseline_vs_proposed.csv      # Threshold sweep summary
│   └── confusion_matrix_v2.csv
├── tools/
│   ├── make_backtest_report.py       # Auto-generates PDF + PNG figures
│   └── make_assignment3_figures.py
├── plots/                            # LaTeX-ready output figures
├── reports/                          # CSV trade logs + figure exports
├── main.tex                          # LaTeX report
└── requirements.txt
```

---

## Problem Summary

During **00:00–06:00 UTC** (Asian session), XAUUSD often forms a tight liquidity range. After the London open (~07:00–10:00 UTC), price frequently sweeps one side of that range and reverses.

Auralis formalizes a fully mechanical approach to:
- Detect the Asian session range
- Label sweep + confirmation signals
- Execute a strict risk model (hard SL + fixed RR targets)
- Evaluate behavior with consistent metrics and plots

---

## Pipeline — 3 Stages

### Stage 1 — Data & Labeling

**Instrument:** XAUUSD (M5), 2018–2025, UTC timestamps

**Labeling logic (`pipelines/labeler.py`):**
- Compute per-day Asia High/Low between 00:00–06:00 UTC
- Sweep: candle breaks outside the Asian box
- Confirmation: price re-closes inside within 1–2 bars
- Outputs `entry_long`, `entry_short` booleans → saved as `.parquet`

> ⚠️ All timestamps are UTC. Misaligned timezones will degrade signals.

---

### Stage 2 — Strategy Optimization

**Key additions over baseline:**

- **Asia Range Filters** — skip low-quality days (range too tight/wide, holiday chop, unfavorable volatility)
- **Retracement Entry Model** — wait for BOS/MSS candle after sweep, enter on retracement, SL outside wick, TP 80% at 2R + runner to BE
- **EMA Bias Filter** — directional confirmation before entry
- **5-year backtest** — full 2018–2025 run with next-bar execution

Best variant: `reports/baseline_D_2R_retrace.csv` → **15.58 R**

---

### Stage 3 — XGBoost ML Filter

Two models trained to classify trade quality (high vs low probability):

**Model V1** — basic ASLS feature set → **-8.25 R** (underfitting, weak features)

**Model V2 (Final)** — advanced feature engineering:
- Regime features (volatility, trending vs ranging)
- Momentum features (price velocity, session bias)
- Microstructure features (spread, candle body ratios)

Result: **+94.92 R** at θ=0.50 threshold — best performance across all Auralis tests.

---

## How to Run

### Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
```

### Stage 1 — Generate Labels

```bash
python pipelines\labeler.py
# Output: data/labels/auralis_labels.parquet
```

### Stage 2 — Run Optimized Backtest

```bash
python pipelines\backtest_baseline_optimized.py
# Output: reports/baseline_2018_2025_baseline.csv

python tools\make_backtest_report.py `
  --inputs "reports\baseline_D_2R_retrace.csv" `
  --outdir "reports\figures_optimized" `
  --title "Auralis — Optimized Baseline"
```

### Stage 3 — Train ML Model & Backtest

```bash
# Build features
python pipelines\signals_v2.py

# Train XGBoost V2
python run_experiments_v2.py
# Outputs: models/xgb_filter_v2.json, results/proposed_v2_thr_50.csv

# Generate figures
python tools\make_assignment3_figures.py
```

---

## Output Figures

| Figure | Description |
|--------|-------------|
| `equity_curve.png` | Cumulative R over time |
| `threshold_sweep.png` | Performance vs classification threshold |
| `feature_importance.png` | Top XGBoost features |
| `confusion_matrix_v2.png` | Model classification accuracy |
| `DrawDownCurve_Date.pdf` | Drawdown over time |
| `WinRate_time.pdf` | Win rate by hour of day |
| `MonthlyPerformace.pdf` | Monthly P&L breakdown |

---

## Tech Stack

`Python` `XGBoost` `pandas` `numpy` `scikit-learn` `pyarrow` `matplotlib` `LaTeX`

---

## Author

**Muntazir Mehdi** — ML Engineer & Full-Stack Developer  
[GitHub](https://github.com/muntazir-mehdii) · [Upwork](https://www.upwork.com/freelancers/~015ab18bf2700e35b7)
