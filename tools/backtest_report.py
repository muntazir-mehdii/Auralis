# tools/make_backtest_report.py
# Auralis: Generate equity, drawdown, distribution and time-structure plots
# from one or more backtest CSV files.
#
# Usage (PowerShell example):
#   python tools\make_backtest_report.py `
#       --inputs reports\baseline_2025.csv `
#       --outdir reports\figures `
#       --title "Auralis — Baseline 2025"
#
# This will create:
#   reports/figures/backtest_figures.pdf      (multi-page)
#   reports/figures/<stem>_equity.pdf/.png
#   reports/figures/DrawDownCurve_Date.pdf/.png
#   reports/figures/DistributionPerTrade.pdf/.png
#   reports/figures/PerTrade_RScatter.pdf/.png
#   reports/figures/WinRate_time.pdf/.png
#   reports/figures/MonthlyPerformace.pdf/.png

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_backtest(path: Path) -> pd.DataFrame:
    """Load a backtest CSV and normalize columns."""
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # time column
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        raise ValueError(f"{path}: no 'time' or 'date' column found")

    # result column (R per trade)
    if "total_r" in df.columns:
        df["r"] = df["total_r"]
    elif "result_r" in df.columns:
        df["r"] = df["result_r"]
    else:
        raise ValueError(f"{path}: no 'total_R' or 'result_R' column found")

    df = df.sort_values("time").reset_index(drop=True)
    df["cum_r"] = df["r"].cumsum()

    # max drawdown in R (peak-to-trough of cum_r)
    rolling_max = df["cum_r"].cummax()
    df["dd_r"] = df["cum_r"] - rolling_max  # negative numbers
    return df


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, outdir: Path, basename: str, pdf_pages: PdfPages) -> None:
    """Save a figure as PDF + PNG and append to multipage PDF."""
    pdf_path = outdir / f"{basename}.pdf"
    png_path = outdir / f"{basename}.png"

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    pdf_pages.savefig(fig)
    plt.close(fig)


def plot_equity(
    dfs: List[pd.DataFrame],
    labels: List[str],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for df, lab in zip(dfs, labels):
        ax.plot(df["time"], df["cum_r"], label=lab)
    ax.set_title(f"Cumulative R vs Time\n{title}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative R")
    if len(dfs) > 1:
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def plot_drawdown(dfs: List[pd.DataFrame], labels: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for df, lab in zip(dfs, labels):
        ax.plot(df["time"], df["dd_r"], label=lab)
    ax.set_title("Drawdown Curve (R)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (R)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if len(dfs) > 1:
        ax.legend()
    return fig


def plot_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["r"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of R per Trade")
    ax.set_xlabel("R per trade")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def plot_scatter(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # x-axis: trade index; y-axis: R
    ax.scatter(range(len(df)), df["r"], s=15)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Per-Trade R Scatter (Chronological)")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("R per trade")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def plot_winrate_by_hour(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tmp = df.copy()
    tmp["hour"] = tmp["time"].dt.hour
    grouped = tmp.groupby("hour")["r"]
    winrate = (grouped.apply(lambda x: (x > 0).mean()) * 100.0).fillna(0)
    count = grouped.size()

    ax.bar(winrate.index, winrate.values)
    ax.set_title("Win Rate by Hour (UTC)")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Win rate (%)")
    ax.set_xticks(sorted(winrate.index))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # annotate counts above bars (optional, small text)
    for h, wr in winrate.items():
        ax.text(h, wr + 1, str(int(count.loc[h])), ha="center", va="bottom", fontsize=7)
    return fig


def plot_monthly_performance(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tmp = df.copy()
    tmp["month"] = tmp["time"].dt.to_period("M").dt.to_timestamp()
    agg = tmp.groupby("month")["r"].sum()

    ax.bar(agg.index, agg.values, width=20)  # ~month width
    ax.set_title("Monthly Performance (Sum of R)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total R")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.autofmt_xdate()
    return fig


def print_summary(df: pd.DataFrame, label: str) -> None:
    wins = (df["r"] > 0).sum()
    losses = (df["r"] < 0).sum()
    total_trades = len(df)
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_r = df["r"].sum()
    avg_r = df["r"].mean() if total_trades > 0 else 0.0
    max_dd = df["dd_r"].min()  # negative

    print(f"\n[Summary] {label}")
    print(f"  Trades      : {total_trades}")
    print(f"  Wins/Losses : {wins} / {losses}")
    print(f"  Win rate    : {win_rate:.2%}")
    print(f"  Total R     : {total_r:.2f}")
    print(f"  Avg R/trade : {avg_r:.3f}")
    print(f"  Max DD (R)  : {max_dd:.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate backtest report plots (equity, DD, distribution, etc.)"
    )
    p.add_argument(
        "--inputs",
        "-i",
        nargs="+",
        required=True,
        help="One or more backtest CSV files.",
    )
    p.add_argument(
        "--outdir",
        "-o",
        default="reports/figures",
        help="Output directory for plots (default: reports/figures).",
    )
    p.add_argument(
        "--title",
        "-t",
        default="Auralis Backtest",
        help="Title for equity plot.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    input_paths = [Path(p) for p in args.inputs]
    dfs: List[pd.DataFrame] = []
    labels: List[str] = []

    for p in input_paths:
        df = load_backtest(p)
        dfs.append(df)
        labels.append(p.stem)

    # multi-page PDF container
    pdf_path = outdir / "backtest_figures.pdf"
    with PdfPages(pdf_path) as pdf_pages:

        # Equity (R over time)
        fig_eq = plot_equity(dfs, labels, args.title)
        # For the single-baseline case, use <stem>_equity, else generic.
        if len(labels) == 1:
            eq_name = f"{labels[0]}_equity"
        else:
            eq_name = "CummulativeR_Date"
        save_fig(fig_eq, outdir, eq_name, pdf_pages)

        # Drawdown curve
        fig_dd = plot_drawdown(dfs, labels)
        save_fig(fig_dd, outdir, "DrawDownCurve_Date", pdf_pages)

        # The following plots use the first dataset as the "primary" one
        df0 = dfs[0]

        # Distribution of R
        fig_hist = plot_distribution(df0)
        save_fig(fig_hist, outdir, "DistributionPerTrade", pdf_pages)

        # Scatter of R per trade
        fig_scatter = plot_scatter(df0)
        save_fig(fig_scatter, outdir, "PerTrade_RScatter", pdf_pages)

        # Win rate by hour
        fig_hour = plot_winrate_by_hour(df0)
        save_fig(fig_hour, outdir, "WinRate_time", pdf_pages)

        # Monthly performance
        fig_month = plot_monthly_performance(df0)
        save_fig(fig_month, outdir, "MonthlyPerformace", pdf_pages)

    print(f"\n[Auralis] Report figures written -> {pdf_path}")
    print(f"[Auralis] Individual PDFs/PNGs saved under -> {outdir}")

    # Print summary for each input backtest to console
    for df, lab in zip(dfs, labels):
        print_summary(df, lab)


if __name__ == "__main__":
    main()
