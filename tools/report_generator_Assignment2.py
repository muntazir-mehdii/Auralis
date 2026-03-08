# tools/report_generator.py
# Generate per-baseline reports using make_backtest_report utilities.
#
# This will create:
#   plots/baseline_A/backtest_figures.pdf (and individual PNG/PDFs)
#   plots/baseline_B/...
#   plots/baseline_C/...
#   plots/baseline_D/...

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import helper functions from your existing report script
from backtest_report import (
    load_backtest,
    ensure_outdir,
    plot_equity,
    plot_drawdown,
    plot_distribution,
    plot_scatter,
    plot_winrate_by_hour,
    plot_monthly_performance,
    save_fig,
    print_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_ROOT = ROOT / "plots"


# (baseline_folder_name, csv_filename, plot_title)
BASELINES = [
    (
        "baseline_A",
        "baseline_A_2R.csv",
        "Auralis — Baseline A (2R, Market Entry)",
    ),
    (
        "baseline_B",
        "baseline_B_3R.csv",
        "Auralis — Baseline B (3R, Market Entry)",
    ),
    (
        "baseline_C",
        "baseline_C_2R_asiaFilter.csv",
        "Auralis — Baseline C (2R, Market + Asia Range Filter)",
    ),
    (
        "baseline_D",
        "baseline_D_2R_retrace.csv",
        "Auralis — Baseline D (2R, Retrace Entry)",
    ),
]


def generate_report_for_baseline(folder_name: str, csv_name: str, title: str) -> None:
    csv_path = RESULTS_DIR / csv_name
    if not csv_path.exists():
        print(f"[WARN] CSV not found for {folder_name}: {csv_path}")
        return

    df = load_backtest(csv_path)

    outdir = PLOTS_ROOT / folder_name
    ensure_outdir(outdir)

    pdf_path = outdir / "backtest_figures.pdf"
    with PdfPages(pdf_path) as pdf_pages:
        # Equity curve
        fig_eq = plot_equity([df], [folder_name], title)
        save_fig(fig_eq, outdir, f"{folder_name}_equity", pdf_pages)

        # Drawdown curve
        fig_dd = plot_drawdown([df], [folder_name])
        save_fig(fig_dd, outdir, "DrawDownCurve_Date", pdf_pages)

        # Distribution of R
        fig_hist = plot_distribution(df)
        save_fig(fig_hist, outdir, "DistributionPerTrade", pdf_pages)

        # Scatter of R per trade
        fig_scatter = plot_scatter(df)
        save_fig(fig_scatter, outdir, "PerTrade_RScatter", pdf_pages)

        # Win rate by hour
        fig_hour = plot_winrate_by_hour(df)
        save_fig(fig_hour, outdir, "WinRate_time", pdf_pages)

        # Monthly performance
        fig_month = plot_monthly_performance(df)
        save_fig(fig_month, outdir, "MonthlyPerformace", pdf_pages)

    print(f"\n[Auralis] {folder_name}: figures written -> {pdf_path}")
    print_summary(df, folder_name)


def main():
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    for folder_name, csv_name, title in BASELINES:
        print(f"\n=== Generating report for {folder_name} ===")
        generate_report_for_baseline(folder_name, csv_name, title)


if __name__ == "__main__":
    main()
