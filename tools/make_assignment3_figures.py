# -*- coding: utf-8 -*-
"""
Auralis Assignment 3 — Auto Figure Generator (No Arguments Needed)

Automatically loads:
  - results/baseline_D_2R_retrace.csv
  - results/proposed_v2_thr_50.csv
  - results/baseline_vs_proposed_v2.csv
  - results/confusion_matrix_v2.csv
  - models/xgb_filter_v2.json

Outputs:
  plots/proposed_v2/equity_curve.png
  plots/proposed_v2/equity_curve_example.png
  plots/proposed_v2/threshold_sweep.png
  plots/proposed_v2/feature_importance.png
  results/confusion_matrix_v2.png

Just press F5 to run it from the project root (Auralis).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

# -------------------------------------------
# Try loading XGBoost model (optional)
# -------------------------------------------
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

ROOT = Path(__file__).resolve().parents[1]

# Auto-detected file paths (fixed to your layout)
PATH_BASELINE = ROOT / "results" / "baseline_D_2R_retrace.csv"
PATH_PROP_V2 = ROOT / "results" / "proposed_v2_thr_50.csv"
PATH_BVP = ROOT / "results" / "baseline_vs_proposed_v2.csv"
PATH_CM = ROOT / "results" / "confusion_matrix_v2.csv"
PATH_MODEL = ROOT / "models" / "xgb_filter_v2.json"

OUT_EQ = ROOT / "plots" / "proposed_v2"
OUT_CM_PNG = ROOT / "results" / "confusion_matrix_v2.png"


# ===============================================================
# Helpers
# ===============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def check_file(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"[Auralis] Missing {desc}: {path}")
    return path


def load_backtest_csv(path: Path) -> pd.DataFrame:
    path = check_file(path, "backtest CSV")
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # detect time column
    if "time" in cols:
        tcol = cols["time"]
    elif "date" in cols:
        tcol = cols["date"]
    elif "entry_time" in cols:
        tcol = cols["entry_time"]
    else:
        raise ValueError(f"{path}: no valid time/date/entry_time column found")

    # detect R column
    if "total_r" in cols:
        rcol = cols["total_r"]
    elif "result_r" in cols:
        rcol = cols["result_r"]
    elif "r" in cols:
        rcol = cols["r"]
    else:
        raise ValueError(f"{path}: no R column found (total_R/result_R/r)")

    df["time"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    df["r"] = df[rcol].astype(float)
    df["cum_r"] = df["r"].cumsum()
    df["dd_r"] = df["cum_r"] - df["cum_r"].cummax()
    return df


# ===============================================================
# PLOTTING FUNCTIONS
# ===============================================================

def plot_equity(df_base, df_prop):
    ensure_dir(OUT_EQ)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(df_base["time"], df_base["cum_r"],
            label="Baseline_D_2R_retrace", linewidth=1.5)
    ax.plot(df_prop["time"], df_prop["cum_r"],
            label="Proposed V2 (θ=0.50)", linewidth=1.5)

    ax.set_title("Cumulative R vs Time")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Cumulative R")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_EQ / "equity_curve.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_EQ / "equity_curve_example.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix():
    path = check_file(PATH_CM, "confusion matrix CSV")

    # Try a few ways to read it, but DO NOT crash if shape is weird.
    try:
        df = pd.read_csv(path, header=None)
    except Exception as e:
        print(f"[Auralis] WARNING: could not read confusion matrix CSV ({e}) — skipping.")
        return

    # If it's not 2x2, try alternative parsing once
    if df.shape != (2, 2):
        try:
            df2 = pd.read_csv(path, index_col=0)
            if df2.shape == (2, 2):
                df = df2
            else:
                print(f"[Auralis] WARNING: confusion_matrix_v2.csv is shape {df.shape}, "
                      "expected 2x2 — skipping confusion-matrix figure.")
                return
        except Exception:
            print(f"[Auralis] WARNING: confusion_matrix_v2.csv is shape {df.shape}, "
                  "expected 2x2 — skipping confusion-matrix figure.")
            return

    cm = df.values.astype(float)

    ensure_dir(OUT_CM_PNG.parent)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Loss (0)", "Win (1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Loss (0)", "Win (1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=11)

    ax.set_title("Confusion Matrix — Proposed V2")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUT_CM_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_threshold_sweep():
    ensure_dir(OUT_EQ)
    path = check_file(PATH_BVP, "baseline_vs_proposed_v2.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # keep only ProposedV2 rows
    mask = df["model"].astype(str).str.contains("ProposedV2", case=False, na=False)
    sub = df.loc[mask].copy()
    if sub.empty:
        print("[Auralis] WARNING: No ProposedV2 rows found in baseline_vs_proposed_v2.csv, "
              "skipping threshold sweep plot.")
        return

    # theta from column or from name
    if "threshold" in sub.columns and sub["threshold"].notna().any():
        sub["theta"] = sub["threshold"]
    else:
        sub["theta"] = (
            sub["model"].astype(str)
            .str.extract(r"thr[_\- ]?([0-9]*\.?[0-9]+)", expand=False)
            .astype(float)
        )

    sub = sub.sort_values("theta")

    fig, ax1 = plt.subplots(figsize=(7.8, 4.5))
    ax1.plot(sub["theta"], sub["total_R"], marker="o", linewidth=1.8)
    ax1.set_xlabel(r"Threshold $\theta$")
    ax1.set_ylabel("Total R")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(sub["theta"], sub["trades"], marker="s", linestyle="--")
    ax2.set_ylabel("Trades")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title("Proposed V2 — Threshold Sweep (Total R & Trades)")
    fig.tight_layout()
    fig.savefig(OUT_EQ / "threshold_sweep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance():
    if not HAVE_XGB:
        print("[Auralis] WARNING: xgboost not installed — skipping feature importance.")
        return

    if not PATH_MODEL.exists():
        print(f"[Auralis] WARNING: model JSON not found at {PATH_MODEL} — skipping feature importance.")
        return

    model = XGBClassifier()
    model.load_model(str(PATH_MODEL))

    feat_names = None
    meta = PATH_MODEL.parent / "xgb_filter_meta_v2.parquet"
    if meta.exists():
        try:
            meta_df = pd.read_parquet(meta)
            col_list = meta_df["feature_cols"].iloc[0]
            # col_list might be list or stringified list
            if isinstance(col_list, list):
                feat_names = col_list
            else:
                import ast
                feat_names = list(ast.literal_eval(col_list))
        except Exception:
            feat_names = None

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[Auralis] WARNING: model has no feature_importances_ attribute.")
        return

    idx = np.argsort(importances)[::-1][:20]
    vals = importances[idx]
    if feat_names:
        names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in idx]
    else:
        names = [f"f{i}" for i in idx]

    ensure_dir(OUT_EQ)
    fig, ax = plt.subplots(figsize=(7.8, 5))
    y = np.arange(len(names))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance — Model V2")
    fig.tight_layout()
    fig.savefig(OUT_EQ / "feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ===============================================================
# MAIN
# ===============================================================

def main():
    print("\n[Auralis] Auto-Generating Assignment 3 Figures...")

    df_base = load_backtest_csv(PATH_BASELINE)
    df_prop = load_backtest_csv(PATH_PROP_V2)

    plot_equity(df_base, df_prop)
    plot_confusion_matrix()
    plot_threshold_sweep()
    plot_feature_importance()

    print("\n[Auralis] Figures saved in:")
    print(f"  {OUT_EQ}")
    print(f"  {OUT_CM_PNG}")
    print("[Auralis] Done! 🎉\n")


if __name__ == "__main__":
    main()
