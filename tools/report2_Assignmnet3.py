def plot_confusion_matrix() -> None:
    path = check_file(PATH_CM, "confusion matrix CSV")
    if path is None:
        return

    try:
        # Try without header first; this will give us raw numeric-ish matrix
        df = pd.read_csv(path, header=None)
    except Exception as e:
        print(f"[Auralis] WARNING: could not read confusion matrix CSV ({e}) — skipping.")
        return

    # If it's not at least 2x2, we really can't do anything sensible
    if df.shape[0] < 2 or df.shape[1] < 2:
        print(f"[Auralis] WARNING: confusion_matrix_v2.csv is shape {df.shape}, "
              "needs at least 2x2 — skipping confusion-matrix figure.")
        return

    # If it's exactly 2x2, use it as-is.
    # If it's bigger (e.g. 3x2, 3x3), take the bottom-right 2x2 block.
    if df.shape != (2, 2):
        print(f"[Auralis] INFO: confusion_matrix_v2.csv is shape {df.shape}, "
              "using bottom-right 2x2 block as confusion matrix.")
        df = df.iloc[-2:, -2:]

    cm = df.values.astype(float)
    ensure_dir(OUT_CM_PNG.parent)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Loss (0)", "Win (1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Loss (0)", "Win (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Proposed V2")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUT_CM_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
