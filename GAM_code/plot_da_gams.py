import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys

# ------------------------------------------------------------------ #
# Config — change label to "bko21" or "bkr21"
# ------------------------------------------------------------------ #
DATASET   = "bko21"   # <- change to bko21 to plot the other dataset
INPUT_CSV = f"{DATASET}_gam_curves.csv"
OUTPUT    = f"{DATASET}_gam_agree_vs_disagree.pdf"

# Model order (left to right, matches reference figure)
MODEL_ORDER = ["GPT-2", "GPT-Neo", "GPT-NeoX", "GPT-J", "GPT-2XL", "OLMO-2", "LLaMA-2"]

MODEL_COLORS = {
    "GPT-2": "#5B4E8A",   # deep purple
    "GPT-Neo":  "#4B6CA8",   # blue-purple
    "GPT-NeoX": "#3D8FA6",   # teal-blue
    "GPT-J":    "#3DA694",   # teal
    "GPT-2XL":  "#3DAF82",   # teal-green
    "OLMO-2":   "#5BBF6A",   # medium green
    "LLaMA-2":  "#A8CE3A",  # yellow-green
}

FALLBACK_COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, len(MODEL_ORDER)))

DISPLAY_LABELS = {
    "GPT-2": "gpt2-small",
    "gpt2": "gpt2-small",
}

SCALE = 1.6
FS_NO_DATA = 8 * SCALE
FS_TICKS = 8 * SCALE * 2.0
FS_XY_LABEL = 9 * SCALE
FS_PANEL_TITLE = 10 * SCALE
FS_ROW_LABEL = 11 * SCALE
FS_SUPTITLE = 11 * SCALE * 1.6

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"ERROR: {INPUT_CSV} not found.")
    print("Run extract_gam_curves.R first to generate the curve CSVs.")
    sys.exit(1)

# Validate
required_cols = {"x", "fit", "se_upper", "se_lower", "model", "condition"}
missing = required_cols - set(df.columns)
if missing:
    print(f"ERROR: Missing columns in {INPUT_CSV}: {missing}")
    sys.exit(1)

# Only keep models we expect
df = df[df["model"].isin(MODEL_ORDER)].copy()
df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
n_models = len(MODEL_ORDER)
# Make figure about 20% taller to reduce crowding.
fig = plt.figure(figsize=(3.1 * n_models, 8.4))

# Two rows (agree / disagree), n_models columns
# Row labels on left, model names on top
gs = gridspec.GridSpec(
    2, n_models,
    figure=fig,
    hspace=0.35,
    wspace=0.25,
    left=0.07, right=0.98,
    top=0.84,  bottom=0.10
)

row_labels = ["Agree", "Disagree"]
conditions = ["agree", "disagree"]

axes = {}
for row_idx, (cond, row_label) in enumerate(zip(conditions, row_labels)):
    for col_idx, model_name in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        axes[(row_idx, col_idx)] = ax

        # Get color
        color = MODEL_COLORS.get(model_name, FALLBACK_COLORS[col_idx])

        subset = df[(df["model"] == model_name) & (df["condition"] == cond)]

        if subset.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FS_NO_DATA, color="grey")
        else:
            subset = subset.sort_values("x")

            # Shaded CI
            ax.fill_between(
                subset["x"],
                subset["se_lower"],
                subset["se_upper"],
                alpha=0.25,
                color=color,
                linewidth=0
            )

            # Fitted line
            ax.plot(subset["x"], subset["fit"],
                    color=color, linewidth=1.8)

            # Zero reference line
            ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)

        # ---- Axis formatting ----
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FS_TICKS)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_yticks([100, 50, 0, -50, -100])

        # x-axis label only once: bottom-left panel (under GPT-2/GPT-2-small)
        if row_idx == 1 and col_idx == 0:
            ax.set_xlabel("Surprisal", fontsize=FS_XY_LABEL)
        else:
            ax.set_xticklabels([])

        # y-axis label only on leftmost column
        if col_idx == 0:
            ax.set_ylabel("Reading Time (ms)", fontsize=FS_XY_LABEL)
        else:
            ax.set_yticklabels([])

        # Model name on top row only
        if row_idx == 0:
            title_label = DISPLAY_LABELS.get(model_name, model_name)
            ax.set_title(title_label, fontsize=FS_PANEL_TITLE, fontweight="bold", pad=6)

# Row labels on the far left
for row_idx, row_label in enumerate(row_labels):
    fig.text(
        0.01, 0.75 - row_idx * 0.5,
        row_label,
        fontsize=FS_ROW_LABEL, fontweight="bold",
        va="center", ha="left",
        rotation=90
    )

# Overall title
dataset_tag = DATASET.lower()
if dataset_tag.endswith("21"):
    dataset_tag = dataset_tag[:-2]
fig.suptitle(
    f"GAM Surprisals for Agree vs. Disagree Subsets ({dataset_tag})",
    fontsize=FS_SUPTITLE, fontweight="bold", y=0.98
)

plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT} and {OUTPUT.replace('.pdf', '.png')}")
plt.show()
