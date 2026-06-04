
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ============================================================
# CONFIGURATION — adjust fonts, sizes, appearance
# ============================================================

INPUT_CSV   = "gam_agree_disagree.csv"
OUTPUT_FILE = "gam_agree_disagree_plot_v2.png"
TITLE       = " "

# Figure dimensions
FIG_WIDTH   = 30
FIG_HEIGHT  = 9.0      # doubled from original to accommodate 2 rows

# Font sizes
TITLE_FONTSIZE      = 30
MODEL_FONTSIZE      = 25
ROW_LABEL_FONTSIZE  = 22
XLABEL_FONTSIZE     = 25
YLABEL_FONTSIZE     = 25
TICK_FONTSIZE       = 15

# Whitespace
TITLE_PAD  = 20
XLABEL_PAD = 15
YLABEL_PAD = 15
TICK_PAD   = 5

# Line / CI appearance
LINE_WIDTH  = 2.0
CI_ALPHA    = 0.25

# Subplot spacing
WSPACE = 0.12
HSPACE = 0.45

# Color palette (same as original)
PALETTE = "viridis"

# Model display order (no Cloze in agree/disagree data)
MODEL_ORDER = ["GPT-2", "GPT-Neo", "GPT-NeoX", "GPT-J", "GPT-2XL", "OLMO-2", "LLaMA-2"]

# Row config
SUBSETS    = ["agree", "disagree"]
ROW_LABELS = {"agree": "Agree", "disagree": "Disagree"}

# ============================================================
# Load & filter
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.0)

df = pd.read_csv(INPUT_CSV)

if 'measure' in df.columns:
    df = df[df['measure'] == 'Surprisal']

models = [m for m in MODEL_ORDER if m in df['model'].unique()]
N = len(models)

palette      = sns.color_palette(PALETTE, N)
model_colors = dict(zip(models, palette))

# Shared y-limits PER ROW (so agree row and disagree row each have their own scale)
def row_ylim(subset):
    sub = df[df['subset'] == subset]
    lo, hi = sub['lower'].min(), sub['upper'].max()
    pad = (hi - lo) * 0.05
    return (lo - pad, hi + pad)

ylims = {s: row_ylim(s) for s in SUBSETS}

# ============================================================
# Build figure — 2 rows × N cols via GridSpec
# ============================================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')

gs = gridspec.GridSpec(
    2, N,
    figure=fig,
    hspace=HSPACE,
    wspace=WSPACE,
    left=0.055, right=0.99,
    top=0.88,   bottom=0.09,
)

for row_i, subset in enumerate(SUBSETS):
    sub_df = df[df['subset'] == subset]
    ylim   = ylims[subset]

    for col_i, model in enumerate(models):
        ax   = fig.add_subplot(gs[row_i, col_i])
        data = sub_df[sub_df['model'] == model].sort_values('x')
        c    = model_colors[model]

        ax.plot(data['x'], data['y'], color=c, linewidth=LINE_WIDTH)
        ax.fill_between(data['x'], data['lower'], data['upper'],
                        alpha=CI_ALPHA, color=c)

        # Model name only on top row
        if row_i == 0:
            ax.set_title(model, fontsize=MODEL_FONTSIZE,
                         fontweight='bold', pad=TITLE_PAD)

        # X label only on bottom row
        if row_i == 1:
            ax.set_xlabel("Surprisal", fontsize=XLABEL_FONTSIZE,
                          labelpad=XLABEL_PAD)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        # Y label only on leftmost column
        if col_i == 0:
            ax.set_ylabel("Reading Time (ms)", fontsize=YLABEL_FONTSIZE,
                          labelpad=YLABEL_PAD)
        else:
            ax.set_ylabel("")

        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major',
                       labelsize=TICK_FONTSIZE, pad=TICK_PAD)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Row label (Agree / Disagree) to the left of each row
    row_y = 0.735 if row_i == 0 else 0.275
    fig.text(
        0.003, row_y,
        ROW_LABELS[subset],
        va='center', ha='left',
        fontsize=ROW_LABEL_FONTSIZE,
        fontweight='bold',
        color='#333333',
        rotation=90,
    )

# ============================================================
# Overall title & save
# ============================================================
fig.suptitle(TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold', y=0.97)

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved -> {OUTPUT_FILE}")
plt.show()
