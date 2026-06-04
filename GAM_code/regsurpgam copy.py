#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#
## ============================================================
## CONFIGURATION — adjust fonts, sizes, appearance
## ============================================================
#
#INPUT_CSV   = "gam_plot_data_region_bkr21.csv"
#OUTPUT_FILE = "region_surp_gam_rep.png"
#TITLE       = "GAM Smooths for Region Surprisal"
#
## Figure dimensions
#FIG_WIDTH   = 30
#FIG_HEIGHT  = 4.5
#
## Font sizes
#TITLE_FONTSIZE  = 30
#MODEL_FONTSIZE  = 25
#XLABEL_FONTSIZE = 25
#YLABEL_FONTSIZE = 25
#TICK_FONTSIZE   = 15
#
## Whitespace
#TITLE_PAD  = 20
#XLABEL_PAD = 15
#YLABEL_PAD = 15
#TICK_PAD   = 5
#
## Line / CI appearance
#LINE_WIDTH  = 2.0
#CI_ALPHA    = 0.25
#
## Spacing between subplots
#WSPACE = 0.12
#
## Color palette
#PALETTE = "viridis"
#
## Raw model names in the CSV, in display order
#MODEL_ORDER = [
#    "clozeregion", "gpt2region", "gptneoregion", "gptneoxregion",
#    "gptjregion", "gpt2xlregion", "olmoregion", "llama2region"
#]
#
## Pretty labels for titles
#MODEL_NAMES = {
#    "clozeregion":   "Cloze",
#    "gpt2region":    "GPT-2",
#    "gptneoregion":  "GPT-Neo",
#    "gptneoxregion": "GPT-NeoX",
#    "gptjregion":    "GPT-J",
#    "gpt2xlregion":  "GPT-2XL",
#    "olmoregion":    "OLMO-2",
#    "llama2region":  "LLaMA-2",
#}
#
## ============================================================
## Load data
## ============================================================
#plt.style.use('seaborn-v0_8-whitegrid')
#sns.set_context("paper", font_scale=1.0)
#
#try:
#    df = pd.read_csv(INPUT_CSV)
#except FileNotFoundError:
#    print(f"ERROR: {INPUT_CSV} not found.")
#    exit()
#
#if 'measure' in df.columns:
#    df = df[df['measure'] == 'Surprisal']
#
## Filter to models present in the data, preserving order
#models_raw = [m for m in MODEL_ORDER if m in df['model'].unique()]
#N = len(models_raw)
#
## Colors keyed on raw names (so df filtering still works)
#palette      = sns.color_palette(PALETTE, N)
#model_colors = dict(zip(models_raw, palette))
#
#y_min = df['lower'].min()
#y_max = df['upper'].max()
#y_pad = (y_max - y_min) * 0.05
#y_lim = (y_min - y_pad, y_max + y_pad)
#
## ============================================================
## Build figure
## ============================================================
#fig, axes = plt.subplots(
#    1, N,
#    figsize=(FIG_WIDTH, FIG_HEIGHT),
#    sharey=True,
#    gridspec_kw={'wspace': WSPACE}
#)
#
#if N == 1:
#    axes = [axes]
#
#for i, model_raw in enumerate(models_raw):
#    ax    = axes[i]
#    data  = df[df['model'] == model_raw].sort_values('x')
#    color = model_colors[model_raw]
#
#    ax.plot(data['x'], data['y'], color=color, linewidth=LINE_WIDTH)
#    ax.fill_between(data['x'], data['lower'], data['upper'],
#                    alpha=CI_ALPHA, color=color)
#
#    # Use pretty name for the title
#    ax.set_title(MODEL_NAMES[model_raw], fontsize=MODEL_FONTSIZE,
#                 fontweight='bold', pad=TITLE_PAD)
#    ax.set_xlabel("Surprisal", fontsize=XLABEL_FONTSIZE, labelpad=XLABEL_PAD)
#
#    if i == 0:
#        ax.set_ylabel("Reading Time (ms)", fontsize=YLABEL_FONTSIZE, labelpad=YLABEL_PAD)
#    else:
#        ax.set_ylabel("")
#
#    ax.set_ylim(y_lim)
#    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, pad=TICK_PAD)
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
#    ax.grid(axis='y', linestyle='--', alpha=0.4)
#
## ============================================================
## Overall title & save
## ============================================================
#fig.suptitle(TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold', y=1.1)
#plt.tight_layout(rect=[0, 0, 1, 0.90])
#plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
#print(f"Saved -> {OUTPUT_FILE}")
#plt.show()

#ver 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION — adjust fonts, sizes, appearance
# ============================================================

INPUT_CSV   = "gam_plot_data_region_bkr21.csv"   # swap for bkr21 etc.
OUTPUT_FILE = "region_surp_gam_rep.png"
#OUTPUT_FILE = "region_surp_gam_orig.png"
TITLE       = "GAM Smooths for Region Surprisal"

# Figure dimensions
FIG_WIDTH   = 30      # total width in inches
FIG_HEIGHT  = 4.5     # total height

# Font sizes
TITLE_FONTSIZE  = 30
MODEL_FONTSIZE  = 25   # subplot titles (model names)
XLABEL_FONTSIZE = 25   # "Surprisal" label on each subplot
YLABEL_FONTSIZE = 25   # "Reading Time (ms)"
TICK_FONTSIZE   = 25

# Whitespace
TITLE_PAD  = 20        # space between model name and plot top
XLABEL_PAD = 15        # space between x-axis ticks and "Surprisal" label
YLABEL_PAD = 15        # space between y-axis ticks and "Reading Time" label
TICK_PAD   = 5         # space between tick marks and tick labels

# Line / CI appearance
LINE_WIDTH  = 2.0
CI_ALPHA    = 0.25

# Spacing between subplots
WSPACE = 0.12

# Color palette
PALETTE = "viridis"

# Model display order
MODEL_ORDER = [
    "Cloze", "GPT-2", "GPT-Neo", "GPT-NeoX",
    "GPT-J", "GPT-2XL", "OLMO-2", "LLaMA-2"
]

# Display labels (CSV "model" values stay the same; only presentation changes).
MODEL_DISPLAY_NAMES = {
    "Cloze": "Cloze",
    "GPT-2": "GPT-2-small",
    "GPT-Neo": "GPT-Neo",
    "GPT-NeoX": "GPT-NeoX",
    "GPT-J": "GPT-J",
    "GPT-2XL": "GPT-2XL",
    "OLMO-2": "OLMO-2",
    "LLaMA-2": "LLaMA-2",
}

# ============================================================
# Load data
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.0)

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"ERROR: {INPUT_CSV} not found.")
    exit()

if 'measure' in df.columns:
    df = df[df['measure'] == 'Surprisal']

models = [m for m in MODEL_ORDER if m in df['model'].unique()]
N = len(models)

# Use a stable palette keyed by the full model order, so colors do not shift
# if some models are absent from the CSV.
full_palette = sns.color_palette(PALETTE, len(MODEL_ORDER))
base_color_map = dict(zip(MODEL_ORDER, full_palette))
model_colors = {m: base_color_map[m] for m in models}

y_min = df['lower'].min()
y_max = df['upper'].max()
y_pad = (y_max - y_min) * 0.05
y_lim = (y_min - y_pad, y_max + y_pad)

# ============================================================
# Build figure
# ============================================================
fig, axes = plt.subplots(
    1, N,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    sharey=True,
    gridspec_kw={'wspace': WSPACE}
)

if N == 1:
    axes = [axes]

for i, model in enumerate(models):
    ax    = axes[i]
    data  = df[df['model'] == model].sort_values('x')
    color = model_colors[model]

    ax.plot(data['x'], data['y'], color=color, linewidth=LINE_WIDTH)
    ax.fill_between(data['x'], data['lower'], data['upper'],
                    alpha=CI_ALPHA, color=color)

    ax.set_title(MODEL_DISPLAY_NAMES.get(model, model),
                 fontsize=MODEL_FONTSIZE, fontweight='bold', pad=TITLE_PAD)
    # Only the leftmost subplot gets the x-axis label.
    if i == 0:
        ax.set_xlabel("Surprisal", fontsize=XLABEL_FONTSIZE, labelpad=XLABEL_PAD)
    else:
        ax.set_xlabel("")

    if i == 0:
        ax.set_ylabel("Reading Time (ms)", fontsize=YLABEL_FONTSIZE, labelpad=YLABEL_PAD)
    else:
        ax.set_ylabel("")

    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, pad=TICK_PAD)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

# ============================================================
# Overall title & save
# ============================================================
plt.tight_layout(rect=[0, 0, 1, 0.90])  # reserves the top 5% for the suptitle
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"Saved -> {OUTPUT_FILE}")
plt.show()
