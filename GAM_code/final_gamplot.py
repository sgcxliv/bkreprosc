import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path

# --- Setup and Data Loading ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=0.9)

def load_gam_data() -> pd.DataFrame:
    """Load extracted GAM curve points from known locations."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        # script_dir / "410" / "finalresults" / "gam" / "gam_plot_data_orig.csv",
        # script_dir / "gam_plot_data_orig.csv",
        script_dir / "bko21_SIC_LC_raw_gam_curves.csv",
        # script_dir / "bko_gam_smooth_data.csv",

    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Could not find GAM data CSV. Tried: {tried}"
        )

    df = pd.read_csv(data_path)

    # Backward compatibility: one `model_name` column instead of `model` / `measure`.
    if "model_name" in df.columns and not {"model", "measure"}.issubset(df.columns):
        split = df["model_name"].astype(str).str.rsplit(" ", n=1, expand=True)
        if split.shape[1] == 2:
            df["model"] = split[0].str.replace("^Human ", "", regex=True)
            df["measure"] = split[1]

    # Accept either:
    # - legacy columns: y/lower/upper
    # - extracted columns: fit/se_lower/se_upper
    if {"fit", "se_lower", "se_upper"}.issubset(df.columns):
        df = df.rename(columns={"fit": "y", "se_lower": "lower", "se_upper": "upper"})

    required = {"x", "y", "lower", "upper", "model", "measure"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {data_path}: {sorted(missing)} "
            f"(found: {list(df.columns)})"
        )

    print(f"Loaded data from: {data_path}")
    return df


try:
    df = load_gam_data()
except (FileNotFoundError, ValueError) as e:
    print(f"ERROR: {e}")
    raise SystemExit(1)

# Unique models
model_order = [
    "Cloze", "GPT-2", "GPT-Neo", "GPT-NeoX",
    "GPT-J", "GPT-2XL", "OLMO-2", "LLaMA-2"
]
models_in_order = [m for m in model_order if m in df['model'].unique()]
display_labels = {"GPT-2": "GPT2-small", "gpt2": "GPT2-small"}

# Color palette (consistent per model)
model_palette = sns.color_palette("viridis", len(models_in_order))
model_colors = dict(zip(models_in_order, model_palette))

# Y-limits
y_min = df['lower'].min()
y_max = df['upper'].max()
y_pad = max((y_max - y_min) * 0.08, 1e-4)


# -------------------------------------------------------------------
# Plotting Function
# -------------------------------------------------------------------
def plot_gam_smooth(
    ax,
    data,
    model_color,
    is_surprisal=False,
    show_xlabel=False,
    y_lim=(y_min - y_pad, y_max + y_pad),
):
    """Plots the GAM smooth for a single model."""

    if data.empty:
        ax.set_visible(False)
        return

    # Keep plotting order deterministic and identical to extracted curve points.
    data = data.sort_values("x", kind="mergesort")
    x = data['x']
    y = data['y']
    lower = data['lower']
    upper = data['upper']

    # Plot: Probability (linear)
    if not is_surprisal:
        ax.plot(x, y, color=model_color, linewidth=1.5)
        ax.fill_between(x, lower, upper, alpha=0.25, color=model_color)
        ax.set_xlabel("Probability" if show_xlabel else "", fontsize=15)

        # Pretty linear ticks
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda val, pos: f"{val:.2f}")
        )

    # Plot: Surprisal (linear)
    else:
        ax.plot(x, y, color=model_color, linewidth=1.5)
        ax.fill_between(x, lower, upper, alpha=0.25, color=model_color)
        ax.set_xlabel("Surprisal" if show_xlabel else "", fontsize=15)
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda val, pos: f"{val:g}")
        )

    # Aesthetic cleanup
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    # Use data-driven ticks so log-RT partial effects are visible.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# -------------------------------------------------------------------
# Figure Layout (2x8)
# -------------------------------------------------------------------
N_MODELS = len(models_in_order)
fig, axes = plt.subplots(
    2, N_MODELS, figsize=(18, 5),
    sharey='row', gridspec_kw={'wspace': 0.15, 'hspace': 0.35}
)

if N_MODELS == 1:
    axes = np.array([axes])

# -------------------------------------------------------------------
# ROW 1 – Probability (Linear X)
# -------------------------------------------------------------------
for i, model in enumerate(models_in_order):
    ax = axes[0, i]

    data_subset = df[(df['model'] == model) & (df['measure'] == 'Probability')]
    model_color = model_colors[model]

    plot_gam_smooth(
        ax,
        data_subset,
        model_color,
        is_surprisal=False,
        show_xlabel=(i == 0),
    )

    # Title on top
    ax.set_title(display_labels.get(model, model), fontsize=15, fontweight='bold', pad=5)

    # Y-label only on first plot
    if i == 0:
        ax.set_ylabel("Partial Effect on log(SUM_3RT_trimmed)", fontsize=12)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


# -------------------------------------------------------------------
# ROW 2 – Surprisal (Linear X)
# -------------------------------------------------------------------
for i, model in enumerate(models_in_order):
    ax = axes[1, i]
    data_subset = df[(df['model'] == model) & (df['measure'] == 'Surprisal')]
    model_color = model_colors[model]

    plot_gam_smooth(
        ax,
        data_subset,
        model_color,
        is_surprisal=True,
        show_xlabel=(i == 0),
    )

    if i == 0:
        ax.set_ylabel("Partial Effect on log(SUM_3RT_trimmed)", fontsize=12)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


# -------------------------------------------------------------------
# Row Labels
# -------------------------------------------------------------------
fig.text(0.01, 0.74, 'Probability', va='center',
         rotation='vertical', fontsize=15, fontweight='bold')
fig.text(0.01, 0.27, 'Surprisal', va='center',
         rotation='vertical', fontsize=15, fontweight='bold')

# Main Title
fig.suptitle("GAM Smooths for Language Model Predictability Measures",
             fontsize=14, y=1.02)

# Final formatting
plt.tight_layout(rect=[0.02, 0.0, 1, 1])

plt.savefig("final_2x8_gam_plots_fixed.png", dpi=300, bbox_inches='tight')
print("Saved figure to final_2x8_gam_plots_fixed.png")

plt.show()
