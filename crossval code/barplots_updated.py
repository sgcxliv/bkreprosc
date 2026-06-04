#!/usr/bin/env python3
"""
Bar plots for all cross-validation results.
Loads everything directly from bkresults_fdr.csv (Replication) and
bkoresults_fdr.csv (B&K Original).

Produces 12 plots:
  01_rep_word.png            Replication: word-level
  02_bko_word.png            B&K Original: word-level
  03_rep_region.png          Replication: region-level
  04_bko_region.png          B&K Original: region-level
  05_rep_cloze_word.png      Replication: Cloze vs models, word
  06_bko_cloze_word.png      B&K Original: Cloze vs models, word
  07_rep_cloze_region.png    Replication: Cloze vs models, region
  08_bko_cloze_region.png    B&K Original: Cloze vs models, region
  09_rep_surp_vs_region.png  Replication: within-model Surp vs Region Surp
  10_bko_surp_vs_region.png  B&K Original: within-model Surp vs Region Surp
  10a_rep_clozeprob_word.png    Replication: ClozeProb vs model word surprisal
  10a_bko_clozeprob_word.png    B&K Original: ClozeProb vs model word surprisal
  10b_rep_clozeprob_region.png  Replication: ClozeProb vs model region surprisal
  10b_bko_clozeprob_region.png  B&K Original: ClozeProb vs model region surprisal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb

# ============================================================================
# FILE PATHS
# ============================================================================

BKR_CSV = "bkr_results_fdr.csv"
BKO_CSV = "bko_results_fdr.csv"

# ============================================================================
# CANONICAL MODEL NAMES
# ============================================================================

WORD_MODELS = {
    "cloze":   "Cloze",
    "gpt2":    "GPT-2-small",
    "gptneo":  "GPT-Neo",
    "gptneox": "GPT-NeoX",
    "gptj":    "GPT-J",
    "gpt2xl":  "GPT-2XL",
    "olmo":    "OLMO-2",
    "llama2":  "LLaMA-2",
}

REGION_MODELS = {
    "clozeregion":   "Cloze",
    "gpt2region":    "GPT-2-small",
    "gptneoregion":  "GPT-Neo",
    "gptneoxregion": "GPT-NeoX",
    "gptjregion":    "GPT-J",
    "gpt2xlregion":  "GPT-2XL",
    "olmoregion":    "OLMO-2",
    "llama2region":  "LLaMA-2",
}


MODEL_ORDER = [
    "Cloze", "GPT-2-small", "GPT-Neo", "GPT-NeoX",
    "GPT-J", "GPT-2XL", "OLMO-2", "LLaMA-2",
]

_full_palette  = sns.color_palette("viridis", len(MODEL_ORDER))  # line colors
BASE_MODEL_COLORS   = dict(zip(MODEL_ORDER, _full_palette))

CI_ALPHA_FOR_BAR = 0.7

def color_as_ci_fill(color, ci_alpha: float = CI_ALPHA_FOR_BAR):
    rgb = to_rgb(color)
    return tuple(ci_alpha * c + (1 - ci_alpha) * 1.0 for c in rgb)

MODEL_COLORS = {m: color_as_ci_fill(BASE_MODEL_COLORS[m]) for m in MODEL_ORDER}

DISPLAY_ORDER  = MODEL_ORDER  

# ============================================================================
# NAME NORMALIZATION
# ============================================================================

def normalize(name: str) -> str:
    n = str(name).strip()
    n = n.replace("newprob_x", "prob")    # gpt2newprob_x  -> gpt2prob
    n = n.replace("new_x",     "")        # gpt2new_x      -> gpt2
    n = n.replace("prob_x",    "prob")    # clozeprob_x    -> clozeprob
    n = n.replace("regionnew", "region")  # gpt2regionnew  -> gpt2region
    return n

# ============================================================================
# DATA LOADING
# ============================================================================

def load(path):
    df = pd.read_csv(path)
    df["_left"]  = df["Left Model"].apply(normalize)
    df["_right"] = df["Right Model"].apply(normalize)
    return df

def get_rows(df, comp_type, has_region=None):
    mask = df["Comparison Type"] == comp_type
    if has_region is not None:
        mask &= df["Has Region"] == has_region
    return df[mask].copy()

# ============================================================================
# BUILD DATAFRAMES
# ============================================================================

def build_standard_df(df, has_region: bool) -> pd.DataFrame:
    model_map = REGION_MODELS if has_region else WORD_MODELS

    null_prob = get_rows(df, "Null vs. PROB",     has_region)
    null_surp = get_rows(df, "Null vs. Base",     has_region)
    # "Surp over Both": surprisal-only vs joint (prob+surp) — always Base vs. Combined
    surp_both = get_rows(df, "Base vs. Combined", has_region)
    # "Prob over Both": probability-only vs joint — separate comparison type in CSV
    prob_both = get_rows(df, "PROB vs. Combined", has_region)

    rows = []
    for canon, pretty in model_map.items():
        prob_c = canon + "prob"

        def null_val(sub, rc):
            r = sub[sub["_right"] == rc]
            return (float(r["ΔLL"].iloc[0]), float(r["Raw p-value"].iloc[0])) if len(r) else (np.nan, np.nan)

        def left_val(sub, lc):
            r = sub[sub["_left"] == lc]
            return (float(r["ΔLL"].iloc[0]), float(r["Raw p-value"].iloc[0])) if len(r) else (np.nan, np.nan)

        op_dll, op_p = null_val(null_prob, prob_c)
        os_dll, os_p = null_val(null_surp, canon)
        pb_dll, pb_p = left_val(prob_both, prob_c)
        sb_dll, sb_p = left_val(surp_both, canon)

        rows.append(dict(
            Model=pretty,
            O_vs_Prob_DLL=op_dll,    O_vs_Prob_P=op_p,
            O_vs_Surp_DLL=os_dll,    O_vs_Surp_P=os_p,
            Prob_vs_Both_DLL=pb_dll, Prob_vs_Both_P=pb_p,
            Surp_vs_Both_DLL=sb_dll, Surp_vs_Both_P=sb_p,
        ))
    return pd.DataFrame(rows)


def build_cloze_df(df, has_region: bool) -> pd.DataFrame:
    """Cloze (surprisal) vs each model (surprisal) — bakeoff rows."""
    model_map   = REGION_MODELS if has_region else WORD_MODELS
    left_canon  = "clozeregion" if has_region else "cloze"
    bakeoff     = get_rows(df, "Model Bakeoff (PT)", has_region)
    cloze_rows  = bakeoff[bakeoff["_left"] == left_canon]

    rows = []
    for canon, pretty in model_map.items():
        if canon == left_canon:
            continue
        r = cloze_rows[cloze_rows["_right"] == canon]
        if len(r) == 0:
            continue
        rows.append(dict(Model=pretty,
                         DLL=float(r["ΔLL"].iloc[0]),
                         P=float(r["Raw p-value"].iloc[0])))
    return pd.DataFrame(rows)


def build_clozeprob_df(df, has_region: bool) -> pd.DataFrame:
    """ClozeProb vs each model's surprisal (PT rows under Cloze vs. Other).

    Word: 'ClozeProb vs. Model Surp (PT)', left = clozeprob, right = word surp.
    Region: 'ClozeProb vs. Model Region Surp (PT)', left = clozeprob,
            right = region surprisal (gpt2region, …).

    ΔLL = right LL − left LL  (positive → model surp beats ClozeProb)
    """
    model_map = REGION_MODELS if has_region else WORD_MODELS
    comp = (
        "ClozeProb vs. Model Region Surp (PT)"
        if has_region
        else "ClozeProb vs. Model Surp (PT)"
    )
    left_canon = "clozeprob"

    sub = get_rows(df, comp, has_region)
    cp_rows = sub[sub["_left"] == left_canon]

    rows = []
    for canon, pretty in model_map.items():
        if canon in ("cloze", "clozeregion"):
            continue
        r = cp_rows[cp_rows["_right"] == canon]
        if len(r) == 0:
            continue
        rows.append(dict(Model=pretty,
                         DLL=float(r["ΔLL"].iloc[0]),
                         P=float(r["Raw p-value"].iloc[0])))
    return pd.DataFrame(rows, columns=["Model", "DLL", "P"])


def build_surp_vs_region_df(df) -> pd.DataFrame:
    sub  = get_rows(df, "Within-model Surp vs Region Surp (PT)")
    rows = []
    for canon, pretty in WORD_MODELS.items():
        r = sub[sub["_left"] == canon]
        if len(r) == 0:
            continue
        rows.append(dict(Model=pretty,
                         DLL=float(r["ΔLL"].iloc[0]),
                         P=float(r["Raw p-value"].iloc[0])))
    return pd.DataFrame(rows)

# ============================================================================
# PLOTTING
# ============================================================================

TITLE_FONTSIZE       = 24
AXIS_LABEL_FONTSIZE  = 22
TICK_LABEL_FONTSIZE  = 20
LEGEND_FONTSIZE      = 17
STAR_FONTSIZE        = 17
GROUP_LABEL_FONTSIZE = 22
BAR_WIDTH            = 0.28
GROUP_GAP            = 0.4
Y_PADDING            = 0.20


def is_sig(p):
    try: return float(p) < 0.05
    except: return False


def add_star(ax, x, y, p):
    if not is_sig(p):
        return
    ylo, yhi = ax.get_ylim()
    offset = (yhi - ylo) * 0.02
    if y >= 0:
        ax.text(x, y + offset, "*", ha="center", va="bottom",
                fontsize=STAR_FONTSIZE, fontweight="bold", color="black")
    else:
        ax.text(x, y - offset, "*", ha="center", va="top",
                fontsize=STAR_FONTSIZE, fontweight="bold", color="black")


def plot_grouped_bars(data_df, comparison_cols, comparison_labels,
                      title, ylabel, output_file, figsize=(14, 7),
                      xticklabel_fontsize=TICK_LABEL_FONTSIZE,
                      models_to_plot=None, show_legend=True):
    """
    models_to_plot : list of display names to include (in order).
                     Defaults to the full DISPLAY_ORDER.
                     Colors are always drawn from MODEL_COLORS keyed by name,
                     so skipping a model does NOT shift the palette.
    show_legend : if False, no model color key (plots 06+ use this).
    """
    if models_to_plot is None:
        models_to_plot = DISPLAY_ORDER

    fig, ax = plt.subplots(figsize=figsize)

    current_x     = 0
    group_centers = []
    all_values    = []

    for dll_col, _ in comparison_cols:
        if dll_col in data_df.columns:
            all_values.extend(data_df[dll_col].dropna().tolist())

    y_min = min(all_values) if all_values else 0
    y_max = max(all_values) if all_values else 1
    y_buf = (y_max - y_min) * Y_PADDING
    if y_min >= 0:
        y_min = 0
    if y_max <= 0:
        y_max = 0

    for group_idx, ((dll_col, p_col), label) in enumerate(
            zip(comparison_cols, comparison_labels)):
        group_start = current_x

        for model in models_to_plot:
            row = data_df[data_df["Model"] == model]
            if len(row) == 0 or dll_col not in data_df.columns:
                current_x += BAR_WIDTH
                continue
            dll = float(row[dll_col].iloc[0])
            p   = float(row[p_col].iloc[0]) if p_col in data_df.columns else np.nan

            ax.bar(
                current_x,
                dll,
                BAR_WIDTH,
                color=MODEL_COLORS[model],
                edgecolor="none",
                linewidth=0,
                label=model if group_idx == 0 else "",
            )
            add_star(ax, current_x, dll, p)
            current_x += BAR_WIDTH

        group_centers.append(group_start + (len(models_to_plot) * BAR_WIDTH) / 2)
        current_x += GROUP_GAP

    ax.set_xticks(group_centers)
    ax.set_xticklabels(comparison_labels, fontsize=xticklabel_fontsize,
                       fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=26)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    bottom = y_min if y_min < 0 else 0
    top    = y_max if y_max > 0 else 0
    ax.set_ylim(bottom - (y_buf if y_min < 0 else 0),
                top    + (y_buf if y_max > 0 else 0))
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    if show_legend:
        handles, labs = ax.get_legend_handles_labels()
        ax.legend(handles, labs, loc="best", ncol=2, frameon=True,
                  fancybox=True, shadow=True, fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ {output_file}")
    plt.close()


def plot_single_group_bars(data_df, dll_col, p_col,
                            title, ylabel, output_file, figsize=(10, 7),
                            models_to_plot=None, show_legend=True):
    plot_grouped_bars(data_df, [(dll_col, p_col)], [""],
                      title, ylabel, output_file, figsize=figsize,
                      models_to_plot=models_to_plot, show_legend=show_legend)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("Loading CSVs...")
    print("="*70)

    bkr = load(BKR_CSV)
    bko = load(BKO_CSV)

    standard_cols   = [
        ("O_vs_Prob_DLL",    "O_vs_Prob_P"),
        ("O_vs_Surp_DLL",    "O_vs_Surp_P"),
        ("Prob_vs_Both_DLL", "Prob_vs_Both_P"),
        ("Surp_vs_Both_DLL", "Surp_vs_Both_P"),
    ]
    standard_labels = ["Prob over Ø", "Surp over Ø", "Both over Prob", "Both over Surp"]

    # Models that appear in the standard plots (all 8, including Cloze).
    # Colors stay stable because MODEL_COLORS is keyed by name.
    all_models    = MODEL_ORDER                          # all 8
    no_cloze      = [m for m in MODEL_ORDER if m != "Cloze"]   # 7 LMs

    print("\nGenerating plots...\n")

    # ---- 01–04: Standard grouped bar plots (all 8 models) ------------------

    plot_grouped_bars(build_standard_df(bkr, False), standard_cols, standard_labels,
        "Model Comparisons", "ΔLL (Log-Likelihood Difference)",
        "01_rep_word.png", figsize=(16, 7),
        xticklabel_fontsize=GROUP_LABEL_FONTSIZE,
        models_to_plot=all_models)

    plot_grouped_bars(build_standard_df(bko, False), standard_cols, standard_labels,
        "Model Comparisons", "ΔLL (Log-Likelihood Difference)",
        "02_bko_word.png", figsize=(16, 7),
        xticklabel_fontsize=GROUP_LABEL_FONTSIZE,
        models_to_plot=all_models)

    plot_grouped_bars(build_standard_df(bkr, True), standard_cols, standard_labels,
        "Region-Level Comparisons", "ΔLL (Log-Likelihood Difference)",
        "03_rep_region.png", figsize=(16, 7),
        xticklabel_fontsize=GROUP_LABEL_FONTSIZE,
        models_to_plot=all_models)

    plot_grouped_bars(build_standard_df(bko, True), standard_cols, standard_labels,
        "Region-Level Comparisons", "ΔLL (Log-Likelihood Difference)",
        "04_bko_region.png", figsize=(16, 7),
        xticklabel_fontsize=GROUP_LABEL_FONTSIZE,
        models_to_plot=all_models)

    # ---- 05–06: Cloze vs models, word-level --------------------------------
    # Sign is flipped so bars are positive (Model − Cloze → how much worse model is).
    # Cloze is the baseline here, so only the 7 LMs are plotted.
    # Model color legend only on 05 (plots 01–05); 06+ omit the legend.

    cloze_word_bkr = build_cloze_df(bkr, False)
    cloze_word_bkr["DLL"] = -cloze_word_bkr["DLL"]
    plot_single_group_bars(cloze_word_bkr, "DLL", "P",
        "Model over Cloze (Word-Level)", "ΔLL (Model − Cloze)",
        "05_rep_cloze_word.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    cloze_word_bko = build_cloze_df(bko, False)
    cloze_word_bko["DLL"] = -cloze_word_bko["DLL"]
    plot_single_group_bars(cloze_word_bko, "DLL", "P",
        "Model over Cloze (Word-Level)", "ΔLL (Model − Cloze)",
        "06_bko_cloze_word.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    # ---- 07–08: Cloze vs models, region-level ------------------------------

    plot_single_group_bars(build_cloze_df(bkr, True), "DLL", "P",
        "Model over Cloze (Region-Level)", "ΔLL (Cloze − Model)",
        "07_rep_cloze_region.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    plot_single_group_bars(build_cloze_df(bko, True), "DLL", "P",
        "Model over Cloze (Region-Level)", "ΔLL (Cloze − Model)",
        "08_bko_cloze_region.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    # ---- 09–10: Within-model Surp vs Region Surp ---------------------------

    plot_single_group_bars(build_surp_vs_region_df(bkr), "DLL", "P",
        "Within-Model: Region Surp over Word Surp",
        "ΔLL (Word Surp − Region Surp)",
        "09_rep_surp_vs_region.png", figsize=(10, 7),
        models_to_plot=all_models, show_legend=False)

    plot_single_group_bars(build_surp_vs_region_df(bko), "DLL", "P",
        "Within-Model: Region Surp over Word Surp",
        "ΔLL (Word Surp − Region Surp)",
        "10_bko_surp_vs_region.png", figsize=(10, 7),
        models_to_plot=all_models, show_legend=False)

    # ---- 10a: ClozeProb vs model word surprisal ----------------------------
    # Left = clozeprob, Right = model word surprisal.
    # ΔLL positive → model word surp beats ClozeProb.
    # Cloze (surprisal) is excluded; only the 7 LMs appear.

    plot_single_group_bars(build_clozeprob_df(bkr, False), "DLL", "P",
        "Model Word Surprisal over ClozeProb",
        "ΔLL (Model Word Surp − ClozeProb)",
        "10a_rep_clozeprob_word.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    plot_single_group_bars(build_clozeprob_df(bko, False), "DLL", "P",
        "Model Word Surprisal over ClozeProb",
        "ΔLL (Model Word Surp − ClozeProb)",
        "10a_bko_clozeprob_word.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    # ---- 10b: ClozeProb vs model region surprisal --------------------------
    # Left = clozeregionprob, Right = model region surprisal.

    plot_single_group_bars(build_clozeprob_df(bkr, True), "DLL", "P",
        "Model Region Surprisal over ClozeProb",
        "ΔLL (Model Region Surp − ClozeProb)",
        "10b_rep_clozeprob_region.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    plot_single_group_bars(build_clozeprob_df(bko, True), "DLL", "P",
        "Model Region Surprisal over ClozeProb",
        "ΔLL (Model Region Surp − ClozeProb)",
        "10b_bko_clozeprob_region.png", figsize=(10, 7),
        models_to_plot=no_cloze, show_legend=False)

    print("\n" + "="*70)
    print("✓ ALL 14 PLOTS COMPLETE")
    print("="*70)
