#!/usr/bin/env python3
"""
Generate the SIA/SIC main GAM figures for BKO and BKR.

Inputs are the *_gam_smooth_data.csv files exported from the GAM runs.
The duplicated "(1)" CSVs are ignored; this script uses the clean filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "plots"

MODEL_ORDER = [
    "Cloze",
    "GPT-2",
    "GPT-Neo",
    "GPT-NeoX",
    "GPT-J",
    "GPT-2XL",
    "OLMO-2",
    "LLaMA-2",
]

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

DATASET_LABELS = {
    "bko": "BKO",
    "bkr": "BKR",
}

PALETTE = "viridis"
LINE_WIDTH = 1.8
CI_ALPHA = 0.18


@dataclass(frozen=True)
class PlotSpec:
    dataset: str
    analysis: str
    csv_name: str
    title_label: str
    y_label: str


SPECS = [
    PlotSpec(
        dataset="bko",
        analysis="SIA_log",
        csv_name="bko_SIA_log_gam_smooth_data.csv",
        title_label="SIA: log(SUMMED_3RT)",
        y_label="Partial Effect on log(SUMMED_3RT)",
    ),
    PlotSpec(
        dataset="bkr",
        analysis="SIA_log",
        csv_name="bkr_SIA_log_gam_smooth_data.csv",
        title_label="SIA: log(SUMMED_3RT)",
        y_label="Partial Effect on log(SUMMED_3RT)",
    ),
    PlotSpec(
        dataset="bko",
        analysis="SIC_HC",
        csv_name="bko_SIC_HC_gam_smooth_data.csv",
        title_label="SIC HC Items: SUMMED_3RT",
        y_label="Partial Effect on SUMMED_3RT (ms)",
    ),
    PlotSpec(
        dataset="bkr",
        analysis="SIC_HC",
        csv_name="bkr_SIC_HC_gam_smooth_data.csv",
        title_label="SIC HC Items: SUMMED_3RT",
        y_label="Partial Effect on SUMMED_3RT (ms)",
    ),
    PlotSpec(
        dataset="bko",
        analysis="SIC_LC",
        csv_name="bko_SIC_LC_gam_smooth_data.csv",
        title_label="SIC LC Items: SUMMED_3RT",
        y_label="Partial Effect on SUMMED_3RT (ms)",
    ),
    PlotSpec(
        dataset="bkr",
        analysis="SIC_LC",
        csv_name="bkr_SIC_LC_gam_smooth_data.csv",
        title_label="SIC LC Items: SUMMED_3RT",
        y_label="Partial Effect on SUMMED_3RT (ms)",
    ),
]


def load_smooth_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"fit", "se_lower", "se_upper"}.issubset(df.columns):
        df = df.rename(columns={"fit": "y", "se_lower": "lower", "se_upper": "upper"})

    required = {"x", "y", "lower", "upper", "model", "measure"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    return df


def model_colors(models: list[str]) -> dict[str, tuple[float, float, float]]:
    palette = sns.color_palette(PALETTE, len(MODEL_ORDER))
    base_colors = dict(zip(MODEL_ORDER, palette))
    return {model: base_colors[model] for model in models}


def robust_ylim(
    df: pd.DataFrame,
    *,
    q_low: float = 0.03,
    q_high: float = 0.97,
    pad_fraction: float = 0.12,
    include_zero: bool = True,
) -> tuple[float, float]:
    """Use central CI quantiles so a few edge-band extremes do not hide the smooth."""
    values = pd.concat(
        [df["lower"], df["upper"], df["y"]],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        return (-1.0, 1.0)

    low = float(values.quantile(q_low))
    high = float(values.quantile(q_high))

    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)

    if np.isclose(low, high):
        pad = max(abs(low) * 0.1, 1e-3)
    else:
        pad = (high - low) * pad_fraction

    return low - pad, high + pad


def style_axis(ax, *, y_lim: tuple[float, float]) -> None:
    ax.axhline(0, color="0.35", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_ylim(y_lim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", which="major", labelsize=10.5, length=5, width=1)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_curve(ax, df: pd.DataFrame, color, *, y_lim: tuple[float, float]) -> None:
    data = df.sort_values("x", kind="mergesort")
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    lower = data["lower"].to_numpy()
    upper = data["upper"].to_numpy()

    ax.fill_between(x, lower, upper, alpha=CI_ALPHA, color=color, linewidth=0)
    ax.plot(x, y, color=color, linewidth=LINE_WIDTH)
    style_axis(ax, y_lim=y_lim)


def plot_main_gam(spec: PlotSpec, df: pd.DataFrame) -> Path:
    models = [model for model in MODEL_ORDER if model in set(df["model"])]
    colors = model_colors(models)
    n_models = len(models)

    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(18, 5.3),
        sharey="row",
        gridspec_kw={"wspace": 0.20, "hspace": 0.42},
    )
    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    row_defs = [
        ("Probability", "Probability"),
        ("Surprisal", "Surprisal"),
    ]
    row_ylims = {
        measure: robust_ylim(df[df["measure"] == measure])
        for measure, _ in row_defs
    }

    for row_idx, (measure, x_label) in enumerate(row_defs):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            subset = df[(df["model"] == model) & (df["measure"] == measure)]

            if subset.empty:
                ax.set_visible(False)
                continue

            plot_curve(ax, subset, colors[model], y_lim=row_ylims[measure])

            if row_idx == 0:
                ax.set_title(
                    MODEL_DISPLAY_NAMES.get(model, model),
                    fontsize=14,
                    fontweight="bold",
                    pad=5,
                )

            if col_idx == 0:
                ax.set_xlabel(x_label, fontsize=13)
                ax.set_ylabel(spec.y_label, fontsize=12)
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if measure == "Probability":
                ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.1f}"))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))

    fig.text(0.01, 0.73, "Probability", va="center", rotation="vertical", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.28, "Surprisal", va="center", rotation="vertical", fontsize=14, fontweight="bold")
    fig.suptitle(
        f"{DATASET_LABELS[spec.dataset]} {spec.title_label}: Main GAM",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    output = OUTPUT_DIR / f"{spec.dataset}_{spec.analysis}_main_gam.png"
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.13, top=0.83, wspace=0.20, hspace=0.42)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_region_surprisal_gam(spec: PlotSpec, df: pd.DataFrame) -> Path:
    df = df[df["measure"] == "Surprisal"].copy()
    models = [model for model in MODEL_ORDER if model in set(df["model"])]
    colors = model_colors(models)
    n_models = len(models)
    y_lim = robust_ylim(df)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(24, 4.8),
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    if n_models == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        subset = df[df["model"] == model]

        if subset.empty:
            ax.set_visible(False)
            continue

        plot_curve(ax, subset, colors[model], y_lim=y_lim)
        ax.set_title(
            MODEL_DISPLAY_NAMES.get(model, model),
            fontsize=17,
            fontweight="bold",
            pad=10,
        )
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))

        if i == 0:
            ax.set_xlabel("Region Surprisal", fontsize=15)
            ax.set_ylabel(spec.y_label, fontsize=15)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    fig.suptitle(
        f"{DATASET_LABELS[spec.dataset]} {spec.title_label}: Region Surprisal GAM",
        fontsize=19,
        fontweight="bold",
        y=1.04,
    )

    output = OUTPUT_DIR / f"{spec.dataset}_{spec.analysis}_region_surprisal_gam.png"
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.18, top=0.75, wspace=0.12)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=0.95)
    OUTPUT_DIR.mkdir(exist_ok=True)

    outputs: list[Path] = []
    for spec in SPECS:
        csv_path = ROOT / spec.csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing expected GAM smooth CSV: {csv_path}")

        df = load_smooth_data(csv_path)
        outputs.append(plot_main_gam(spec, df))
        outputs.append(plot_region_surprisal_gam(spec, df))

    print("Saved plots:")
    for output in outputs:
        print(f"  {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
