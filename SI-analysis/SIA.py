import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def add_panel_label(ax, letter):
    ax.text(
        0.02, 0.98, letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_rt_density(csv_path, out_prefix):
    plot_df = (
        pd.read_csv(csv_path)[["clozeprob", "cloze", "SUM_3RT_trimmed"]]
        .dropna()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(
        x=plot_df["clozeprob"],
        y=plot_df["SUM_3RT_trimmed"],
        fill=True, thresh=0, levels=100, cmap="Blues", bw_adjust=0.5,
        ax=axes[0],
    )
    axes[0].set_xlabel("Cloze probability", fontsize=12)
    axes[0].set_ylabel("Reading Time (ms)", fontsize=12)
    add_panel_label(axes[0], "A")

    sns.kdeplot(
        x=plot_df["cloze"],
        y=plot_df["SUM_3RT_trimmed"],
        fill=True, thresh=0, levels=100, cmap="Greens", bw_adjust=0.5,
        ax=axes[1],
    )
    axes[1].set_xlabel("Cloze surprisal", fontsize=12)
    axes[1].set_ylabel("Reading Time (ms)", fontsize=12)
    add_panel_label(axes[1], "B")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


plot_rt_density("bkr21_spr.csv", "bkr_SI_A_raw_RT_density")
