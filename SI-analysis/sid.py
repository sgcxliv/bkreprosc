import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bkr21_spr.csv")
plot_df = df[["cloze", "log_cloze", "SUM_3RT_trimmed"]].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SI D: Raw SUMMED_3RT across cloze probability and cloze surprisal",
             fontsize=14, y=1.02)

# Plot A: probability scale
sns.kdeplot(
    x=plot_df["cloze"],
    y=plot_df["SUM_3RT_trimmed"],
    fill=True, thresh=0, levels=100, cmap="Blues", bw_adjust=0.5,
    ax=axes[0]
)
axes[0].set_xlabel("Cloze probability", fontsize=12)
axes[0].set_ylabel("SUMMED_3RT (ms)", fontsize=12)
axes[0].set_title("A. Cloze probability", fontsize=13, fontweight="bold")

# Plot B: surprisal scale 
sns.kdeplot(
    x=-plot_df["log_cloze"],
    y=plot_df["SUM_3RT_trimmed"],
    fill=True, thresh=0, levels=100, cmap="Greens", bw_adjust=0.5,
    ax=axes[1]
)
axes[1].set_xlabel("Cloze surprisal", fontsize=12)
axes[1].set_ylabel("SUMMED_3RT (ms)", fontsize=12)
axes[1].set_title("B. Cloze surprisal", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("bkr_SI_D_raw_RT_density.pdf", bbox_inches="tight", dpi=300)
plt.savefig("bkr_SI_D_raw_RT_density.png", bbox_inches="tight", dpi=200)
plt.show()

df = pd.read_csv("bko21_spr.csv")
plot_df = df[["cloze", "log_cloze", "SUM_3RT_trimmed"]].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SI D: Raw SUMMED_3RT across cloze probability and cloze surprisal",
             fontsize=14, y=1.02)

# Plot A: probability scale
sns.kdeplot(
    x=plot_df["cloze"],
    y=plot_df["SUM_3RT_trimmed"],
    fill=True, thresh=0, levels=100, cmap="Blues", bw_adjust=0.5,
    ax=axes[0]
)
axes[0].set_xlabel("Cloze probability", fontsize=12)
axes[0].set_ylabel("SUMMED_3RT (ms)", fontsize=12)
axes[0].set_title("A. Cloze probability", fontsize=13, fontweight="bold")

# Plot B: surprisal scale 
sns.kdeplot(
    x=-plot_df["log_cloze"],
    y=plot_df["SUM_3RT_trimmed"],
    fill=True, thresh=0, levels=100, cmap="Greens", bw_adjust=0.5,
    ax=axes[1]
)
axes[1].set_xlabel("Cloze surprisal", fontsize=12)
axes[1].set_ylabel("SUMMED_3RT (ms)", fontsize=12)
axes[1].set_title("B. Cloze surprisal", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("bko_SI_D_raw_RT_density.pdf", bbox_inches="tight", dpi=300)
plt.savefig("bko_SI_D_raw_RT_density.png", bbox_inches="tight", dpi=200)
plt.show()