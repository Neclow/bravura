import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src._config import DEFAULT_BRMS_DIR, DEFAULT_PHYSIO_DIR_A

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG_DIR = f"{DEFAULT_IMG_DIR}/fig4"
os.makedirs(FIG_DIR, exist_ok=True)

DV_ORDER = ["HR", "HRV_RC1", "HRV_RC2", "HRV_RC3"]
DV_TITLES = ["HR", "HRV RC1\n(power)", "HRV RC2\n(vagal)", "HRV RC3\n(entropy)"]

FOCUS_BLOCK = "1.1"
LOADINGS_PATH = f"{DEFAULT_PHYSIO_DIR_A}/hrv_pca_loadings.csv"
RC_LABELS = ["RC1 (power)", "RC2 (vagal)", "RC3 (entropy)"]


def short_contrast(c):
    """Shorten contrast label for axis readability."""
    return c.replace("(Non-aggressive) - ", "NA–").replace("Proactive - ", "P–")


def load_data():
    """Load cardiac pairwise contrasts for block 1.1.

    Returns
    -------
    block11 : DataFrame
        Pairwise contrasts filtered to the focus block.
    """
    contrasts = pd.read_csv(f"{DEFAULT_BRMS_DIR}/physio_cardiac/pairwise_contrasts.csv")
    return contrasts[contrasts["block"] == float(FOCUS_BLOCK)]


def plot_cardiac_contrasts(block11):
    """Plot block 1.1 pairwise contrasts for HR and HRV RC1-3 (Supp. Fig. 9)."""
    with plt.style.context(DEFAULT_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(9, 2.5), sharey=True)

        for ax, dv, title in zip(axes, DV_ORDER, DV_TITLES):
            sub = block11[block11["DV"] == dv].copy()
            sub["short"] = sub["contrast"].apply(short_contrast)
            y = np.arange(len(sub))

            colors = [
                "red" if (lo > 0 or hi < 0) else "0.5"
                for lo, hi in zip(sub["Q2.5"], sub["Q97.5"])
            ]

            ax.barh(
                y,
                sub["estimate"],
                xerr=[
                    sub["estimate"] - sub["Q2.5"],
                    sub["Q97.5"] - sub["estimate"],
                ],
                color=colors,
                capsize=3,
                height=0.5,
                alpha=0.7,
            )
            ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
            ax.set_yticks(y)
            ax.set_yticklabels(sub["short"])
            ax.set_title(title, fontsize=10)

        axes[0].set_xlabel("Δ (bpm)")
        for ax in axes[1:]:
            ax.set_xlabel("Δ (score)")
            ax.set_xlim(-2, 2)

        fig.suptitle("Block 1.1 pairwise contrasts (95% CrI)", y=0.96, fontsize=11)
        fig.tight_layout()
        stem = f"{FIG_DIR}/figS17a_cardiac_contrasts"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG_DIR}/tableS11_cardiac_contrasts.md"
    block11.round(3).to_markdown(table_path, index=False)
    print(f"Saved {table_path}")


def load_loadings():
    """Load varimax-rotated PCA loadings for HRV features.

    Returns
    -------
    df_loadings : DataFrame
        Features (index) x RC components (columns).
    """
    return pd.read_csv(LOADINGS_PATH, index_col="feature")


def plot_hrv_loadings(df_loadings):
    """Plot varimax-rotated PCA loadings for HRV RC1-3 (Supp. Fig. 17b)."""
    with plt.style.context(DEFAULT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7, 3.5), sharex=True)

        for rc, ax in zip(RC_LABELS, axes):
            sorted_vals = df_loadings[rc].sort_values()
            colors = ["#c44e52" if v < 0 else "#4c72b0" for v in sorted_vals]
            y = np.arange(len(sorted_vals))
            ax.barh(y, sorted_vals.values, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(sorted_vals.index)
            ax.set_xlabel("Loading")
            ax.set_title(rc)
            ax.set_axisbelow(True)
            ax.grid(alpha=0.3, axis="x")
            ax.set_xticks(np.arange(-1, 1.1, 0.5))

        fig.tight_layout()
        stem = f"{FIG_DIR}/figS17b_hrv_pca_loadings"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


if __name__ == "__main__":
    block11 = load_data()
    plot_cardiac_contrasts(block11)

    df_loadings = load_loadings()
    plot_hrv_loadings(df_loadings)
