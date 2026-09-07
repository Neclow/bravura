import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src._config import DEFAULT_BRMS_DIR

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG_DIR = f"{DEFAULT_IMG_DIR}/fig4"
os.makedirs(FIG_DIR, exist_ok=True)

DV_ORDER = ["HR", "HRV_RC1", "HRV_RC2", "HRV_RC3"]
DV_TITLES = ["HR", "HRV RC1\n(power)", "HRV RC2\n(vagal)", "HRV RC3\n(entropy)"]

FOCUS_BLOCK = "1.1"


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
    contrasts = pd.read_csv(
        f"{DEFAULT_BRMS_DIR}/physio_cardiac/pairwise_contrasts.csv"
    )
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

        fig.suptitle(
            "Block 1.1 pairwise contrasts (95% CrI)", y=0.96, fontsize=11
        )
        fig.tight_layout()
        stem = f"{FIG_DIR}/figS8_cardiac_contrasts"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG_DIR}/tableS11_cardiac_contrasts.md"
    block11.round(3).to_markdown(table_path, index=False)
    print(f"Saved {table_path}")


if __name__ == "__main__":
    block11 = load_data()
    plot_cardiac_contrasts(block11)
