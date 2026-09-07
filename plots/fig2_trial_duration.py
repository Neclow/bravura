import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.io import loadmat
from statannotations.Annotator import Annotator

from src._config import DEFAULT_BRMS_DIR, DEFAULT_DATA_DIR, DEFAULT_SHARED_DIR

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

TRIAL_EVENTS_FNAME = "trial_events.csv"
OUTLIERS_FNAME = "outliers.mat"
POSTERIOR_DIR = f"{DEFAULT_BRMS_DIR}/trial_duration"
FIG2_DIR = f"{DEFAULT_IMG_DIR}/fig2"
os.makedirs(FIG2_DIR, exist_ok=True)

DECISION_ORDER = ["Enlarge", "Shock"]
DECISION_MAP = {
    "ring": "Loop enlarged",
    "shock": "Shock given",
    float("nan"): "Nothing",
}
DECISION_LABELS = ["Loop enlarged", "Shock given"]

TEST_SUBJECTS = ["P089"]


def _parse_contrast(contrast):
    parts = contrast.split(" - ")
    return parts[0].strip("() "), parts[1].strip("() ")


def plot_trial_duration(cohort):
    """Plot trial duration vs. decision with posterior predictions.

    Parameters
    ----------
    cohort : str
        Cohort label, e.g. "A" or "B".
    """
    suffix = f"_{cohort.lower()}"

    trials = pd.read_csv(f"{DEFAULT_SHARED_DIR}/{TRIAL_EVENTS_FNAME}")
    trials_clean = trials[trials["duration"].between(0, 30)]
    trials_cohort = trials_clean.query(f"cohort == '{cohort}'")
    trials_cohort = trials_cohort[~trials_cohort["subject"].isin(TEST_SUBJECTS)]

    outliers = loadmat(
        f"{DEFAULT_DATA_DIR}/cohort_{cohort.lower()}/{OUTLIERS_FNAME}", squeeze_me=True
    )["outliers"]
    trials_cohort = trials_cohort[~trials_cohort["subject"].isin(outliers)]
    trials_cohort["decision"] = trials_cohort["choice"].map(DECISION_MAP)
    counts = trials_cohort.groupby("decision").size()

    duration_posterior = pd.read_csv(f"{POSTERIOR_DIR}/posterior_epred.csv")
    bf_duration = pd.read_csv(f"{POSTERIOR_DIR}/bayes_factors.csv")

    dur_post = duration_posterior.query(f"cohort == '{cohort}' & decision != 'Nothing'")

    stem = f"{FIG2_DIR}/fig2d_trial_duration{suffix}"

    with plt.style.context(DEFAULT_STYLE):
        plot_dict = {
            "data": dur_post,
            "x": "decision",
            "y": "duration",
        }

        fig, ax = plt.subplots(figsize=(2.5, 3))
        sns.barplot(
            **plot_dict,
            order=DECISION_ORDER,
            palette="Greys_r",
            capsize=0.05,
            errorbar=("pi", 95),
            ax=ax,
        )
        labels = [f"{d}\n(N = {counts[d]})" for d in DECISION_LABELS]
        ax.set_xticklabels(labels)
        ax.set_ylabel("Trial duration (s)")
        ax.set_xlabel("")
        ax.axhline(4, color="darkred", linestyle="--", alpha=0.5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

        sig = bf_duration[(bf_duration["cohort"] == cohort) & bf_duration["excl_zero"]]
        pairs = []
        for _, row in sig.iterrows():
            c1, c2 = _parse_contrast(row["contrast"])
            if c1 in DECISION_ORDER and c2 in DECISION_ORDER:
                pairs.append((c1, c2))

        if pairs:
            annotator = Annotator(
                ax,
                pairs=pairs,
                **plot_dict,
                order=DECISION_ORDER,
                verbose=False,
            )
            annotator.configure(loc="outside", verbose=False)
            annotator.set_custom_annotations(["*"] * len(pairs))
            annotator.annotate()

        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG2_DIR}/tableS3_trial_duration{suffix}.md"
    bf_duration.set_index("contrast").round(3).to_markdown(table_path)
    print(f"Saved {table_path}")

    with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
        bf_row = bf_duration.query(
            f"cohort == '{cohort}' & contrast == 'Enlarge - Shock'"
        ).iloc[0]
        f.write(
            f"Difference: {-bf_row['estimate']:.2f} s "
            f"[{-bf_row['Q97.5']:.2f}, {-bf_row['Q2.5']:.2f}]\n"
        )

        wide = dur_post.pivot_table(
            index=".draw", columns="decision", values="duration"
        )
        pct = (wide["Shock"] - wide["Enlarge"]) / wide["Enlarge"] * 100
        f.write(
            f"Relative: {pct.mean():.1f}% "
            f"[{pct.quantile(0.025):.1f}%, {pct.quantile(0.975):.1f}%]\n"
        )
    print(f"Saved {stem}_stats.txt")


if __name__ == "__main__":
    plot_trial_duration("A")
