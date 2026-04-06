"""Parse trial events from BioPac MatlabEvents files."""

from pathlib import Path

import pandas as pd
from scipy.io import loadmat


# Event marker codes from BioPac (see data/raw/Markers legend.xlsx)
MARKER_EXP_START = 101  # Start of experiment
MARKER_PIPE_VISIBLE = 106  # Trial start (pipe/wire becomes visible)
MARKER_CHOICE_OPEN = 107  # Participant can choose
MARKER_CHOSE_SHOCK = 108  # Chose to shock
MARKER_CHOSE_RING = 109  # Chose to enlarge ring/sphere
MARKER_WIN = 110  # Trial won
MARKER_LOSE = 111  # Trial lost


def parse_events(mat_path):
    """Extract per-trial data from a single MatlabEvents .mat file.

    Parameters
    ----------
    mat_path : str or Path
        Path to a MatlabEvents .mat file (e.g., P035.mat or BF069.mat)

    Returns
    -------
    list[dict] or None
        List of trial dicts with keys: trial, opponent, duration, won, choice.
        Returns None if no experiment start marker (101) is found.
    """
    mat = loadmat(mat_path)
    events = mat["event"][0]
    fields = events[0].dtype.names

    # P* files use 'time', BF* files use 'seconds'
    time_field = "seconds" if "seconds" in fields else "time"

    nids = [int(e["nid"].flatten()[0]) for e in events]
    secs = [float(e[time_field].flatten()[0]) for e in events]

    # Find experiment start
    exp_starts = [s for s, n in zip(secs, nids) if n == MARKER_EXP_START]
    if not exp_starts:
        return None
    exp_start = exp_starts[0]

    # Collect events after experiment start
    trial_starts = [
        (s, n) for s, n in zip(secs, nids) if n == MARKER_PIPE_VISIBLE and s > exp_start
    ]
    trial_ends = [
        (s, n)
        for s, n in zip(secs, nids)
        if n in (MARKER_WIN, MARKER_LOSE) and s > exp_start
    ]
    choices = [
        (s, n)
        for s, n in zip(secs, nids)
        if n in (MARKER_CHOSE_SHOCK, MARKER_CHOSE_RING) and s > exp_start
    ]

    # Match each trial start to its next end and preceding choice
    trials = []
    end_idx = 0
    for i, (start_sec, _) in enumerate(trial_starts):
        while end_idx < len(trial_ends) and trial_ends[end_idx][0] <= start_sec:
            end_idx += 1

        chose = None
        for cs, cn in choices:
            if cs < start_sec and (i == 0 or cs > trial_starts[i - 1][0]):
                chose = "shock" if cn == MARKER_CHOSE_SHOCK else "ring"

        if end_idx < len(trial_ends):
            end_sec, outcome = trial_ends[end_idx]
            trials.append(
                {
                    "trial": i + 1,
                    "opponent": 1 if (i + 1) <= 15 else 2,
                    "duration": end_sec - start_sec,
                    "won": outcome == MARKER_WIN,
                    "choice": chose,
                }
            )
            end_idx += 1

    return trials


def load_all_trial_events(events_dir="data/raw/MatlabEvents"):
    """Parse trial events for all participants in a MatlabEvents directory.

    Parameters
    ----------
    events_dir : str or Path
        Directory containing per-participant .mat files

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, cohort, trial, opponent, duration, won, choice
    """
    events_dir = Path(events_dir)
    all_trials = []
    errors = []

    for f in sorted(events_dir.glob("*.mat")):
        subj = f.stem
        cohort = "B" if subj.startswith("BF") else "A"
        try:
            trials = parse_events(f)
            if trials is None:
                errors.append((subj, "no marker 101"))
                continue
            for t in trials:
                t["subject"] = subj
                t["cohort"] = cohort
            all_trials.extend(trials)
        except Exception as e:
            errors.append((subj, str(e)))

    if errors:
        print(f"Warning: {len(errors)} participants could not be parsed:")
        for subj, err in errors:
            print(f"  {subj}: {err}")

    return pd.DataFrame(all_trials)
