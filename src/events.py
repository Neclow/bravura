"""Parse trial events from BioPac MatlabEvents files.

Ports the canonical MATLAB extraction logic from
``dev/physio/aggression_choices_biopac.m`` so that decisions reconstructed
from the raw event stream match the summary files (NF_TransposedAggroAll,
aggroPerformance.xlsx, vba_input.xlsx).

Rules:

* Two opponent blocks, each anchored on a ``T101`` (Start Exp) marker.
* Within a block, every trial is anchored on a ``T107`` (Choice Open).
* The participant's decision is the FIRST ``T108`` (chose shock) or
  ``T109`` (chose ring) within the next 1-2 events after T107.
* If no T108/T109 follows T107 (next event is T128/T129/T105/T106) then
  the participant did not press a button -- recorded as choice=0
  (no shock).
* The first 15 T107 events in each block are used; any extras are
  silently dropped (matches the MATLAB ``for j=1:15`` loop).

For sessions with more than two T101 markers (restarted blocks), this
module tries to find a consecutive T101 pair whose intervening events
contain exactly 15 T107 markers; if none exists, the first 15 T107s
within the chosen window are still extracted.
"""

from pathlib import Path

import pandas as pd
from scipy.io import loadmat


MARKER_EXP_START = 101  # Start of opponent block
MARKER_CUBE_START = 105  # StartCube (pre-trial countdown)
MARKER_PIPE_VISIBLE = 106  # Trial start (pipe/wire visible)
MARKER_CHOICE_OPEN = 107  # Participant can choose
MARKER_CHOSE_SHOCK = 108
MARKER_CHOSE_RING = 109
MARKER_WIN = 110
MARKER_LOSE = 111
MARKER_RING_FEEDBACK = 129
MARKER_SHOCK_FEEDBACK = 128


def _select_block_bounds(nids):
    """Return ((opp1_start, opp1_end), (opp2_start, opp2_end)) for both blocks.

    A "valid" T101-anchored block is one containing exactly 15 T107
    (Choice Open) events before the next T101 or end-of-file. With exactly
    two T101 events the bounds are unambiguous. With session restarts
    (more than two T101 events), prefer the first and last T101 whose
    windows are valid; intermediate aborted attempts are skipped.

    Returns ``None`` if no two valid T101 windows can be found.
    """
    t101 = [i for i, n in enumerate(nids) if n == MARKER_EXP_START]
    if len(t101) < 2:
        return None

    # Window for each T101: from its index to the next T101 (or EOF).
    boundaries = t101 + [len(nids)]
    windows = [(t101[k], boundaries[k + 1]) for k in range(len(t101))]
    t107_counts = [
        sum(1 for i, n in enumerate(nids)
            if n == MARKER_CHOICE_OPEN and start <= i < end)
        for start, end in windows
    ]

    valid = [k for k, c in enumerate(t107_counts) if c == 15]

    if len(valid) >= 2:
        opp1_idx, opp2_idx = valid[0], valid[-1]
    elif len(t101) == 2:
        # Abnormal block(s): fall back to using the two T101s anyway.
        opp1_idx, opp2_idx = 0, 1
    else:
        # >2 T101s but no two valid windows — best-effort: first and last.
        opp1_idx, opp2_idx = 0, len(t101) - 1

    return windows[opp1_idx], windows[opp2_idx]


def _extract_block(events, n_trials=15):
    """Apply the canonical T107-anchored choice rule to one block.

    Returns a list of n_trials dicts with keys: trial_in_block,
    choice ('shock', 'ring', or 'none'), shock (0/1).
    Slices events as ``trial(idx:end)`` and advances ``idx`` after each
    T107, exactly as in aggression_choices_biopac.m.
    """
    nids = [int(e['nid'].flatten()[0]) for e in events]
    out = []
    idx = 0
    for j in range(n_trials):
        ons = None
        for k in range(idx, len(nids)):
            if nids[k] == MARKER_CHOICE_OPEN:
                ons = k
                break
        if ons is None:
            out.append({'trial_in_block': j + 1, 'choice': 'none', 'shock': 0})
            continue

        next1 = nids[ons + 1] if ons + 1 < len(nids) else None
        next2 = nids[ons + 2] if ons + 2 < len(nids) else None

        if next1 == MARKER_CHOSE_SHOCK or next2 == MARKER_CHOSE_SHOCK:
            out.append({'trial_in_block': j + 1, 'choice': 'shock', 'shock': 1})
        elif next1 == MARKER_CHOSE_RING or next2 == MARKER_CHOSE_RING:
            out.append({'trial_in_block': j + 1, 'choice': 'ring', 'shock': 0})
        else:
            out.append({'trial_in_block': j + 1, 'choice': 'none', 'shock': 0})

        idx = ons + 1

    return out


# Manual patches for subjects whose NF/aggroPerformance entries differ from
# the canonical script output. These appear to have been hand-curated
# (the canonical MATLAB script prints debug warnings on these abnormal
# cases — the `disp(['j = ...'])` in the abnormal branch — so whoever ran
# it likely fixed the per-trial sequences manually before writing NF).
# Format: {subject_id: {trial_index_0based: shock_value}}
_CANONICAL_PATCHES = {
    # P102: block 2 has 16 T107s (one duplicate). Canonical drops the dud
    # at idx 786 (trial 27, ME=0) and shifts; net effect: trial 27 becomes
    # shock=1, trial 28 becomes shock=0 (rather than 0, 1).
    'P102': {26: 1, 27: 0},
    # BF326: block 2 has 16 T107s (duplicate around idx 695/697). Drop the
    # dud and shift; trials 23,24,26,28 of overall sequence move by one.
    'BF326': {22: 1, 23: 0, 25: 1, 27: 0},
    # BF060: block 1 has only 14 T107s instead of 15. The recording cut off
    # mid-trial-14 (T107->T108 but no T105/T106/T110/T111) and skipped
    # straight to block 2's T101 after an unusual 181-second gap (normal:
    # ~53s). Trial 15 of block 1 is completely absent from the event
    # stream; its true value is unknown. Mark as NaN. Block 2's trial 30
    # is unambiguously a shock in the raw events (T107->T108->T105->T106
    # ->T111->T104) and is left as recorded — this differs from NF, which
    # set it to 0, but matches the actual data.
    'BF060': {14: float('nan')},
}


def parse_events(mat_path, apply_patches=True):
    """Extract per-trial decisions from a single MatlabEvents .mat file.

    Parameters
    ----------
    mat_path : str or Path
        Path to a MatlabEvents .mat file (e.g., P035.mat or BF069.mat).
    apply_patches : bool, default True
        If True, apply manual patches from ``_CANONICAL_PATCHES`` so the
        output matches the curated NF/aggroPerformance.xlsx for subjects
        whose raw events have anomalies the MATLAB script doesn't handle.

    Returns
    -------
    list[dict] or None
        List of 30 trial dicts (15 per opponent), each with keys
        ``trial``, ``opponent``, ``choice`` (``'shock'``, ``'ring'``,
        or ``'none'``), ``shock`` (0/1). Returns ``None`` if fewer
        than two T101 markers are present.
    """
    mat = loadmat(mat_path)
    events = mat["event"][0]
    nids = [int(e["nid"].flatten()[0]) for e in events]

    bounds = _select_block_bounds(nids)
    if bounds is None:
        return None
    (b1_start, b1_end), (b2_start, b2_end) = bounds

    block1 = events[b1_start:b1_end]
    block2 = events[b2_start:b2_end]

    trials_b1 = _extract_block(block1)
    trials_b2 = _extract_block(block2)

    trials = []
    for t in trials_b1:
        trials.append({
            'trial': t['trial_in_block'],
            'opponent': 1,
            'choice': t['choice'],
            'shock': t['shock'],
        })
    for t in trials_b2:
        trials.append({
            'trial': 15 + t['trial_in_block'],
            'opponent': 2,
            'choice': t['choice'],
            'shock': t['shock'],
        })

    if apply_patches:
        import math
        subj = Path(mat_path).stem
        for trial_idx, shock_val in _CANONICAL_PATCHES.get(subj, {}).items():
            trials[trial_idx]['shock'] = shock_val
            if isinstance(shock_val, float) and math.isnan(shock_val):
                trials[trial_idx]['choice'] = 'unknown'
            else:
                trials[trial_idx]['choice'] = 'shock' if shock_val == 1 else 'ring'

    return trials


def load_all_trial_events(events_dir="data/raw/MatlabEvents"):
    """Parse trial events for all participants in a MatlabEvents directory.

    Parameters
    ----------
    events_dir : str or Path
        Directory containing per-participant .mat files.

    Returns
    -------
    pd.DataFrame
        Columns: subject, cohort, trial, opponent, choice, shock.
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
                errors.append((subj, "fewer than 2 T101 markers"))
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
