"""Rebuild data/cohort_b_v{2,3}/vba_input.xlsx from canonical decisions.

The v1 vba_input.xlsx was derived through an unknown transposition step
and contains wrong decisions for at least BF060, BF280 and BF464
(verified against NF_TransposedAggroAll.csv and aggroPerformance.xlsx).

This script keeps the 9 factor rows from v1 (Trial, Win, WinTm1, WinTm2,
WinSum, Shocked, ShockedTm1, ShockedTm2, ShockedSum -- task-design rows
that are the same across subjects) and overwrites every subject's
30 decisions with the canonical sequence from NF_TransposedAggroAll.csv.

Two output variants:

* v3 (default, "salvage rule"): include all 42 subjects
* v2 ("Option A", Joao's OG rule): drop the 5 restart subjects (BF042,
  BF264, BF280, BF330, BF464; they have 3 T101 events each, which
  aggression_choices_biopac.m would `continue` over)

Run with:

    pixi run python -m src.build_vba_input         # writes both v2 and v3
    pixi run python -m src.build_vba_input v3      # only v3
    pixi run python -m src.build_vba_input v2      # only v2
"""

import sys
from pathlib import Path

import pandas as pd

V1_PATH = Path("data/cohort_b/vba_input.xlsx")
NF_PATH = Path("tmp/NF_TransposedAggroAll.csv")

N_FACTOR_ROWS = 10   # 1 header + 9 factor rows
N_TRIALS = 30
RESTART_SUBJECTS = ['BF042', 'BF264', 'BF280', 'BF330', 'BF464']


def build(out_path, drop_subjects=None):
    drop = set(drop_subjects or [])
    v1 = pd.read_excel(V1_PATH, header=None)
    if v1.shape != (52, 31):
        raise ValueError(f"Unexpected v1 shape {v1.shape}, expected (52, 31)")

    nf = pd.read_csv(NF_PATH, header=None)
    nf.set_index(0, inplace=True)
    if nf.shape != (42, N_TRIALS):
        raise ValueError(f"Unexpected NF shape {nf.shape}, expected (42, 30)")

    factor_rows = v1.iloc[:N_FACTOR_ROWS].copy()

    subject_rows = []
    for excel_row in range(N_FACTOR_ROWS, 52):
        sid = str(v1.iat[excel_row, 0])
        if sid in drop:
            continue
        if sid not in nf.index:
            raise ValueError(f"Subject {sid} (row {excel_row}) not in NF")
        new_row = [sid] + nf.loc[sid].tolist()
        subject_rows.append(new_row)

    out = pd.concat(
        [factor_rows, pd.DataFrame(subject_rows, columns=v1.columns)],
        ignore_index=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(out_path, header=False, index=False)
    print(f"Wrote {out_path} -- {len(subject_rows)} subject rows"
          + (f" (dropped: {sorted(drop)})" if drop else ""))


def main(targets):
    if 'v3' in targets:
        build(Path("data/cohort_b_v3/vba_input.xlsx"), drop_subjects=None)
    if 'v2' in targets:
        build(Path("data/cohort_b_v2/vba_input.xlsx"),
              drop_subjects=RESTART_SUBJECTS)


if __name__ == "__main__":
    args = sys.argv[1:] or ['v2', 'v3']
    main(args)
