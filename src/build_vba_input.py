"""Rebuild data/cohort_b/vba_input.xlsx from canonical decisions.

The legacy vba_input.xlsx was derived through an unknown transposition
step and contains wrong decisions for at least BF060, BF280 and BF464
(verified against NF_TransposedAggroAll.csv and aggroPerformance.xlsx).

This script preserves the 9 task-design factor rows (Trial, Win, WinTm1,
WinTm2, WinSum, Shocked, ShockedTm1, ShockedTm2, ShockedSum -- same
across subjects) and overwrites every subject's 30 decisions with the
canonical sequence from tmp/NF_TransposedAggroAll.csv.

Run with:

    pixi run python -m src.build_vba_input
"""

from pathlib import Path

import pandas as pd

LEGACY_PATH = Path("data/cohort_b/vba_input.xlsx")
NF_PATH = Path("tmp/NF_TransposedAggroAll.csv")
OUT_PATH = Path("data/cohort_b/vba_input.xlsx")

N_FACTOR_ROWS = 10   # 1 header + 9 factor rows
N_TRIALS = 30


def main():
    legacy = pd.read_excel(LEGACY_PATH, header=None)
    if legacy.shape != (52, 31):
        raise ValueError(
            f"Unexpected legacy shape {legacy.shape}, expected (52, 31)"
        )

    nf = pd.read_csv(NF_PATH, header=None)
    nf.set_index(0, inplace=True)
    if nf.shape != (42, N_TRIALS):
        raise ValueError(f"Unexpected NF shape {nf.shape}, expected (42, 30)")

    factor_rows = legacy.iloc[:N_FACTOR_ROWS].copy()

    subject_rows = []
    for excel_row in range(N_FACTOR_ROWS, 52):
        sid = str(legacy.iat[excel_row, 0])
        if sid not in nf.index:
            raise ValueError(f"Subject {sid} (row {excel_row}) not in NF")
        subject_rows.append([sid] + nf.loc[sid].tolist())

    out = pd.concat(
        [factor_rows, pd.DataFrame(subject_rows, columns=legacy.columns)],
        ignore_index=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUT_PATH, header=False, index=False)
    print(f"Wrote {OUT_PATH} -- {len(subject_rows)} subject rows")


if __name__ == "__main__":
    main()
