# PSAP (Point Subtraction Aggression Paradigm)

MATLAB source for the PSAP task used in the concurrent-validity analysis
(Bravura vs. PSAP). Adapted by João Rodrigues from the Millisecond Inquisit
implementation, based on Cherek et al. (1997).

## Contents

- **`Run_Experiment.m`** — the Psychtoolbox task. A competitive game against a
  fictitious opponent: press **A** (×100) to earn a point, **B** (×10) to
  steal a point and start a protection interval, or **C** (×10) to start a
  protection interval. Provocations subtract points whenever no protection is
  active. Writes `<SubjectID>.mat` containing the `InGameVars` struct, whose
  `Events` cell array (rows: `name; time; points`) logs every selection.
- **`PSAPResponses.m`** — feature extraction. Scores the `.mat` files into
  proactive (`pA/pB/pC`) and reactive (`rA/rB/rC`) button proportions, split
  at 120 s. Run `PSAPResponses('data/raw/psap', 'PSAPResps.xlsx')`.
- **`Functions/`** — Psychtoolbox draw/input helpers used by the task.

## Requirements

- **`Run_Experiment.m`** + `Functions/` require [Psychtoolbox-3](http://psychtoolbox.org/)
  (`Screen`, `KbWait`/`KbCheck`, `GetSecs`, `WaitSecs`, `IOPort`,
  `DrawFormattedText`). The default serial trigger port uses Psychtoolbox's
  `IOPort`; the optional parallel-port branch additionally needs `io32`/`io64`.
- **`PSAPResponses.m`** needs only base MATLAB (`load`, `regexp`,
  `struct2table`, `writetable`). Writing `.xlsx` output relies on MATLAB's
  built-in spreadsheet support.

## Phase split

The proactive/reactive boundary is built into the task: the first provocation
is scheduled at `120 s + rand_sample(6, 45)` (see `selection()` in
`Run_Experiment.m`), so no provocation can occur in the first two minutes.
Responses before 120 s are therefore proactive (free operant); responses after
are reactive (post-provocation).

## Provenance of the analysed data

`Run_Experiment.m` → per-participant `data/raw/psap/P###.mat` → the
proactive/reactive proportions from `PSAPResponses.m` → the
`pA/pB/pC/rA/rB/rC` columns in `data/raw/additional.xlsx` →
`notebooks/fig3.ipynb` (`data/processed/psap_ilr.csv`) → `src/brms/psap.R`.

The raw `.mat` files (extracted from `data/raw/PSAP.zip` into `data/raw/psap/`)
are not tracked in git.
