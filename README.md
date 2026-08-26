# v5.9.6 decision-layer reproduction package

This repository accompanies the manuscript **“From emergency admission
prediction to hospital priorities: an auditable evidence–value interface for
pre-implementation screening of resource directions.”**

Stable public repository: https://github.com/Zweisamkeit320/emergency-admission-evidence-value-interface

The public package supports a direct replay of the decision layer from
deidentified panel inputs and aggregate patient-evidence weights. It does not
redistribute MIMIC-IV/MIMIC-IV-ED patient rows, derived patient-level feature
tables, or fitted model objects.

## What can be replayed directly

- Fuzzy RANCOM panel-value weights from 100 weak-order records per panel.
- Paired PLTS/T2NN representation, Rough-region propagation, and all 100
  panel-by-direction-by-domain utility cells.
- Interval-SPOTIS nominal rankings and necessary/possible preferences.
- The nine cross-panel primary DD-CDW necessary-preference certificates.
- First-encounter DD-CDW estimated in the explanation/weight-estimation
  partition.
- Full-precision downstream sensitivity for logistic grouped permutation
  importance and grouped TreeSHAP.

## Reproducibility boundary

`DD-CDW` is a study-specific name for grouped **marginal** permutation
importance measured by the increase in Brier loss. All raw input columns in a
domain are permuted jointly and passed through the frozen preprocessing,
calibration, and prediction pipeline. It is not conditional resampling from
the feature distribution, a causal effect, an objective measure of patient
need, or a hospital-value weight.

The patient-model scripts are included for credentialed users, but an
end-to-end patient-layer rerun requires authorized MIMIC-IV/MIMIC-IV-ED data
and the corresponding engineered feature table. The public package starts the
patient-model pipeline from that feature table; it does not claim to reconstruct
the table from raw MIMIC relations.

See `docs/REPRODUCIBILITY_SCOPE.md` and `docs/DATA_DICTIONARY.md` for details.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -B run_reproduction.py
.venv/Scripts/python -B -m pytest -q
```

On macOS/Linux, replace `.venv/Scripts/python` with `.venv/bin/python`.

The replay writes `results_rerun/`. A successful run must recover:

| Patient-evidence specification | Panel A | Panel B | Shared necessary relations |
|---|---|---|---:|
| Primary DD-CDW | A5>A1>A4>A2>A3 | A5>A4>A1>A2>A3 | 9 |
| Explanation-partition first encounter | A1>A5>A2>A4>A3 | A5>A1>A4>A2>A3 | 8 |
| Logistic grouped PFI | A5>A1>A4>A2>A3 | A5>A4>A1>A2>A3 | 9 |
| Grouped TreeSHAP | A5>A1>A2>A4>A3 | A5>A1>A4>A2>A3 | 7 |

## Directory map

- `code/`: decision-layer and conditional patient-layer code.
- `data/`: deidentified panel inputs, aggregate patient-evidence weights,
  model specification, and the feature-to-domain dictionary.
- `results_reference/`: full-precision reference outputs reported or used in
  sensitivity checks.
- `tests/`: automated replay checks.

## Data access

MIMIC-IV and MIMIC-IV-ED are credentialed PhysioNet resources. Obtain access
from PhysioNet and comply with its data use agreement. Do not place patient
rows, derived patient-level tables, or fitted models from MIMIC in an ordinary
public repository.

## Status

This release reproduces the decision layer and supplies aggregate evidence for
the patient layer. It is not an external validation or an implementation-effect
evaluation.
