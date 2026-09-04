# v1.0.3 aggregate decision-layer reproduction package

This repository accompanies the manuscript **“From emergency admission
prediction to candidate resource-direction screening: an auditable
evidence–value interface.”**

Stable public repository: https://github.com/Zweisamkeit320/emergency-admission-evidence-value-interface

The public package supports an aggregate decision-layer replay from panel-level
criterion weights, 100 aggregate direction-by-domain utility cells, and four
aggregate patient-evidence weight specifications. It does not redistribute
participant-level hospital-panel records, MIMIC-IV/MIMIC-IV-ED patient rows,
derived patient-level feature tables, or fitted model objects.

## What can be replayed directly

- Combination of released aggregate panel-value and patient-evidence weights at
  the declared `lambda=0.50` setting.
- Interval-SPOTIS propagation from all 100 released aggregate
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

## Cross-source transportability boundary

The patient-evidence layer uses 2011–2019 MIMIC-IV/MIMIC-IV-ED data from the
Beth Israel Deaconess Medical Center data ecosystem. The hospital-value layer
uses two separately convened, non-overlapping panels in Dalian, China, convened
in 2025–2026. The panel hospitals supplied no patient data, and the MIMIC model
was not externally validated in either hospital. DD-CDW therefore demonstrates
how domain-level model-dependence evidence can enter the interface; it is not
an estimate of current patient demand in the panel hospitals. Operational use
requires the patient-evidence layer to be re-estimated and validated in the
target hospital's local cohort.

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

- `code/`: aggregate decision-layer and credential-gated patient-layer code.
- `data/`: aggregate panel-value weights, aggregate utility cells, aggregate
  patient-evidence weights, model specifications, and the feature-to-domain
  dictionary.
- `results_reference/`: full-precision reference outputs reported or used in
  sensitivity checks.
- `tests/`: automated replay checks.

## Data access

MIMIC-IV and MIMIC-IV-ED are credentialed PhysioNet resources. Obtain access
from PhysioNet and comply with its data use agreement. Do not place patient
rows, derived patient-level tables, or fitted models from MIMIC in an ordinary
public repository.

## Status

This release reproduces the decision layer from frozen aggregate panel objects
and supplies aggregate evidence for the patient layer. It does not re-estimate
Fuzzy RANCOM weights or PLTS/T2NN/Rough utilities from participant-level panel
records. It supports candidate resource-direction screening, not external
validation, final resource allocation, or implementation-effect evaluation.
