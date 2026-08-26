# Reproducibility scope

## Direct public replay

The public inputs are sufficient to recompute panel-value weights, paired
linguistic utilities, interval distances, rankings, pairwise relations, and
cross-panel necessary-preference certificates. They also support propagation
of four declared patient-evidence specifications through the same decision
layer.

## Credential-gated patient-layer replay

`code/run_patient_model_weight_pipeline.py` begins from a 38-feature table with
patient-disjoint split labels. `code/run_first_encounter_explanation.py` selects
the first chronological eligible encounter for each patient in the `tuning`
partition and re-estimates DD-CDW without loading the internal-evaluation
partition.

The public repository does not contain:

- MIMIC-IV or MIMIC-IV-ED source rows;
- derived patient-level feature tables or patient-level permutation deltas;
- fitted model objects;
- direct identifiers, participant names, signatures, or role-to-person maps;
- the exploratory Route-B stress-test archive.

The public patient-model code therefore supports a credential-gated rerun by a
credentialed researcher who supplies the authorized engineered feature table
and fitted or newly trained model object. Reconstruction of the engineered
feature table from raw MIMIC relations is not claimed in this release.

## Interpretation boundaries

- DD-CDW is grouped marginal permutation importance under Brier loss.
- The logistic grouped PFI and grouped TreeSHAP analyses are sensitivity
  specifications, not replacements for the primary analysis.
- Identification bounds are not confidence intervals or probabilities.
- Necessary preferences are sufficient relations within the declared
  analytical set, not cross-hospital guarantees.
- A1–A5 are combinable preimplementation screening directions, not mutually
  exclusive procurement packages or implemented interventions.
