# Reproducibility scope

## Direct aggregate public replay

The public inputs are sufficient to recompute combined criterion weights,
interval distances, rankings, pairwise relations, and cross-panel
necessary-preference certificates from frozen aggregate panel-value weights
and 100 aggregate utility cells. They also support propagation of four declared
patient-evidence specifications through the same decision layer.

Participant-level weak orders, linguistic ratings, and SNA edges are not part
of the public release. Consequently, the public replay does not re-estimate
Fuzzy RANCOM weights or reconstruct PLTS/T2NN/Rough utilities from individual
panel responses. The released aggregate objects are sufficient for the
reported interval-SPOTIS rankings and preference certificates.

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
- participant-level panel records, direct identifiers, participant names,
  signatures, or role-to-person maps;
- the exploratory Route-B stress-test archive.

The public patient-model code therefore supports a credential-gated rerun by a
credentialed researcher who supplies the authorized engineered feature table
and fitted or newly trained model object. Reconstruction of the engineered
feature table from raw MIMIC relations is not claimed in this release.

## Interpretation boundaries

- DD-CDW is grouped marginal permutation importance under Brier loss.
- The patient-evidence layer comes from 2011–2019 MIMIC data; the 2025–2026
  Dalian panels supplied value judgments, not patient data or an external
  validation cohort.
- DD-CDW is model-dependence evidence for demonstrating the cross-source
  interface, not a local demand estimate for either panel hospital. Local use
  requires re-estimation and validation with a target-hospital patient cohort.
- The logistic grouped PFI and grouped TreeSHAP analyses are sensitivity
  specifications, not replacements for the primary analysis.
- Identification bounds are not confidence intervals or probabilities.
- Necessary preferences are sufficient relations within the declared
  analytical set, not cross-hospital guarantees.
- A1–A5 are combinable preimplementation screening directions, not mutually
  exclusive procurement packages or implemented interventions.
- A5 is an enabling information-and-preparedness layer and is reported
  separately from the A1–A4 direct-strategy order.
- A3's position is conditional on patient-domain coverage criteria; the replay
  does not estimate waiting-time, boarding, capacity, staffing, cost, safety,
  or throughput effects.
