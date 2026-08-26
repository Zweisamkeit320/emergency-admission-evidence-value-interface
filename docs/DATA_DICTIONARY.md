# Public data dictionary

## Panel inputs

- `panel_A_criterion_weak_orders.csv` and
  `panel_B_criterion_weak_orders.csv`: 100 deidentified
  participant-by-domain final weak-order records per panel.
- `panel_A_linguistic_ratings.csv` and
  `panel_B_linguistic_ratings.csv`: 500 deidentified
  participant-by-direction-by-domain linguistic ratings per panel.
- `panel_A_sna_edges.csv` and `panel_B_sna_edges.csv`: 400 computational SNA
  cells per panel, including structural diagonal zeros. SNA does not alter the
  primary equal-participant weighting.

Participant codes (`DM1`–`DM10`) are analysis labels and do not identify a
person. No participant role roster is supplied or inferred.

## Patient-evidence inputs

- `grouped_model_permutation_weights.csv`: full-precision XGBoost DD-CDW and
  logistic grouped permutation weights.
- `grouped_treeshap_weights.csv`: full-precision grouped TreeSHAP weights.
- `first_encounter_explanation_ddcdw.csv`: first-encounter DD-CDW computed in
  the explanation/weight-estimation partition.
- `patient_evidence_weight_specifications.csv`: harmonized long-form weights
  used by the decision-layer sensitivity replay.
- `feature_domain_dictionary.csv`: mapping of the 38 raw features to D1–D10.
- `patient_model_specification.json`: model, preprocessing, calibration, split,
  and software specifications without patient rows or fitted objects.

## Reference results

- `utility_cells_100.csv`: 2 panels × 5 directions × 10 domains.
- `decision_rankings_all_weight_specs.csv`: nominal distance and interval
  endpoints under each patient-evidence specification.
- `decision_pairwise_all_weight_specs.csv`: all directed necessary/possible
  relation classifications.
- `alternative_attribution_sensitivity_summary.csv`: the compact sensitivity
  result used in the supplementary manuscript table.
- `attribution_weight_concordance.csv`: Spearman, Kendall, and top-five overlap
  against primary DD-CDW.

