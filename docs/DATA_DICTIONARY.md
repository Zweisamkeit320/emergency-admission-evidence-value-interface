# Public data dictionary

## Aggregate panel inputs

- `panel_value_weights.csv`: 2 panels × 10 domains of normalized panel-level
  criterion weights used in the primary equal-participant analysis.
- `aggregate_utility_cells.csv`: 2 panels × 5 candidate directions × 10 domains
  of aggregate nominal utility and interval endpoints, with aggregate
  diagnostic columns retained for traceability.

No participant codes, individual weak orders, individual linguistic ratings,
or individual SNA edges are included. The public replay therefore begins at
the frozen aggregate panel objects.

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
