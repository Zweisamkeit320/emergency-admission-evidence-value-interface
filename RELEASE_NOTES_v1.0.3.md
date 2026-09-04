# v1.0.3 release notes

This release aligns the public repository with an aggregate-only hospital-panel
data route.

- Removed participant-level weak orders, linguistic ratings, and SNA edges.
- Removed participant-indexed leave-one-out output tables; the aggregate
  robustness summary reported in the supplement is unchanged.
- Added frozen aggregate panel-value weights and 100 aggregate utility cells as
  the public decision-layer inputs.
- Added a default aggregate replay that recomputes combined weights,
  interval-SPOTIS distances, rankings, and necessary/possible preferences for
  four patient-evidence specifications.
- Retained the credential-gated patient-model scripts and aggregate model
  outputs, without MIMIC patient rows, engineered patient-level tables, fitted
  model objects, or patient-level permutation deltas.

The scientific rankings, nine primary shared necessary relations, and reported
sensitivity results are unchanged. This release narrows the public panel-data
scope; it does not constitute a new scientific analysis.
