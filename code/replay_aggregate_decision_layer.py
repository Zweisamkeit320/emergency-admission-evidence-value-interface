"""Replay the public aggregate decision layer without participant-level panel records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


CRITERIA = tuple(f"D{i}" for i in range(1, 11))
ALTERNATIVES = tuple(f"A{i}" for i in range(1, 6))
PANELS = ("A", "B")
ANALYSES = (
    "primary_ddcdw",
    "first_encounter_explanation_ddcdw",
    "logistic_grouped_pfi",
    "grouped_treeshap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", required=True, type=Path)
    parser.add_argument("--panel-value-weights", required=True, type=Path)
    parser.add_argument("--aggregate-utility-cells", required=True, type=Path)
    parser.add_argument("--patient-evidence-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_semantic_core(path: Path):
    sys.path.insert(0, str(path.resolve()))
    try:
        from paired_semantic_core import (  # type: ignore
            combine_criterion_weights,
            dense_ranks_ascending,
            interval_spotis,
        )
    finally:
        sys.path.pop(0)
    return combine_criterion_weights, dense_ranks_ascending, interval_spotis


def normalized_vector(frame: pd.DataFrame, value: str) -> np.ndarray:
    series = frame.set_index("criterion_id")[value].astype(float).reindex(CRITERIA)
    if series.isna().any():
        raise ValueError(f"Incomplete criterion vector for {value}")
    values = series.to_numpy(dtype=float)
    if not np.isclose(values.sum(), 1.0, atol=1e-12):
        raise ValueError(f"Criterion vector does not close: {value} sum={values.sum()}")
    return values


def utility_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = frame.set_index(["alternative_id", "criterion_id"])
    expected = pd.MultiIndex.from_product(
        [ALTERNATIVES, CRITERIA], names=["alternative_id", "criterion_id"]
    )
    if len(frame) != 50 or not key.index.is_unique or not expected.equals(key.reindex(expected).index):
        raise ValueError("Each panel must contain 50 unique direction-by-domain utility cells")
    ordered = key.reindex(expected)
    if ordered[["u0", "u_lower", "u_upper"]].isna().any().any():
        raise ValueError("Aggregate utility input is incomplete")
    shape = (len(ALTERNATIVES), len(CRITERIA))
    return tuple(ordered[column].to_numpy(dtype=float).reshape(shape) for column in ("u0", "u_lower", "u_upper"))


def ranking_and_pairs(
    analysis: str,
    panel: str,
    u0: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    weights: np.ndarray,
    dense_ranks_ascending,
    interval_spotis,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = interval_spotis(u0, u_lower, u_upper, weights)
    ranks = dense_ranks_ascending(result["distance_nominal"])
    scenario = f"{panel}_{analysis}"
    rank_rows = []
    pair_rows = []
    for i, alternative in enumerate(ALTERNATIVES):
        rank_rows.append(
            {
                "analysis": analysis,
                "panel": panel,
                "scenario": scenario,
                "lambda_objective": 0.5,
                "alternative_id": alternative,
                "spotis_D0": result["distance_nominal"][i],
                "spotis_D_lower": result["distance_lower"][i],
                "spotis_D_upper": result["distance_upper"][i],
                "rank": ranks[i],
                "necessary_outdegree": int(result["necessary_preference"][i].sum()),
                "possible_outdegree": int(result["possible_preference"][i].sum()),
            }
        )
        for j, right in enumerate(ALTERNATIVES):
            if i == j:
                continue
            pair_rows.append(
                {
                    "analysis": analysis,
                    "panel": panel,
                    "scenario": scenario,
                    "lambda_objective": 0.5,
                    "left_alternative": alternative,
                    "right_alternative": right,
                    "left_nominally_preferred": bool(
                        result["distance_nominal"][i] < result["distance_nominal"][j] - 1e-12
                    ),
                    "left_necessarily_preferred": bool(result["necessary_preference"][i, j]),
                    "left_possibly_preferred": bool(result["possible_preference"][i, j]),
                }
            )
    return pd.DataFrame(rank_rows), pd.DataFrame(pair_rows)


def ranking_string(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["rank", "spotis_D0", "alternative_id"])
    return ">".join(ordered["alternative_id"].astype(str))


def common_necessary(pair_tables: dict[str, pd.DataFrame]) -> list[str]:
    output = []
    for left in ALTERNATIVES:
        for right in ALTERNATIVES:
            if left == right:
                continue
            if all(
                bool(
                    pair_tables[panel].loc[
                        (pair_tables[panel]["left_alternative"] == left)
                        & (pair_tables[panel]["right_alternative"] == right),
                        "left_necessarily_preferred",
                    ].iloc[0]
                )
                for panel in PANELS
            ):
                output.append(f"{left}>{right}")
    return sorted(output)


def exact_lambda_analysis(
    u0: np.ndarray, objective: np.ndarray, subjective: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d_objective = (1.0 - u0) @ (objective / objective.sum())
    d_subjective = (1.0 - u0) @ (subjective / subjective.sum())
    crossings = []
    breakpoints = {0.0, 1.0}
    for i in range(len(ALTERNATIVES)):
        for j in range(i + 1, len(ALTERNATIVES)):
            delta_sub = d_subjective[i] - d_subjective[j]
            delta_slope = (d_objective[i] - d_objective[j]) - delta_sub
            if abs(delta_slope) <= 1e-15:
                continue
            root = -delta_sub / delta_slope
            if -1e-12 <= root <= 1.0 + 1e-12:
                root = float(np.clip(root, 0.0, 1.0))
                breakpoints.add(root)
                crossings.append(
                    {
                        "alternative_1": ALTERNATIVES[i],
                        "alternative_2": ALTERNATIVES[j],
                        "lambda_crossing": root,
                        "distance_at_crossing": (1.0 - root) * d_subjective[i]
                        + root * d_objective[i],
                    }
                )
    segments = []
    points = sorted(breakpoints)
    for left, right in zip(points[:-1], points[1:]):
        midpoint = (left + right) / 2.0
        distances = (1.0 - midpoint) * d_subjective + midpoint * d_objective
        order = np.argsort(distances, kind="mergesort")
        segments.append(
            {
                "lambda_left": left,
                "lambda_right": right,
                "representative_lambda": midpoint,
                "ranking": ">".join(ALTERNATIVES[index] for index in order),
                "top_alternative": ALTERNATIVES[int(order[0])],
            }
        )
    return pd.DataFrame(crossings), pd.DataFrame(segments)


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> int:
    return len(set(np.argsort(-a)[:k]).intersection(np.argsort(-b)[:k]))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combine_weights, dense_ranks, interval_spotis = load_semantic_core(args.core_dir)

    panel_weights = pd.read_csv(args.panel_value_weights)
    utilities = pd.read_csv(args.aggregate_utility_cells)
    patient_weights = pd.read_csv(args.patient_evidence_weights)
    observed_analyses = set(patient_weights["analysis"].drop_duplicates())
    if observed_analyses != set(ANALYSES):
        raise ValueError(f"Unexpected patient-evidence specifications: {sorted(observed_analyses)}")

    patient_vectors = {
        name: normalized_vector(patient_weights[patient_weights["analysis"] == name], "normalized_weight")
        for name in ANALYSES
    }
    primary = patient_vectors["primary_ddcdw"]
    concordance_rows = []
    for name, vector in patient_vectors.items():
        concordance_rows.append(
            {
                "analysis": name,
                "spearman_vs_primary_ddcdw": float(spearmanr(primary, vector).statistic),
                "kendall_vs_primary_ddcdw": float(kendalltau(primary, vector).statistic),
                "top5_overlap_vs_primary_ddcdw": top_k_overlap(primary, vector),
            }
        )

    matrices = {
        panel: utility_matrices(utilities[utilities["panel"] == panel]) for panel in PANELS
    }
    subjective_vectors = {
        panel: normalized_vector(panel_weights[panel_weights["panel"] == panel], "panel_value_weight")
        for panel in PANELS
    }

    ranking_frames = []
    pair_frames = []
    combined_rows = []
    summary_rows = []
    first_crossings = []
    first_segments = []
    for name, objective in patient_vectors.items():
        panel_pairs = {}
        panel_ranks = {}
        for panel in PANELS:
            subjective = subjective_vectors[panel]
            combined = combine_weights(objective, subjective, 0.5)
            rank, pair = ranking_and_pairs(
                name,
                panel,
                *matrices[panel],
                combined,
                dense_ranks,
                interval_spotis,
            )
            ranking_frames.append(rank)
            pair_frames.append(pair)
            panel_pairs[panel] = pair
            panel_ranks[panel] = ranking_string(rank)
            for criterion, objective_value, subjective_value, combined_value in zip(
                CRITERIA, objective, subjective, combined, strict=True
            ):
                combined_rows.append(
                    {
                        "analysis": name,
                        "panel": panel,
                        "criterion_id": criterion,
                        "patient_evidence_weight": objective_value,
                        "panel_value_weight": subjective_value,
                        "combined_weight_lambda_0_5": combined_value,
                    }
                )
            if name == "first_encounter_explanation_ddcdw":
                crossings, segments = exact_lambda_analysis(matrices[panel][0], objective, subjective)
                crossings.insert(0, "panel", panel)
                segments.insert(0, "panel", panel)
                first_crossings.append(crossings)
                first_segments.append(segments)

        common = common_necessary(panel_pairs)
        summary_rows.append(
            {
                "analysis": name,
                "panel_A_ranking": panel_ranks["A"],
                "panel_B_ranking": panel_ranks["B"],
                "shared_necessary_count": len(common),
                "shared_necessary_relations": ";".join(common),
            }
        )

    summary = pd.DataFrame(summary_rows)
    expected_primary = summary.loc[summary["analysis"] == "primary_ddcdw"].iloc[0]
    if (
        expected_primary["panel_A_ranking"] != "A5>A1>A4>A2>A3"
        or expected_primary["panel_B_ranking"] != "A5>A4>A1>A2>A3"
        or int(expected_primary["shared_necessary_count"]) != 9
    ):
        raise AssertionError("Primary rankings or shared certificate count changed")

    outputs = {
        "alternative_attribution_sensitivity_summary.csv": summary,
        "attribution_weight_concordance.csv": pd.DataFrame(concordance_rows),
        "decision_rankings_all_weight_specs.csv": pd.concat(ranking_frames, ignore_index=True),
        "decision_pairwise_all_weight_specs.csv": pd.concat(pair_frames, ignore_index=True),
        "combined_weights_all_specs.csv": pd.DataFrame(combined_rows),
        "utility_cells_100.csv": utilities,
        "first_encounter_explanation_lambda_crossings.csv": pd.concat(first_crossings, ignore_index=True),
        "first_encounter_explanation_lambda_segments.csv": pd.concat(first_segments, ignore_index=True),
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            args.output_dir / filename,
            index=False,
            encoding="utf-8-sig",
            float_format="%.17g",
        )

    status = {
        "status": "PASS",
        "scope": "aggregate decision-layer replay for four patient-evidence specifications",
        "summary": summary.to_dict(orient="records"),
        "patient_model_refit": False,
        "participant_level_panel_aggregation_recomputed": False,
        "aggregate_decision_layer_recomputed": True,
        "public_panel_data_level": "aggregate only",
    }
    (args.output_dir / "full_precision_sensitivity_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
