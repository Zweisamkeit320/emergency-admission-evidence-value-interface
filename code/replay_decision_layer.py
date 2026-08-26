"""Replay the decision layer with declared and alternative patient-evidence weights."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


CRITERIA = tuple(f"D{i}" for i in range(1, 11))
ALTERNATIVES = tuple(f"A{i}" for i in range(1, 6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", required=True, type=Path)
    parser.add_argument("--panel-a-importance", required=True, type=Path)
    parser.add_argument("--panel-a-ratings", required=True, type=Path)
    parser.add_argument("--panel-b-importance", required=True, type=Path)
    parser.add_argument("--panel-b-ratings", required=True, type=Path)
    parser.add_argument("--grouped-weights", required=True, type=Path)
    parser.add_argument("--group-shap-weights", required=True, type=Path)
    parser.add_argument("--first-encounter-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_core(path: Path):
    sys.path.insert(0, str(path.resolve()))
    try:
        analysis = importlib.import_module("analysis_pipeline")
        semantic = importlib.import_module("paired_semantic_core")
    finally:
        sys.path.pop(0)
    return analysis, semantic


def vector(frame: pd.DataFrame, key: str, value: str) -> np.ndarray:
    series = frame.set_index(key)[value].astype(float).reindex(CRITERIA)
    if series.isna().any():
        raise ValueError(f"Incomplete criterion vector in {value}")
    values = series.to_numpy(dtype=float)
    if not np.isclose(values.sum(), 1.0, atol=1e-12):
        raise ValueError(f"Criterion vector does not close: {value} sum={values.sum()}")
    return values


def criterion_from_domain(value: str) -> str:
    return value.split("_", 1)[0]


def load_panel(args: argparse.Namespace, panel: str, analysis):
    if panel == "A":
        importance = pd.read_csv(args.panel_a_importance)
        ratings = pd.read_csv(args.panel_a_ratings)
        return analysis.validate_importance(importance), analysis.validate_performance(ratings)

    raw_i = pd.read_csv(args.panel_b_importance)
    raw_r = pd.read_csv(args.panel_b_ratings)
    importance = pd.DataFrame(
        {
            "site_id": "A",
            "expert_id": raw_i["expert_id"],
            "criterion_id": raw_i["criterion_id"],
            "rank": raw_i["rank_value"],
        }
    )
    ratings = pd.DataFrame(
        {
            "site_id": "A",
            "alternative_id": raw_r["alternative"],
            "expert_id": raw_r["expert_id"],
            "criterion_id": raw_r["criterion_id"],
            "term": raw_r["term"],
        }
    )
    return analysis.validate_importance(importance), analysis.validate_performance(ratings)


def ranking_string(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["rank", "spotis_D0", "alternative_id"])
    return ">".join(ordered["alternative_id"].astype(str))


def common_necessary(pair_tables: dict[str, pd.DataFrame]) -> list[str]:
    output: list[str] = []
    for left in ALTERNATIVES:
        for right in ALTERNATIVES:
            if left == right:
                continue
            status = []
            for panel in ("A", "B"):
                table = pair_tables[panel]
                row = table[
                    (table["left_alternative"] == left)
                    & (table["right_alternative"] == right)
                ]
                if len(row) != 1:
                    raise AssertionError(f"Missing pair {panel}:{left}>{right}")
                status.append(bool(row.iloc[0]["left_necessarily_preferred"]))
            if all(status):
                output.append(f"{left}>{right}")
    return sorted(output)


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> int:
    return len(set(np.argsort(-a)[:k]).intersection(np.argsort(-b)[:k]))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis, semantic = load_core(args.core_dir)
    config = semantic.AggregationConfig()

    grouped = pd.read_csv(args.grouped_weights)
    grouped["criterion_id"] = grouped["domain"].map(criterion_from_domain)
    dd = vector(
        grouped[grouped["candidate"] == "xgb_medium"],
        "criterion_id",
        "normalized_weight",
    )
    logistic = vector(
        grouped[grouped["candidate"] == "logistic_c1"],
        "criterion_id",
        "normalized_weight",
    )

    shap = pd.read_csv(args.group_shap_weights)
    shap["criterion_id"] = shap["domain"].map(criterion_from_domain)
    tree_shap = vector(shap, "criterion_id", "normalized_group_shap_weight")

    first = pd.read_csv(args.first_encounter_weights)
    first_dd = vector(first, "criterion_id", "normalized_weight")

    objective_vectors = {
        "primary_ddcdw": dd,
        "first_encounter_explanation_ddcdw": first_dd,
        "logistic_grouped_pfi": logistic,
        "grouped_treeshap": tree_shap,
    }

    concordance_rows = []
    for name, values in objective_vectors.items():
        concordance_rows.append(
            {
                "analysis": name,
                "spearman_vs_primary_ddcdw": float(spearmanr(dd, values).statistic),
                "kendall_vs_primary_ddcdw": float(kendalltau(dd, values).statistic),
                "top5_overlap_vs_primary_ddcdw": top_k_overlap(dd, values),
            }
        )

    panel_objects: dict[str, dict[str, object]] = {}
    utility_rows = []
    for panel in ("A", "B"):
        importance, ratings = load_panel(args, panel, analysis)
        individual, _mac, _tfn, _extensions = analysis.fuzzy_rancom(importance)
        participant_weights = np.full(10, 0.1)
        subjective = analysis.group_subjective(individual, participant_weights)
        tensor = analysis.rating_tensor(ratings)
        u0, u_lower, u_upper, cells, _regions, _memberships = analysis.compute_cells(
            tensor, participant_weights, config, detailed=True
        )
        cells.insert(0, "panel", panel)
        utility_rows.append(cells)
        panel_objects[panel] = {
            "u0": u0,
            "u_lower": u_lower,
            "u_upper": u_upper,
            "subjective": subjective,
        }

    rankings = []
    pairs = []
    combined_weights = []
    summary_rows = []
    first_crossings = []
    first_segments = []
    for name, objective in objective_vectors.items():
        pair_tables: dict[str, pd.DataFrame] = {}
        rank_strings: dict[str, str] = {}
        for panel in ("A", "B"):
            item = panel_objects[panel]
            subjective = np.asarray(item["subjective"], dtype=float)
            weights = semantic.combine_criterion_weights(objective, subjective, 0.5)
            rank, pair = analysis.ranking_table(
                item["u0"],
                item["u_lower"],
                item["u_upper"],
                weights,
                f"{panel}_{name}",
                0.5,
            )
            rank.insert(0, "analysis", name)
            rank.insert(1, "panel", panel)
            pair.insert(0, "analysis", name)
            pair.insert(1, "panel", panel)
            rankings.append(rank)
            pairs.append(pair)
            pair_tables[panel] = pair
            rank_strings[panel] = ranking_string(rank)
            for criterion, objective_value, subjective_value, combined in zip(
                CRITERIA, objective, subjective, weights, strict=True
            ):
                combined_weights.append(
                    {
                        "analysis": name,
                        "panel": panel,
                        "criterion_id": criterion,
                        "patient_evidence_weight": objective_value,
                        "panel_value_weight": subjective_value,
                        "combined_weight_lambda_0_5": combined,
                    }
                )
            if name == "first_encounter_explanation_ddcdw":
                crossings, segments = analysis.exact_lambda_analysis(
                    item["u0"], objective, subjective
                )
                crossings.insert(0, "panel", panel)
                segments.insert(0, "panel", panel)
                first_crossings.append(crossings)
                first_segments.append(segments)

        common = common_necessary(pair_tables)
        summary_rows.append(
            {
                "analysis": name,
                "panel_A_ranking": rank_strings["A"],
                "panel_B_ranking": rank_strings["B"],
                "shared_necessary_count": len(common),
                "shared_necessary_relations": ";".join(common),
            }
        )

    summary = pd.DataFrame(summary_rows)
    primary = summary[summary["analysis"] == "primary_ddcdw"].iloc[0]
    if primary["panel_A_ranking"] != "A5>A1>A4>A2>A3":
        raise AssertionError("Primary Panel A ranking changed")
    if primary["panel_B_ranking"] != "A5>A4>A1>A2>A3":
        raise AssertionError("Primary Panel B ranking changed")
    if int(primary["shared_necessary_count"]) != 9:
        raise AssertionError("Primary shared certificate count changed")

    outputs = {
        "alternative_attribution_sensitivity_summary.csv": summary,
        "attribution_weight_concordance.csv": pd.DataFrame(concordance_rows),
        "decision_rankings_all_weight_specs.csv": pd.concat(rankings, ignore_index=True),
        "decision_pairwise_all_weight_specs.csv": pd.concat(pairs, ignore_index=True),
        "combined_weights_all_specs.csv": pd.DataFrame(combined_weights),
        "utility_cells_100.csv": pd.concat(utility_rows, ignore_index=True),
        "first_encounter_explanation_lambda_crossings.csv": pd.concat(
            first_crossings, ignore_index=True
        ),
        "first_encounter_explanation_lambda_segments.csv": pd.concat(
            first_segments, ignore_index=True
        ),
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
        "scope": "full-precision decision-layer propagation of four patient-evidence specifications",
        "summary": summary.to_dict(orient="records"),
        "patient_model_refit": False,
        "decision_layer_recomputed": True,
    }
    (args.output_dir / "full_precision_sensitivity_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
