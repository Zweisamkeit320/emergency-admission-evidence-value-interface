"""Auditable analysis helpers for the corrected A-hospital rerun."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from paired_semantic_core import (
    TERMS,
    T2NN_CODEBOOK,
    UTILITY_COORDS,
    AggregationConfig,
    aggregate_cell,
    combine_criterion_weights,
    dense_ranks_ascending,
    interval_spotis,
)


EXPERTS = tuple(f"DM{i}" for i in range(1, 11))
CRITERIA = tuple(f"D{i}" for i in range(1, 11))
ALTERNATIVES = tuple(f"A{i}" for i in range(1, 6))
LAYER_WEIGHTS = {
    "communication": 0.10,
    "collaboration": 0.20,
    "trust": 0.30,
    "influence": 0.40,
}
TFN_MAP = {
    1.0: (0.5, 1.0, 1.0),
    0.5: (0.0, 0.5, 1.0),
    0.0: (0.0, 0.0, 0.5),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_importance(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"site_id", "expert_id", "criterion_id", "rank"}
    if not required.issubset(frame.columns):
        raise ValueError(f"importance columns missing: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["rank"] = pd.to_numeric(result["rank"], errors="raise")
    expected = {("A", dm, d) for dm in EXPERTS for d in CRITERIA}
    observed = set(result[["site_id", "expert_id", "criterion_id"]].itertuples(index=False, name=None))
    if len(result) != 100 or result.duplicated(["site_id", "expert_id", "criterion_id"]).any():
        raise ValueError("importance must contain 100 unique site-DM-D keys")
    if observed != expected:
        raise ValueError("importance key set does not match A x DM1-DM10 x D1-D10")
    for dm, subset in result.groupby("expert_id"):
        values = sorted(float(x) for x in subset["rank"])
        position = 1
        for value, group in itertools.groupby(values):
            group_values = list(group)
            expected_midrank = position + (len(group_values) - 1) / 2.0
            if not math.isclose(value, expected_midrank, abs_tol=1e-12):
                raise ValueError(f"{dm} has invalid midrank {value}; expected {expected_midrank}")
            position += len(group_values)
        if not math.isclose(float(subset["rank"].sum()), 55.0, abs_tol=1e-12):
            raise ValueError(f"{dm} rank sum is not 55")
    return result.sort_values(["expert_id", "criterion_id"], key=lambda col: col.map(_natural_id)).reset_index(drop=True)


def validate_performance(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"site_id", "alternative_id", "expert_id", "criterion_id", "term"}
    if not required.issubset(frame.columns):
        raise ValueError(f"performance columns missing: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    expected = {
        ("A", alternative, dm, criterion)
        for alternative in ALTERNATIVES
        for dm in EXPERTS
        for criterion in CRITERIA
    }
    observed = set(
        result[["site_id", "alternative_id", "expert_id", "criterion_id"]].itertuples(
            index=False, name=None
        )
    )
    if len(result) != 500 or result.duplicated(
        ["site_id", "alternative_id", "expert_id", "criterion_id"]
    ).any():
        raise ValueError("performance must contain 500 unique site-A-DM-D keys")
    if observed != expected or not set(result["term"]).issubset(TERMS):
        raise ValueError("performance keys or terms violate the frozen contract")
    return result.sort_values(
        ["alternative_id", "criterion_id", "expert_id"], key=lambda col: col.map(_natural_id)
    ).reset_index(drop=True)


def validate_sna_edges(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"site_id", "layer", "source_expert_id", "target_expert_id", "edge_value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"SNA columns missing: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["edge_value"] = pd.to_numeric(result["edge_value"], errors="raise")
    expected = {
        ("A", layer, source, target)
        for layer in LAYER_WEIGHTS
        for source in EXPERTS
        for target in EXPERTS
    }
    observed = set(
        result[["site_id", "layer", "source_expert_id", "target_expert_id"]].itertuples(
            index=False, name=None
        )
    )
    if len(result) != 400 or result.duplicated(
        ["site_id", "layer", "source_expert_id", "target_expert_id"]
    ).any():
        raise ValueError("SNA must contain 400 unique site-layer-source-target keys")
    if observed != expected or not np.all(np.isin(result["edge_value"], [0.0, 1.0])):
        raise ValueError("SNA keys or binary edge values violate the frozen contract")
    diagonal = result["source_expert_id"] == result["target_expert_id"]
    if not np.allclose(result.loc[diagonal, "edge_value"], 0.0):
        raise ValueError("SNA diagonal must be zero")
    return result.reset_index(drop=True)


def _natural_id(value: object) -> int:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def mac_from_ranks(ranks: Sequence[float]) -> np.ndarray:
    x = np.asarray(ranks, dtype=float)
    if x.shape != (10,) or not np.all(np.isfinite(x)):
        raise ValueError("RANCOM requires ten finite ranks")
    matrix = np.where(x[:, None] < x[None, :], 1.0, 0.0)
    matrix[np.isclose(x[:, None], x[None, :], atol=1e-12)] = 0.5
    if not np.allclose(matrix + matrix.T, 1.0):
        raise AssertionError("RANCOM MAC reciprocity failed")
    return matrix


def tfn_from_mac(matrix: np.ndarray) -> np.ndarray:
    mac = np.asarray(matrix, dtype=float)
    if mac.shape != (10, 10) or not np.all(np.isin(mac, [0.0, 0.5, 1.0])):
        raise ValueError("MAC must be 10 x 10 over {0,0.5,1}")
    return np.asarray([[TFN_MAP[float(value)] for value in row] for row in mac], dtype=float)


def fuzzy_rancom(importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = validate_importance(importance)
    mac_rows: list[dict[str, object]] = []
    tfn_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    kemeny_rows: list[dict[str, object]] = []
    for dm in EXPERTS:
        subset = source[source["expert_id"] == dm].set_index("criterion_id").reindex(CRITERIA)
        ranks = subset["rank"].to_numpy(dtype=float)
        mac = mac_from_ranks(ranks)
        tfn = tfn_from_mac(mac)
        fuzzy_row_sum = tfn.sum(axis=1)
        centroid = fuzzy_row_sum.mean(axis=1)
        weights = centroid / centroid.sum()
        closed_form = (13.0 - ranks) / 75.0
        if not np.allclose(weights, closed_form, atol=1e-12):
            raise AssertionError("Fuzzy RANCOM weights disagree with the rank closed form")
        for i, criterion_i in enumerate(CRITERIA):
            weight_rows.append(
                {
                    "expert_id": dm,
                    "criterion_id": criterion_i,
                    "rank": ranks[i],
                    "fuzzy_row_sum_l": fuzzy_row_sum[i, 0],
                    "fuzzy_row_sum_m": fuzzy_row_sum[i, 1],
                    "fuzzy_row_sum_u": fuzzy_row_sum[i, 2],
                    "centroid": centroid[i],
                    "individual_fuzzy_rancom_weight": weights[i],
                    "closed_form_weight": closed_form[i],
                }
            )
            for j, criterion_j in enumerate(CRITERIA):
                mac_rows.append(
                    {
                        "expert_id": dm,
                        "row_criterion": criterion_i,
                        "column_criterion": criterion_j,
                        "mac_value": mac[i, j],
                    }
                )
                tfn_rows.append(
                    {
                        "expert_id": dm,
                        "row_criterion": criterion_i,
                        "column_criterion": criterion_j,
                        "mac_value": mac[i, j],
                        "tfn_l": tfn[i, j, 0],
                        "tfn_m": tfn[i, j, 1],
                        "tfn_u": tfn[i, j, 2],
                    }
                )
        tie_groups = [
            tuple(CRITERIA[index] for index in np.flatnonzero(np.isclose(ranks, value)))
            for value in sorted(set(ranks))
        ]
        extensions = [()]
        for group in tie_groups:
            extensions = [prefix + perm for prefix in extensions for perm in itertools.permutations(group)]
        for extension_id, order in enumerate(extensions, start=1):
            linearized_ties = sum(len(group) * (len(group) - 1) // 2 for group in tie_groups)
            kemeny_rows.append(
                {
                    "expert_id": dm,
                    "optimal_extension_id": extension_id,
                    "optimal_extension_count": len(extensions),
                    "order_most_to_least": ">".join(order),
                    "strict_pairwise_judgments_changed": 0,
                    "tie_relations_linearized": linearized_ties,
                    "disagreement_on_strict_pairs": 0,
                    "audit_label": "tie-preserving weak-order linear-extension enumeration; not general Kemeny repair",
                }
            )
    return (
        pd.DataFrame(weight_rows),
        pd.DataFrame(mac_rows),
        pd.DataFrame(tfn_rows),
        pd.DataFrame(kemeny_rows),
    )


def group_subjective(individual: pd.DataFrame, participant_weights: Sequence[float]) -> np.ndarray:
    q = np.asarray(participant_weights, dtype=float)
    if q.shape != (len(EXPERTS),) or np.any(q < 0.0) or q.sum() <= 0.0:
        raise ValueError("participant weights must align to DM1-DM10")
    q = q / q.sum()
    matrix = (
        individual.pivot(index="expert_id", columns="criterion_id", values="individual_fuzzy_rancom_weight")
        .reindex(index=EXPERTS, columns=CRITERIA)
        .to_numpy(dtype=float)
    )
    result = q @ matrix
    if not math.isclose(float(result.sum()), 1.0, abs_tol=1e-12):
        raise AssertionError("group subjective weights do not close")
    return result


def sna_matrices(edges: pd.DataFrame) -> dict[str, np.ndarray]:
    source = validate_sna_edges(edges)
    matrices: dict[str, np.ndarray] = {}
    for layer in LAYER_WEIGHTS:
        subset = source[source["layer"] == layer]
        matrix = (
            subset.pivot(index="source_expert_id", columns="target_expert_id", values="edge_value")
            .reindex(index=EXPERTS, columns=EXPERTS)
            .to_numpy(dtype=float)
        )
        matrices[layer] = matrix
    return matrices


def sna_centrality(
    matrices: Mapping[str, np.ndarray],
    omitted_layer: str | None = None,
    pagerank_damping: float = 0.85,
    pagerank_mix: float = 0.70,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    active = {name: matrix for name, matrix in matrices.items() if name != omitted_layer}
    if not active:
        raise ValueError("at least one SNA layer must remain")
    fused = sum(LAYER_WEIGHTS[name] * np.asarray(matrix, dtype=float) for name, matrix in active.items())
    graph = nx.from_numpy_array(fused, create_using=nx.DiGraph)
    pagerank_dict = nx.pagerank(
        graph, alpha=pagerank_damping, weight="weight", tol=1e-14, max_iter=10000
    )
    pagerank = np.asarray([pagerank_dict[i] for i in range(len(EXPERTS))], dtype=float)
    eigen_dict = nx.eigenvector_centrality_numpy(graph, weight="weight")
    eigen = np.asarray([abs(eigen_dict[i]) for i in range(len(EXPERTS))], dtype=float)
    eigen = eigen / eigen.sum()
    q = pagerank_mix * pagerank + (1.0 - pagerank_mix) * eigen
    q = q / q.sum()
    table = pd.DataFrame(
        {
            "expert_id": EXPERTS,
            "pagerank": pagerank,
            "eigenvector_normalized": eigen,
            "sna_weight": q,
            "omitted_layer": omitted_layer or "none",
        }
    )
    return q, table, fused


def rating_tensor(performance: pd.DataFrame) -> np.ndarray:
    source = validate_performance(performance)
    tensor = np.empty((len(ALTERNATIVES), len(CRITERIA), len(EXPERTS)), dtype=object)
    for ai, alternative in enumerate(ALTERNATIVES):
        for di, criterion in enumerate(CRITERIA):
            subset = source[
                (source["alternative_id"] == alternative)
                & (source["criterion_id"] == criterion)
            ].set_index("expert_id").reindex(EXPERTS)
            tensor[ai, di, :] = subset["term"].to_numpy(dtype=object)
    return tensor


def compute_cells(
    tensor: np.ndarray,
    participant_weights: Sequence[float],
    config: AggregationConfig,
    sampled_expert_indices: Sequence[int] | None = None,
    detailed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = np.asarray(participant_weights, dtype=float)
    if sampled_expert_indices is None:
        if q.shape != (len(EXPERTS),):
            raise ValueError("participant weights must have length 10")
        selected = np.arange(len(EXPERTS), dtype=int)
        weights = q / q.sum()
        labels = np.asarray(EXPERTS, dtype=object)
    else:
        selected = np.asarray(sampled_expert_indices, dtype=int)
        if np.any((selected < 0) | (selected >= len(EXPERTS))):
            raise ValueError("sampled expert index outside 0..9")
        weights = np.ones(len(selected), dtype=float) / len(selected)
        labels = np.asarray([f"sample_{pos + 1}:{EXPERTS[index]}" for pos, index in enumerate(selected)])
    u0 = np.zeros((len(ALTERNATIVES), len(CRITERIA)), dtype=float)
    lower = np.zeros_like(u0)
    upper = np.zeros_like(u0)
    cell_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for ai, alternative in enumerate(ALTERNATIVES):
        for di, criterion in enumerate(CRITERIA):
            terms = [str(x) for x in tensor[ai, di, selected]]
            result = aggregate_cell(terms, weights, config)
            u0[ai, di] = float(result["u0"])
            lower[ai, di] = float(result["u_lower"])
            upper[ai, di] = float(result["u_upper"])
            cell = {
                "alternative_id": alternative,
                "criterion_id": criterion,
                "u0": result["u0"],
                "u_lower": result["u_lower"],
                "u_upper": result["u_upper"],
                "interval_width": result["u_upper"] - result["u_lower"],
                "m_L": result["region_mass"]["L"],
                "m_B": result["region_mass"]["B"],
                "m_Neg": result["region_mass"]["Neg"],
                "boundary_discordance": result["boundary_discordance"],
                "rho": result["rho"],
                "ignorance_mass": result["ignorance_mass"],
                "epsilon_n": result["epsilon_n"],
                "epsilon_p": result["epsilon_p"],
                "config_threshold": config.threshold_term,
                "config_target_mode": config.target_mode,
                "config_median_band_radius": config.median_band_radius,
                "config_epsilon_multiplier": config.epsilon_multiplier,
                "config_nu": config.nu,
                "config_t2nn_weight": config.t2nn_modal_weight,
                "config_boundary_metric": config.boundary_metric,
                "config_rho_power": config.rho_power,
                "config_equal_spacing": config.equal_utility_spacing,
            }
            for h, term in enumerate(TERMS):
                cell[f"p0_{term}"] = result["p_nominal"][h]
                cell[f"singleton_{term}"] = result["singleton_mass"][h]
            cell_rows.append(cell)
            if detailed:
                for region_name, data in result["regional"].items():
                    row: dict[str, object] = {
                        "alternative_id": alternative,
                        "criterion_id": criterion,
                        "region": region_name,
                        "region_mass": data["mass"],
                        "member_count": len(data["members"]),
                    }
                    if data["mass"] > 0.0:
                        row["t2nn_score"] = data["t2nn_score"]
                        for h, term in enumerate(TERMS):
                            row[f"p_plts_{term}"] = data["p_plts"][h]
                            row[f"p_t2nn_{term}"] = data["p_t2nn"][h]
                            row[f"p_mix_{term}"] = data["p_mix"][h]
                    region_rows.append(row)
                region_assignment = np.full(len(selected), "", dtype=object)
                for region_name, mask in result["regions"].items():
                    region_assignment[mask] = region_name
                diagnostics = result["rough_diagnostics"]
                for position, label in enumerate(labels):
                    membership_rows.append(
                        {
                            "alternative_id": alternative,
                            "criterion_id": criterion,
                            "participant_record": label,
                            "term": terms[position],
                            "participant_weight": weights[position],
                            "target_member": diagnostics["target"][position],
                            "target_reference_index_zero_based": diagnostics[
                                "target_reference_index_zero_based"
                            ][position],
                            "lower_N": diagnostics["lower_N"][position],
                            "lower_P": diagnostics["lower_P"][position],
                            "upper_N": diagnostics["upper_N"][position],
                            "upper_P": diagnostics["upper_P"][position],
                            "rough_region": region_assignment[position],
                        }
                    )
    return (
        u0,
        lower,
        upper,
        pd.DataFrame(cell_rows),
        pd.DataFrame(region_rows),
        pd.DataFrame(membership_rows),
    )


def ranking_table(
    u0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Sequence[float],
    scenario: str,
    lambda_objective: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = interval_spotis(u0, lower, upper, weights)
    ranks = dense_ranks_ascending(result["distance_nominal"])
    rows = []
    for index, alternative in enumerate(ALTERNATIVES):
        rows.append(
            {
                "scenario": scenario,
                "lambda_objective": lambda_objective,
                "alternative_id": alternative,
                "spotis_D0": result["distance_nominal"][index],
                "spotis_D_lower": result["distance_lower"][index],
                "spotis_D_upper": result["distance_upper"][index],
                "rank": ranks[index],
                "necessary_outdegree": int(result["necessary_preference"][index].sum()),
                "possible_outdegree": int(result["possible_preference"][index].sum()),
            }
        )
    pair_rows = []
    for i, left in enumerate(ALTERNATIVES):
        for j, right in enumerate(ALTERNATIVES):
            if i == j:
                continue
            pair_rows.append(
                {
                    "scenario": scenario,
                    "lambda_objective": lambda_objective,
                    "left_alternative": left,
                    "right_alternative": right,
                    "left_nominally_preferred": bool(
                        result["distance_nominal"][i] < result["distance_nominal"][j] - 1e-12
                    ),
                    "left_necessarily_preferred": bool(result["necessary_preference"][i, j]),
                    "left_possibly_preferred": bool(result["possible_preference"][i, j]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def exact_lambda_analysis(
    u0: np.ndarray,
    objective: Sequence[float],
    subjective: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    objective = np.asarray(objective, dtype=float)
    subjective = np.asarray(subjective, dtype=float)
    d_objective = (1.0 - u0) @ (objective / objective.sum())
    d_subjective = (1.0 - u0) @ (subjective / subjective.sum())
    crossings: list[dict[str, object]] = []
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
                        "distance_at_crossing": (1.0 - root) * d_subjective[i] + root * d_objective[i],
                    }
                )
    points = sorted(breakpoints)
    segments: list[dict[str, object]] = []
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


def scenario_config(base: AggregationConfig, **changes: object) -> AggregationConfig:
    values = asdict(base)
    values.update(changes)
    return AggregationConfig(**values)
