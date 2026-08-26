"""Corrected paired-semantic PLTS-T2NN aggregation primitives.

The module is deliberately independent of the legacy ranking pipeline.  It
implements the frozen 2026-08-07 contract: individual one-hot PLTS ratings,
the fixed nine-component T2NN codebook, dual tolerance rough regions,
within-source aggregation, a fixed cross-modal convex pool, boundary-only
ignorance routing, and fixed-domain interval SPOTIS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


TERMS = ("VWI", "PI", "EI", "SI", "VSI", "AI")
TERM_TO_INDEX = {term: index for index, term in enumerate(TERMS)}

# Entries are (TT, TI, TF, IT, II, IF, FT, FI, FF).
T2NN_CODEBOOK: dict[str, tuple[float, ...]] = {
    "VWI": (0.15, 0.20, 0.10, 0.70, 0.85, 0.85, 0.60, 0.80, 0.70),
    "PI": (0.35, 0.30, 0.25, 0.60, 0.70, 0.80, 0.50, 0.75, 0.65),
    "EI": (0.45, 0.35, 0.35, 0.50, 0.65, 0.55, 0.40, 0.65, 0.55),
    "SI": (0.60, 0.55, 0.55, 0.25, 0.40, 0.45, 0.25, 0.45, 0.35),
    "VSI": (0.70, 0.65, 0.65, 0.20, 0.30, 0.40, 0.20, 0.25, 0.30),
    "AI": (0.95, 0.90, 0.95, 0.10, 0.10, 0.05, 0.05, 0.05, 0.10),
}


def _as_probability_weights(weights: Sequence[float], length: int) -> np.ndarray:
    q = np.asarray(weights, dtype=float)
    if q.shape != (length,) or not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("weights must be a finite nonnegative vector aligned to inputs")
    total = float(q.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return q / total


def t2nn_score(number: Sequence[float]) -> float:
    x = np.asarray(number, dtype=float)
    if x.shape != (9,) or not np.all(np.isfinite(x)):
        raise ValueError("T2NN score requires nine finite components")
    truth = x[0] + 2.0 * x[1] + x[2]
    indeterminacy = x[3] + 2.0 * x[4] + x[5]
    falsity = x[6] + 2.0 * x[7] + x[8]
    return float((8.0 + truth - indeterminacy - falsity) / 12.0)


ANCHOR_SCORES = np.asarray([t2nn_score(T2NN_CODEBOOK[t]) for t in TERMS])
if not np.all(np.diff(ANCHOR_SCORES) > 0.0):
    raise RuntimeError("The frozen T2NN codebook is not strictly ordered")
UTILITY_COORDS = (ANCHOR_SCORES - ANCHOR_SCORES[0]) / (
    ANCHOR_SCORES[-1] - ANCHOR_SCORES[0]
)
EQUAL_UTILITY_COORDS = np.linspace(0.0, 1.0, len(TERMS))
CODEBOOK_MATRIX = np.asarray([T2NN_CODEBOOK[t] for t in TERMS], dtype=float)


def one_hot(term: str) -> np.ndarray:
    if term not in TERM_TO_INDEX:
        raise ValueError(f"unknown linguistic term: {term!r}")
    result = np.zeros(len(TERMS), dtype=float)
    result[TERM_TO_INDEX[term]] = 1.0
    return result


def aa_tnorm(values: Sequence[float], weights: Sequence[float], nu: float) -> float:
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    x = np.asarray(values, dtype=float)
    q = _as_probability_weights(weights, len(x))
    if np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("AA inputs must lie in [0,1]")
    active = q > 0.0
    if np.any(x[active] == 0.0):
        return 0.0
    generator_mean = float(np.sum(q[active] * np.power(-np.log(x[active]), nu)))
    return float(np.exp(-generator_mean ** (1.0 / nu)))


def aa_tconorm(values: Sequence[float], weights: Sequence[float], nu: float) -> float:
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    x = np.asarray(values, dtype=float)
    q = _as_probability_weights(weights, len(x))
    if np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("AA inputs must lie in [0,1]")
    active = q > 0.0
    if np.any(x[active] == 1.0):
        return 1.0
    generator_mean = float(np.sum(q[active] * np.power(-np.log1p(-x[active]), nu)))
    return float(1.0 - np.exp(-generator_mean ** (1.0 / nu)))


def aggregate_t2nn(numbers: Sequence[Sequence[float]], weights: Sequence[float], nu: float) -> np.ndarray:
    matrix = np.asarray(numbers, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 9 or len(matrix) == 0:
        raise ValueError("T2NN aggregation requires a nonempty K by 9 matrix")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("T2NN components must be finite values in [0,1]")
    q = _as_probability_weights(weights, len(matrix))
    truth = [aa_tconorm(matrix[:, j], q, nu) for j in range(3)]
    indeterminacy_falsity = [aa_tnorm(matrix[:, j], q, nu) for j in range(3, 9)]
    return np.asarray(truth + indeterminacy_falsity, dtype=float)


def project_score_to_support(score: float, utility_coords: Sequence[float] | None = None) -> np.ndarray:
    """Project a T2NN score to adjacent linguistic anchors exactly."""

    if not np.isfinite(score):
        raise ValueError("projection score must be finite")
    g = UTILITY_COORDS if utility_coords is None else np.asarray(utility_coords, dtype=float)
    if g.shape != (len(TERMS),) or not np.all(np.diff(g) > 0.0):
        raise ValueError("utility coordinates must be strictly increasing")
    # Projection location is defined on raw T2NN anchor scores.  The supplied
    # g controls only downstream utility and W1 geometry.
    if score <= ANCHOR_SCORES[0]:
        return one_hot(TERMS[0])
    if score >= ANCHOR_SCORES[-1]:
        return one_hot(TERMS[-1])
    right = int(np.searchsorted(ANCHOR_SCORES, score, side="right"))
    left = right - 1
    fraction = float(
        (score - ANCHOR_SCORES[left])
        / (ANCHOR_SCORES[right] - ANCHOR_SCORES[left])
    )
    result = np.zeros(len(TERMS), dtype=float)
    result[left] = 1.0 - fraction
    result[right] = fraction
    return result


def ordered_w1(p: Sequence[float], q: Sequence[float], coords: Sequence[float] = UTILITY_COORDS) -> float:
    left = np.asarray(p, dtype=float)
    right = np.asarray(q, dtype=float)
    g = np.asarray(coords, dtype=float)
    if left.shape != (len(TERMS),) or right.shape != left.shape or g.shape != left.shape:
        raise ValueError("W1 inputs must use the six-term support")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) or not np.all(np.isfinite(g)):
        raise ValueError("W1 inputs must be finite")
    if np.any(left < -1e-12) or np.any(right < -1e-12):
        raise ValueError("probability mass cannot be negative")
    if not math.isclose(float(left.sum()), 1.0, abs_tol=1e-10) or not math.isclose(
        float(right.sum()), 1.0, abs_tol=1e-10
    ):
        raise ValueError("W1 inputs must be normalized")
    if not np.all(np.diff(g) > 0.0):
        raise ValueError("W1 support must be strictly ordered")
    return float(np.sum(np.abs(np.cumsum(left - right)[:-1]) * np.diff(g)))


def jsd_base2(p: Sequence[float], q: Sequence[float]) -> float:
    left = np.asarray(p, dtype=float)
    right = np.asarray(q, dtype=float)
    if left.shape != (len(TERMS),) or right.shape != left.shape:
        raise ValueError("JSD inputs must use the six-term support")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("JSD inputs must be finite")
    if np.any(left < -1e-12) or np.any(right < -1e-12):
        raise ValueError("probability mass cannot be negative")
    if left.sum() <= 0.0 or right.sum() <= 0.0:
        raise ValueError("JSD inputs must have positive total mass")
    left = left / left.sum()
    right = right / right.sum()
    midpoint = 0.5 * (left + right)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def default_tolerances(
    coords: Sequence[float] = UTILITY_COORDS,
    codebook: np.ndarray = CODEBOOK_MATRIX,
) -> tuple[float, float]:
    """Return thresholds that include every adjacent linguistic pair."""

    g = np.asarray(coords, dtype=float)
    matrix = np.asarray(codebook, dtype=float)
    if g.shape != (len(TERMS),) or matrix.shape != (len(TERMS), 9):
        raise ValueError("invalid codebook geometry")
    epsilon_n = float(np.max(np.sum(np.abs(np.diff(matrix, axis=0)), axis=1) / 9.0))
    epsilon_p = float(np.max(np.diff(g)))
    return epsilon_n, epsilon_p


@dataclass(frozen=True)
class AggregationConfig:
    threshold_term: str = "SI"
    target_mode: str = "threshold_upper_set"
    median_band_radius: int = 1
    epsilon_multiplier: float = 1.0
    nu: float = 3.0
    t2nn_modal_weight: float = 0.5
    boundary_metric: str = "W1"
    rho_power: float = 1.0
    equal_utility_spacing: bool = False

    def validate(self) -> None:
        if self.threshold_term not in TERM_TO_INDEX:
            raise ValueError("unknown threshold term")
        if self.target_mode not in {"threshold_upper_set", "weighted_median_band"}:
            raise ValueError("unknown rough target mode")
        if self.median_band_radius < 0:
            raise ValueError("median-band radius cannot be negative")
        if self.epsilon_multiplier <= 0.0 or self.nu <= 0.0:
            raise ValueError("epsilon multiplier and nu must be positive")
        if not 0.0 <= self.t2nn_modal_weight <= 1.0:
            raise ValueError("T2NN modal weight must lie in [0,1]")
        if self.boundary_metric not in {"W1", "JSD"}:
            raise ValueError("boundary metric must be W1 or JSD")
        if self.rho_power <= 0.0:
            raise ValueError("rho exponent must be positive")


def _pairwise_geometry(terms: Sequence[str], coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([TERM_TO_INDEX[t] for t in terms], dtype=int)
    numbers = CODEBOOK_MATRIX[indices]
    distance_n = np.sum(np.abs(numbers[:, None, :] - numbers[None, :, :]), axis=2) / 9.0
    # W1 between one-hot distributions is the distance between support points.
    distance_p = np.abs(coords[indices, None] - coords[None, indices])
    return distance_n, distance_p


def _rough_regions(
    terms: Sequence[str],
    weights: np.ndarray,
    coords: np.ndarray,
    threshold_term: str,
    target_mode: str,
    median_band_radius: int,
    epsilon_multiplier: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, float]:
    distance_n, distance_p = _pairwise_geometry(terms, coords)
    epsilon_n_base, epsilon_p_base = default_tolerances(coords)
    epsilon_n = epsilon_multiplier * epsilon_n_base
    epsilon_p = epsilon_multiplier * epsilon_p_base
    relations = {
        "N": distance_n <= epsilon_n + 1e-12,
        "P": distance_p <= epsilon_p + 1e-12,
    }
    term_indices = np.asarray([TERM_TO_INDEX[t] for t in terms], dtype=int)
    if target_mode == "threshold_upper_set":
        target = term_indices >= TERM_TO_INDEX[threshold_term]
        target_reference = TERM_TO_INDEX[threshold_term]
    else:
        level_mass = np.bincount(term_indices, weights=weights, minlength=len(TERMS))
        target_reference = int(np.searchsorted(np.cumsum(level_mass), 0.5, side="left"))
        target = np.abs(term_indices - target_reference) <= median_band_radius
    lowers: dict[str, np.ndarray] = {}
    uppers: dict[str, np.ndarray] = {}
    for name, relation in relations.items():
        lowers[name] = np.asarray([bool(np.all(target[row])) for row in relation])
        uppers[name] = np.asarray([bool(np.any(target[row])) for row in relation])
    lower = lowers["N"] & lowers["P"]
    upper = uppers["N"] | uppers["P"]
    regions = {
        "L": lower,
        "B": upper & ~lower,
        "Neg": ~upper,
    }
    diagnostics = {
        "target": target,
        "lower_N": lowers["N"],
        "lower_P": lowers["P"],
        "upper_N": uppers["N"],
        "upper_P": uppers["P"],
        "target_reference_index_zero_based": np.full(len(terms), target_reference, dtype=int),
    }
    return regions, diagnostics, epsilon_n, epsilon_p


def aggregate_global(
    terms: Sequence[str],
    weights: Sequence[float],
    config: AggregationConfig | None = None,
) -> dict[str, object]:
    cfg = config or AggregationConfig()
    cfg.validate()
    if len(terms) == 0 or any(t not in TERM_TO_INDEX for t in terms):
        raise ValueError("terms must be a nonempty sequence from the frozen scale")
    q = _as_probability_weights(weights, len(terms))
    coords = EQUAL_UTILITY_COORDS if cfg.equal_utility_spacing else UTILITY_COORDS
    indices = np.asarray([TERM_TO_INDEX[t] for t in terms], dtype=int)
    p_plts = np.bincount(indices, weights=q, minlength=len(TERMS)).astype(float)
    number = aggregate_t2nn(CODEBOOK_MATRIX[indices], q, cfg.nu)
    p_t2nn = project_score_to_support(t2nn_score(number), coords)
    omega_n = cfg.t2nn_modal_weight
    p_mix = omega_n * p_t2nn + (1.0 - omega_n) * p_plts
    return {
        "p_plts": p_plts,
        "t2nn": number,
        "p_t2nn": p_t2nn,
        "p_mix": p_mix,
        "u_plts": float(coords @ p_plts),
        "u_t2nn": float(coords @ p_t2nn),
        "u_mix": float(coords @ p_mix),
    }


def aggregate_cell(
    terms: Sequence[str],
    weights: Sequence[float],
    config: AggregationConfig | None = None,
) -> dict[str, object]:
    cfg = config or AggregationConfig()
    cfg.validate()
    if len(terms) == 0 or any(t not in TERM_TO_INDEX for t in terms):
        raise ValueError("terms must be a nonempty sequence from the frozen scale")
    q = _as_probability_weights(weights, len(terms))
    coords = EQUAL_UTILITY_COORDS if cfg.equal_utility_spacing else UTILITY_COORDS
    regions, rough_diag, epsilon_n, epsilon_p = _rough_regions(
        terms,
        q,
        coords,
        cfg.threshold_term,
        cfg.target_mode,
        cfg.median_band_radius,
        cfg.epsilon_multiplier,
    )
    indices = np.asarray([TERM_TO_INDEX[t] for t in terms], dtype=int)
    omega_n = cfg.t2nn_modal_weight
    regional: dict[str, dict[str, object]] = {}
    p_nominal = np.zeros(len(TERMS), dtype=float)
    region_mass: dict[str, float] = {}
    for name, mask in regions.items():
        mass = float(q[mask].sum())
        region_mass[name] = mass
        if mass <= 1e-15:
            regional[name] = {"mass": 0.0, "members": np.flatnonzero(mask)}
            continue
        q_region = q[mask] / mass
        local_indices = indices[mask]
        p_plts = np.bincount(local_indices, weights=q_region, minlength=len(TERMS)).astype(float)
        number = aggregate_t2nn(CODEBOOK_MATRIX[local_indices], q_region, cfg.nu)
        score = t2nn_score(number)
        p_t2nn = project_score_to_support(score, coords)
        p_mix = omega_n * p_t2nn + (1.0 - omega_n) * p_plts
        regional[name] = {
            "mass": mass,
            "members": np.flatnonzero(mask),
            "conditional_weights": q_region,
            "p_plts": p_plts,
            "t2nn": number,
            "t2nn_score": score,
            "p_t2nn": p_t2nn,
            "p_mix": p_mix,
        }
        p_nominal += mass * p_mix

    boundary_mass = region_mass["B"]
    if boundary_mass > 1e-15:
        boundary = regional["B"]
        if cfg.boundary_metric == "W1":
            discordance = ordered_w1(boundary["p_t2nn"], boundary["p_plts"], coords)
            span = float(coords[-1] - coords[0])
            discordance = discordance / span
        else:
            discordance = jsd_base2(boundary["p_t2nn"], boundary["p_plts"])
        discordance = float(np.clip(discordance, 0.0, 1.0))
        rho = discordance ** cfg.rho_power
        p_boundary = np.asarray(boundary["p_mix"], dtype=float)
    else:
        discordance = 0.0
        rho = 0.0
        p_boundary = np.zeros(len(TERMS), dtype=float)

    ignorance_mass = boundary_mass * rho
    singleton = np.zeros(len(TERMS), dtype=float)
    for name in ("L", "Neg"):
        if region_mass[name] > 1e-15:
            singleton += region_mass[name] * np.asarray(regional[name]["p_mix"], dtype=float)
    if boundary_mass > 1e-15:
        singleton += boundary_mass * (1.0 - rho) * p_boundary

    u0 = float(coords @ p_nominal)
    u_lower = float(coords @ singleton + ignorance_mass * coords[0])
    u_upper = float(coords @ singleton + ignorance_mass * coords[-1])
    recovered = singleton + ignorance_mass * p_boundary

    return {
        "config": cfg,
        "coords": coords.copy(),
        "epsilon_n": epsilon_n,
        "epsilon_p": epsilon_p,
        "regions": regions,
        "rough_diagnostics": rough_diag,
        "regional": regional,
        "region_mass": region_mass,
        "p_nominal": p_nominal,
        "singleton_mass": singleton,
        "ignorance_mass": ignorance_mass,
        "boundary_discordance": discordance,
        "rho": rho,
        "nominal_recovered": recovered,
        "u0": u0,
        "u_lower": u_lower,
        "u_upper": u_upper,
    }


def combine_criterion_weights(
    objective: Sequence[float], subjective: Sequence[float], lambda_objective: float
) -> np.ndarray:
    if not 0.0 <= lambda_objective <= 1.0:
        raise ValueError("lambda must lie in [0,1]")
    left = _as_probability_weights(objective, len(objective))
    right = _as_probability_weights(subjective, len(subjective))
    if left.shape != right.shape:
        raise ValueError("objective and subjective weights must align")
    result = lambda_objective * left + (1.0 - lambda_objective) * right
    return result / result.sum()


def interval_spotis(
    u0: Sequence[Sequence[float]],
    u_lower: Sequence[Sequence[float]],
    u_upper: Sequence[Sequence[float]],
    criterion_weights: Sequence[float],
) -> dict[str, np.ndarray]:
    nominal = np.asarray(u0, dtype=float)
    lower = np.asarray(u_lower, dtype=float)
    upper = np.asarray(u_upper, dtype=float)
    if nominal.ndim != 2 or lower.shape != nominal.shape or upper.shape != nominal.shape:
        raise ValueError("utility matrices must have the same alternatives-by-criteria shape")
    if np.any(lower > nominal + 1e-10) or np.any(nominal > upper + 1e-10):
        raise ValueError("utility intervals do not contain nominal utility")
    if np.any(lower < -1e-10) or np.any(upper > 1.0 + 1e-10):
        raise ValueError("fixed-domain SPOTIS expects utility bounds in [0,1]")
    weights = _as_probability_weights(criterion_weights, nominal.shape[1])
    distance_nominal = (1.0 - nominal) @ weights
    distance_lower = (1.0 - upper) @ weights
    distance_upper = (1.0 - lower) @ weights
    necessary = distance_upper[:, None] < distance_lower[None, :] - 1e-12
    possible = distance_lower[:, None] < distance_upper[None, :] - 1e-12
    np.fill_diagonal(necessary, False)
    np.fill_diagonal(possible, False)
    return {
        "distance_nominal": distance_nominal,
        "distance_lower": distance_lower,
        "distance_upper": distance_upper,
        "necessary_preference": necessary,
        "possible_preference": possible,
    }


def dense_ranks_ascending(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=int)
    rank = 1
    ranks[order[0]] = rank
    for position in range(1, len(order)):
        if not math.isclose(float(x[order[position]]), float(x[order[position - 1]]), abs_tol=1e-12):
            rank += 1
        ranks[order[position]] = rank
    return ranks
