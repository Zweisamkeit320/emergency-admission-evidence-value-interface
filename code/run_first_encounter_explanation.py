"""Re-estimate first-encounter DD-CDW in the frozen explanation partition.

This script uses the already-fitted model and the original patient-model
pipeline.  It never loads the internal-evaluation partition and writes no
patient-level rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_patient_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model-freeze", required=True, type=Path)
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(args.pipeline)
    freeze = json.loads(args.model_freeze.read_text(encoding="utf-8"))

    if sha256(args.features) != freeze["feature_file_sha256"]:
        raise ValueError("Feature table does not match the frozen patient model")

    primary_name = freeze["selection"]["primary_candidate"]
    model_record = next(
        record for record in freeze["artifacts"] if record["candidate"] == primary_name
    )
    if sha256(args.model_artifact) != model_record["artifact_sha256"]:
        raise ValueError("Model artifact does not match model_freeze.json")

    columns = [
        "subject_id",
        "stay_id",
        "intime",
        "analysis_split",
        "label_admitted",
        *pipeline.RAW_FEATURE_COLUMNS,
    ]
    explanation = pd.read_parquet(
        args.features,
        columns=columns,
        filters=[("analysis_split", "==", "tuning")],
    )
    pipeline.validate_feature_contract(explanation)
    observed_splits = sorted(explanation["analysis_split"].unique().tolist())
    if observed_splits != ["tuning"]:
        raise ValueError(f"Expected only tuning, found {observed_splits}")

    explanation["intime"] = pd.to_datetime(explanation["intime"], errors="raise")
    first = (
        explanation.sort_values(["subject_id", "intime", "stay_id"], kind="mergesort")
        .drop_duplicates("subject_id", keep="first")
        .reset_index(drop=True)
    )
    if len(first) != explanation["subject_id"].nunique():
        raise AssertionError("First-encounter selection did not yield one row per patient")
    if first["subject_id"].duplicated().any():
        raise AssertionError("First-encounter subset contains duplicate patients")

    artifact = joblib.load(args.model_artifact)
    weights, _private_patient_delta, raw_bootstrap = pipeline.weights_for_artifact(
        artifact,
        first,
        args.permutation_repeats,
        args.bootstrap_repeats,
        args.seed,
    )
    weights.insert(0, "analysis", "first_encounter_explanation_partition")
    weights.insert(1, "criterion_id", [f"D{i}" for i in range(1, 11)])
    weights.to_csv(
        args.output_dir / "first_encounter_explanation_ddcdw.csv",
        index=False,
        float_format="%.17g",
    )

    raw = raw_bootstrap.to_numpy(dtype=float)
    positive = np.maximum(raw, 0.0)
    denominator = positive.sum(axis=1, keepdims=True)
    normalized = np.divide(
        positive,
        denominator,
        out=np.full_like(positive, np.nan),
        where=denominator > 1e-12,
    )
    bootstrap = pd.concat(
        [
            raw_bootstrap.add_prefix("raw_"),
            pd.DataFrame(normalized, columns=[f"weight_{x}" for x in raw_bootstrap.columns]),
        ],
        axis=1,
    )
    bootstrap.insert(0, "bootstrap_draw", np.arange(1, len(bootstrap) + 1))
    bootstrap.to_csv(
        args.output_dir / "first_encounter_explanation_ddcdw_bootstrap.csv",
        index=False,
        float_format="%.17g",
    )

    provenance = {
        "analysis": "first chronological eligible encounter per patient",
        "input_partition": "tuning",
        "input_partition_role": "explanation/weight-estimation",
        "all_explanation_rows": int(len(explanation)),
        "all_explanation_subjects": int(explanation["subject_id"].nunique()),
        "first_encounter_rows": int(len(first)),
        "first_encounter_subjects": int(first["subject_id"].nunique()),
        "selection_order": ["subject_id", "intime", "stay_id"],
        "primary_candidate": primary_name,
        "permutation_repeats": args.permutation_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
        "method": "joint raw-domain marginal permutation; Brier-loss increment; positive-part normalization; patient bootstrap",
        "internal_evaluation_partition_loaded": False,
        "patient_level_rows_written": False,
        "feature_file_sha256": sha256(args.features),
        "model_artifact_sha256": sha256(args.model_artifact),
        "pipeline_sha256": sha256(args.pipeline),
        "weights_sha256": sha256(args.output_dir / "first_encounter_explanation_ddcdw.csv"),
    }
    (args.output_dir / "first_encounter_explanation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
