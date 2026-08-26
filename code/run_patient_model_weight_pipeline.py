from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RANDOM_SEED = 20260804
REPORT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]

STAGE_SPLITS = {
    "cv": ["development"],
    "fit": ["development", "calibration", "tuning"],
    "weights": ["tuning"],
    "test": ["patient_disjoint_internal_test"],
}

DOMAIN_MAP: dict[str, list[str]] = {
    "D1_prior_testing_treatment": [
        "lab_prev_30d",
        "lab_prev_365d",
        "micro_prev_30d",
        "micro_prev_365d",
        "icu_procedure_prev_30d",
        "icu_procedure_prev_365d",
    ],
    "D2_prior_utilization": [
        "ed_visits_prev_30d",
        "ed_visits_prev_365d",
        "hospital_admissions_prev_30d",
        "hospital_admissions_prev_365d",
    ],
    "D3_triage_urgency_pain": [
        "triage_acuity",
        "triage_acuity_missing",
        "triage_pain",
        "triage_pain_missing",
    ],
    "D4_patient_vulnerability": ["age_years", "prior_inpatient_charlson_index"],
    "D5_respiratory_status": [
        "triage_resprate",
        "triage_resprate_missing",
        "triage_o2sat",
        "triage_o2sat_missing",
    ],
    "D6_prior_medication_device": [
        "medrecon_prev_30d",
        "medrecon_prev_365d",
        "pyxis_prev_30d",
        "pyxis_prev_365d",
        "antibiotic_any_prev_365d",
    ],
    "D7_circulatory_status": [
        "triage_sbp",
        "triage_sbp_missing",
        "triage_dbp",
        "triage_dbp_missing",
        "triage_heartrate",
        "triage_heartrate_missing",
        "triage_map",
        "triage_shock_index",
    ],
    "D8_arrival_mode": ["arrival_transport"],
    "D9_thermal_inflammatory": [
        "triage_temperature_c",
        "triage_temperature_c_missing",
    ],
    "D10_consciousness_neurologic": ["ams_flag_narrow", "chiefcomplaint_missing"],
}

CONTINUOUS_COLUMNS = [
    "age_years",
    "triage_acuity",
    "triage_pain",
    "triage_resprate",
    "triage_o2sat",
    "triage_sbp",
    "triage_dbp",
    "triage_heartrate",
    "triage_map",
    "triage_shock_index",
    "triage_temperature_c",
    "prior_inpatient_charlson_index",
]
COUNT_COLUMNS = [
    "lab_prev_30d",
    "lab_prev_365d",
    "micro_prev_30d",
    "micro_prev_365d",
    "icu_procedure_prev_30d",
    "icu_procedure_prev_365d",
    "ed_visits_prev_30d",
    "ed_visits_prev_365d",
    "hospital_admissions_prev_30d",
    "hospital_admissions_prev_365d",
    "medrecon_prev_30d",
    "medrecon_prev_365d",
    "pyxis_prev_30d",
    "pyxis_prev_365d",
]
BINARY_COLUMNS = [
    "triage_acuity_missing",
    "triage_pain_missing",
    "triage_resprate_missing",
    "triage_o2sat_missing",
    "triage_sbp_missing",
    "triage_dbp_missing",
    "triage_heartrate_missing",
    "triage_temperature_c_missing",
    "antibiotic_any_prev_365d",
    "ams_flag_narrow",
    "chiefcomplaint_missing",
]
CATEGORICAL_COLUMNS = ["arrival_transport"]
RAW_FEATURE_COLUMNS = [column for columns in DOMAIN_MAP.values() for column in columns]

LOGISTIC_CANDIDATES = {
    "logistic_c0.01": {"C": 0.01},
    "logistic_c0.1": {"C": 0.1},
    "logistic_c1": {"C": 1.0},
}
XGB_CANDIDATES = {
    "xgb_shallow": {
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    "xgb_regularized": {
        "n_estimators": 900,
        "max_depth": 2,
        "learning_rate": 0.035,
        "min_child_weight": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 10.0,
        "reg_alpha": 0.1,
    },
    "xgb_medium": {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.035,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 8.0,
        "reg_alpha": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run development CV, freeze models, estimate grouped weights, or test."
    )
    parser.add_argument("stage", choices=["cv", "fit", "weights", "test"])
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--xgb-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--confirm-locked-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_feature_contract(frame: pd.DataFrame) -> None:
    missing = sorted(set(RAW_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing locked raw features: {missing}")
    duplicates = [
        column
        for column in RAW_FEATURE_COLUMNS
        if sum(column in values for values in DOMAIN_MAP.values()) != 1
    ]
    if duplicates:
        raise ValueError(f"Features without exactly one domain: {duplicates}")
    if len(RAW_FEATURE_COLUMNS) != len(set(RAW_FEATURE_COLUMNS)):
        raise ValueError("RAW_FEATURE_COLUMNS contains duplicates")
    if len(RAW_FEATURE_COLUMNS) != 38:
        raise ValueError(f"Expected 38 primary features, got {len(RAW_FEATURE_COLUMNS)}")
    required_meta = {"subject_id", "stay_id", "analysis_split", "label_admitted"}
    if not required_meta.issubset(frame.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required_meta - set(frame.columns))}")
    if frame["stay_id"].duplicated().any():
        raise ValueError("stay_id is not unique")
    if set(frame["label_admitted"].unique()) - {0, 1}:
        raise ValueError("label_admitted is not binary")
    per_subject_splits = (
        frame[["subject_id", "analysis_split"]]
        .drop_duplicates()
        .groupby("subject_id")["analysis_split"]
        .nunique()
    )
    if int(per_subject_splits.max()) != 1:
        raise ValueError("A subject occurs in more than one analysis split")


def load_stage_frame(feature_path: Path, stage: str) -> pd.DataFrame:
    """Read only the partitions authorized for a pipeline stage."""
    expected_splits = STAGE_SPLITS[stage]
    frame = pd.read_parquet(
        feature_path,
        filters=[("analysis_split", "in", expected_splits)],
    )
    validate_feature_contract(frame)
    observed_splits = set(frame["analysis_split"].unique())
    if observed_splits != set(expected_splits):
        raise ValueError(
            f"Stage {stage} expected splits {expected_splits}, observed {sorted(observed_splits)}"
        )
    print(
        f"DATA_LOAD stage={stage} splits={','.join(expected_splits)} rows={len(frame)}",
        flush=True,
    )
    return frame


def prepare_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select locked features and normalize categorical nulls before serialization."""
    prepared = frame[RAW_FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype("object").where(
            pd.notna(prepared[column]), np.nan
        )
    return prepared


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    counts = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one", validate=False),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )
    binary = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous, CONTINUOUS_COLUMNS),
            ("counts", counts, COUNT_COLUMNS),
            ("binary", binary, BINARY_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def calibration_intercept_slope(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_intercept_slope(y, probability)
    return {
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def make_logistic(params: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(
        C=params["C"],
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )


def make_xgb(params: dict[str, Any], device: str, early_stopping: bool) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
        random_state=RANDOM_SEED,
        early_stopping_rounds=40 if early_stopping else None,
    )


def cv_stage(frame: pd.DataFrame, output_dir: Path, n_splits: int, device: str) -> None:
    cv_path = output_dir / "cv_results.csv"
    selection_path = output_dir / "cv_selection.json"
    if cv_path.exists() or selection_path.exists():
        raise FileExistsError("CV output already exists; use a new output directory to avoid overwrite")

    development = frame.loc[frame["analysis_split"] == "development"].reset_index(drop=True)
    x_raw = prepare_raw_features(development)
    y = development["label_admitted"].to_numpy(dtype=np.int8)
    groups = development["subject_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    fold_index = np.full(len(development), -1, dtype=np.int8)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_raw, y, groups), start=1):
        fold_index[valid_idx] = fold
        print(
            f"CV_FOLD_START fold={fold} train={len(train_idx)} valid={len(valid_idx)}",
            flush=True,
        )
        preprocessor = build_preprocessor()
        x_train = preprocessor.fit_transform(x_raw.iloc[train_idx]).astype(np.float32)
        x_valid = preprocessor.transform(x_raw.iloc[valid_idx]).astype(np.float32)
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        for candidate, params in LOGISTIC_CANDIDATES.items():
            started = time.perf_counter()
            model = make_logistic(params)
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_valid)[:, 1]
            row = {
                "candidate": candidate,
                "family": "logistic",
                "fold": fold,
                "fit_seconds": time.perf_counter() - started,
                "best_iteration": np.nan,
                **metrics(y_valid, probability),
            }
            rows.append(row)
            print(
                f"CV_RESULT candidate={candidate} fold={fold} "
                f"brier={row['brier']:.6f} auroc={row['auroc']:.6f}",
                flush=True,
            )

        for candidate, params in XGB_CANDIDATES.items():
            started = time.perf_counter()
            model = make_xgb(params, device=device, early_stopping=True)
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
            probability = model.predict_proba(x_valid)[:, 1]
            best_iteration = getattr(model, "best_iteration", params["n_estimators"] - 1)
            row = {
                "candidate": candidate,
                "family": "xgboost",
                "fold": fold,
                "fit_seconds": time.perf_counter() - started,
                "best_iteration": int(best_iteration),
                **metrics(y_valid, probability),
            }
            rows.append(row)
            print(
                f"CV_RESULT candidate={candidate} fold={fold} "
                f"brier={row['brier']:.6f} auroc={row['auroc']:.6f} "
                f"best_iteration={best_iteration}",
                flush=True,
            )

    if (fold_index < 1).any():
        raise RuntimeError("At least one development row was not assigned a CV fold")
    fold_manifest = development[["subject_id", "stay_id"]].copy()
    fold_manifest["cv_fold"] = fold_index
    fold_manifest.to_parquet(output_dir / "development_cv_folds.parquet", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(cv_path, index=False)
    summary = (
        results.groupby(["candidate", "family"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_brier=("brier", "mean"),
            sd_brier=("brier", "std"),
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_calibration_intercept=("calibration_intercept", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
            median_best_iteration=("best_iteration", "median"),
        )
        .sort_values(["mean_brier", "mean_log_loss", "candidate"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    best_row = summary.iloc[0]
    best_candidate = str(best_row["candidate"])
    best_se = float(best_row["sd_brier"] / math.sqrt(n_splits))
    logistic = summary.loc[summary["family"] == "logistic"].sort_values("mean_brier")
    best_logistic = logistic.iloc[0]
    if float(best_logistic["mean_brier"]) <= float(best_row["mean_brier"]) + best_se:
        primary_candidate = str(best_logistic["candidate"])
        selection_rule = "one-standard-error simplicity rule selected logistic"
    else:
        primary_candidate = best_candidate
        selection_rule = "minimum mean grouped-CV Brier score"
    alternative_family = "xgboost" if primary_candidate.startswith("logistic") else "logistic"
    alternative_candidate = str(
        summary.loc[summary["family"] == alternative_family]
        .sort_values("mean_brier")
        .iloc[0]["candidate"]
    )
    selection = {
        "primary_candidate": primary_candidate,
        "alternative_candidate": alternative_candidate,
        "unconstrained_best_candidate": best_candidate,
        "selection_rule": selection_rule,
        "best_candidate_brier_se": best_se,
        "n_splits": n_splits,
        "random_seed": RANDOM_SEED,
        "development_rows": int(len(development)),
        "development_subjects": int(development["subject_id"].nunique()),
        "input_partitions_loaded": sorted(frame["analysis_split"].unique().tolist()),
        "elapsed_seconds": time.perf_counter() - started_all,
        "locked_test_accessed": False,
    }
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)


def candidate_params(candidate: str, cv_results: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    if candidate in LOGISTIC_CANDIDATES:
        return "logistic", LOGISTIC_CANDIDATES[candidate]
    if candidate in XGB_CANDIDATES:
        params = dict(XGB_CANDIDATES[candidate])
        iterations = cv_results.loc[
            cv_results["candidate"] == candidate, "best_iteration"
        ].dropna()
        if not iterations.empty:
            params["n_estimators"] = max(50, int(np.median(iterations)) + 1)
        return "xgboost", params
    raise KeyError(candidate)


def fit_calibrator(y: np.ndarray, probability: np.ndarray) -> LogisticRegression:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    calibrator.fit(logit, y)
    return calibrator


def apply_calibrator(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def predict_artifact(artifact: dict[str, Any], raw: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
    transformed = artifact["preprocessor"].transform(prepare_raw_features(raw)).astype(
        np.float32
    )
    probability = artifact["model"].predict_proba(transformed)[:, 1]
    if calibrated:
        probability = apply_calibrator(artifact["calibrator"], probability)
    return probability


def fit_one_candidate(
    candidate: str,
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    cv_results: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    family, params = candidate_params(candidate, cv_results)
    preprocessor = build_preprocessor()
    x_development = preprocessor.fit_transform(prepare_raw_features(development)).astype(
        np.float32
    )
    y_development = development["label_admitted"].to_numpy(dtype=np.int8)
    if family == "logistic":
        model = make_logistic(params)
    else:
        model = make_xgb(params, device=device, early_stopping=False)
    model.fit(x_development, y_development)

    x_calibration = preprocessor.transform(prepare_raw_features(calibration)).astype(
        np.float32
    )
    y_calibration = calibration["label_admitted"].to_numpy(dtype=np.int8)
    uncalibrated = model.predict_proba(x_calibration)[:, 1]
    calibrator = fit_calibrator(y_calibration, uncalibrated)
    calibrated = apply_calibrator(calibrator, uncalibrated)
    return {
        "candidate": candidate,
        "family": family,
        "params": params,
        "preprocessor": preprocessor,
        "model": model,
        "calibrator": calibrator,
        "calibration_metrics_uncalibrated": metrics(y_calibration, uncalibrated),
        "calibration_metrics_calibrated": metrics(y_calibration, calibrated),
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "domain_map": DOMAIN_MAP,
    }


def fit_stage(frame: pd.DataFrame, output_dir: Path, device: str, feature_path: Path) -> None:
    selection_path = output_dir / "cv_selection.json"
    cv_results_path = output_dir / "cv_results.csv"
    freeze_path = output_dir / "model_freeze.json"
    if not selection_path.is_file() or not cv_results_path.is_file():
        raise FileNotFoundError("Run stage cv first")
    if freeze_path.exists():
        raise FileExistsError("model_freeze.json already exists; do not silently refit")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cv_results = pd.read_csv(cv_results_path)
    development = frame.loc[frame["analysis_split"] == "development"].reset_index(drop=True)
    calibration = frame.loc[frame["analysis_split"] == "calibration"].reset_index(drop=True)
    explanation = frame.loc[frame["analysis_split"] == "tuning"].reset_index(drop=True)
    candidates = [selection["primary_candidate"], selection["alternative_candidate"]]
    artifact_records = []

    for candidate in candidates:
        print(f"FIT_START candidate={candidate}", flush=True)
        started = time.perf_counter()
        artifact = fit_one_candidate(candidate, development, calibration, cv_results, device)
        artifact_path = output_dir / f"model_{candidate}.joblib"
        joblib.dump(artifact, artifact_path, compress=3)
        uncalibrated_artifact = dict(artifact)
        probability = predict_artifact(artifact, explanation, calibrated=True)
        probability_uncalibrated = predict_artifact(
            uncalibrated_artifact, explanation, calibrated=False
        )
        y_explanation = explanation["label_admitted"].to_numpy(dtype=np.int8)
        record = {
            "candidate": candidate,
            "family": artifact["family"],
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "artifact_bytes": artifact_path.stat().st_size,
            "fit_seconds": time.perf_counter() - started,
            "calibration_metrics_uncalibrated": artifact[
                "calibration_metrics_uncalibrated"
            ],
            "calibration_metrics_calibrated": artifact["calibration_metrics_calibrated"],
            "explanation_metrics_uncalibrated": metrics(
                y_explanation, probability_uncalibrated
            ),
            "explanation_metrics_calibrated": metrics(y_explanation, probability),
        }
        artifact_records.append(record)
        print(
            f"FIT_DONE candidate={candidate} explanation_brier="
            f"{record['explanation_metrics_calibrated']['brier']:.6f}",
            flush=True,
        )

    freeze = {
        "created_at_unix": time.time(),
        "feature_file": str(feature_path),
        "feature_file_sha256": sha256_file(feature_path),
        "feature_file_total_rows_from_parquet_metadata": int(
            pq.ParquetFile(feature_path).metadata.num_rows
        ),
        "pretest_rows_loaded": int(len(frame)),
        "input_partitions_loaded": sorted(frame["analysis_split"].unique().tolist()),
        "raw_feature_count": len(RAW_FEATURE_COLUMNS),
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "domain_map": DOMAIN_MAP,
        "selection": selection,
        "artifacts": artifact_records,
        "preprocessing": {
            "continuous": "development-median imputation then z standardization",
            "counts": "zero imputation, log1p, then z standardization",
            "binary": "constant-zero imputation",
            "arrival_transport": (
                "explicit missing level then development-fitted one-hot; unknown ignored"
            ),
        },
        "decision_thresholds": {
            "values": REPORT_THRESHOLDS,
            "rule": "fixed probability cutoffs for reporting; not optimized on the locked test",
        },
        "calibration": "Platt logistic recalibration on logit probability using calibration split",
        "libraries": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "locked_test_accessed": False,
    }
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MODEL_FREEZE sha256={sha256_file(freeze_path)}", flush=True)


def cluster_bootstrap_importance(
    subject_id: np.ndarray,
    per_row_delta: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    domains = per_row_delta.shape[1]
    frame = pd.DataFrame(per_row_delta)
    frame.insert(0, "subject_id", subject_id)
    sums = frame.groupby("subject_id", sort=False).sum().to_numpy(dtype=np.float64)
    counts = frame.groupby("subject_id", sort=False).size().to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty((repeats, domains), dtype=np.float64)
    n_subjects = len(counts)
    for index in range(repeats):
        sampled = rng.integers(0, n_subjects, size=n_subjects)
        output[index] = sums[sampled].sum(axis=0) / counts[sampled].sum()
    return output


def weights_for_artifact(
    artifact: dict[str, Any],
    explanation: pd.DataFrame,
    repeats: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = explanation["label_admitted"].to_numpy(dtype=np.int8)
    base_probability = predict_artifact(artifact, explanation, calibrated=True)
    base_loss = (y - base_probability) ** 2
    domain_names = list(DOMAIN_MAP)
    row_delta = np.zeros((len(explanation), len(domain_names)), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for domain_index, domain in enumerate(domain_names):
        columns = DOMAIN_MAP[domain]
        accumulated = np.zeros(len(explanation), dtype=np.float64)
        started = time.perf_counter()
        for repeat in range(repeats):
            permutation = rng.permutation(len(explanation))
            permuted = explanation[RAW_FEATURE_COLUMNS].copy()
            permuted.loc[:, columns] = explanation.iloc[permutation][columns].to_numpy()
            probability = predict_artifact(artifact, permuted, calibrated=True)
            accumulated += (y - probability) ** 2 - base_loss
        row_delta[:, domain_index] = accumulated / repeats
        print(
            f"WEIGHT_DOMAIN candidate={artifact['candidate']} domain={domain} "
            f"importance={row_delta[:, domain_index].mean():.8f} "
            f"seconds={time.perf_counter()-started:.1f}",
            flush=True,
        )

    bootstrap = cluster_bootstrap_importance(
        explanation["subject_id"].to_numpy(), row_delta, bootstrap_repeats, seed + 1
    )
    point = row_delta.mean(axis=0)
    positive = np.maximum(point, 0)
    if positive.sum() <= 1e-12:
        normalized = np.full_like(positive, np.nan)
    else:
        normalized = positive / positive.sum()
    boot_positive = np.maximum(bootstrap, 0)
    boot_denominator = boot_positive.sum(axis=1, keepdims=True)
    boot_normalized = np.divide(
        boot_positive,
        boot_denominator,
        out=np.full_like(boot_positive, np.nan),
        where=boot_denominator > 1e-12,
    )
    result = []
    for index, domain in enumerate(domain_names):
        result.append(
            {
                "candidate": artifact["candidate"],
                "domain": domain,
                "raw_importance": float(point[index]),
                "raw_ci_low": float(np.quantile(bootstrap[:, index], 0.025)),
                "raw_ci_high": float(np.quantile(bootstrap[:, index], 0.975)),
                "normalized_weight": float(normalized[index]),
                "weight_ci_low": float(np.nanquantile(boot_normalized[:, index], 0.025)),
                "weight_ci_high": float(np.nanquantile(boot_normalized[:, index], 0.975)),
                "permutation_repeats": repeats,
                "bootstrap_repeats": bootstrap_repeats,
            }
        )
    patient_delta = pd.DataFrame(row_delta, columns=domain_names)
    patient_delta.insert(0, "subject_id", explanation["subject_id"].to_numpy())
    patient_delta = patient_delta.groupby("subject_id", as_index=False).agg(
        {**{domain: "sum" for domain in domain_names}}
    )
    patient_counts = explanation.groupby("subject_id", as_index=False).size().rename(
        columns={"size": "encounters"}
    )
    patient_delta = patient_delta.merge(patient_counts, on="subject_id", validate="one_to_one")
    return pd.DataFrame(result), patient_delta, pd.DataFrame(
        bootstrap, columns=domain_names
    )


def weights_stage(
    frame: pd.DataFrame,
    output_dir: Path,
    repeats: int,
    bootstrap_repeats: int,
    feature_path: Path,
) -> None:
    freeze_path = output_dir / "model_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Run stage fit first")
    if (output_dir / "grouped_weights.csv").exists():
        raise FileExistsError("grouped_weights.csv already exists; do not silently overwrite")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if sha256_file(feature_path) != freeze["feature_file_sha256"]:
        raise ValueError("Feature file hash differs from the model freeze")
    explanation = frame.loc[frame["analysis_split"] == "tuning"].reset_index(drop=True)
    all_weights = []
    for model_index, record in enumerate(freeze["artifacts"]):
        artifact_path = Path(record["artifact"])
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            raise ValueError(f"Artifact hash mismatch: {artifact_path}")
        artifact = joblib.load(artifact_path)
        model_repeats = repeats if model_index == 0 else max(5, repeats // 4)
        model_bootstrap = bootstrap_repeats if model_index == 0 else max(
            200, bootstrap_repeats // 2
        )
        weights, patient_delta, bootstrap = weights_for_artifact(
            artifact,
            explanation,
            model_repeats,
            model_bootstrap,
            RANDOM_SEED + model_index * 1000,
        )
        all_weights.append(weights)
        patient_delta.to_parquet(
            output_dir / f"private_patient_delta_{artifact['candidate']}.parquet",
            index=False,
            compression="zstd",
        )
        bootstrap.to_parquet(
            output_dir / f"weight_bootstrap_{artifact['candidate']}.parquet",
            index=False,
            compression="zstd",
        )
    combined = pd.concat(all_weights, ignore_index=True)
    combined.to_csv(output_dir / "grouped_weights.csv", index=False)
    summary = {
        "primary_candidate": freeze["selection"]["primary_candidate"],
        "explanation_rows": int(len(explanation)),
        "explanation_subjects": int(explanation["subject_id"].nunique()),
        "method": "joint raw-domain permutation; Brier loss increment; patient-cluster bootstrap",
        "model_freeze_sha256": sha256_file(freeze_path),
        "weights_file_sha256": sha256_file(output_dir / "grouped_weights.csv"),
        "input_partitions_loaded": sorted(frame["analysis_split"].unique().tolist()),
        "locked_test_accessed": False,
    }
    (output_dir / "weight_freeze.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def threshold_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    tn = int(np.sum((~prediction) & (y == 0)))
    fn = int(np.sum((~prediction) & (y == 1)))
    return {
        "threshold": float(threshold),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def bootstrap_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    subject_id: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    patient_frame = pd.DataFrame(
        {"subject_id": subject_id, "y": y, "probability": probability}
    )
    groups = [group.index.to_numpy() for _, group in patient_frame.groupby("subject_id")]
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        value = metrics(y[indices], probability[indices])
        for threshold in REPORT_THRESHOLDS:
            threshold_result = threshold_metrics(
                y[indices], probability[indices], threshold
            )
            prefix = f"threshold_{threshold:.2f}_"
            for metric_name in ["sensitivity", "specificity", "ppv", "npv"]:
                value[f"{prefix}{metric_name}"] = threshold_result[metric_name]
        value["repeat"] = repeat
        rows.append(value)
    return pd.DataFrame(rows)


def test_stage(
    frame: pd.DataFrame,
    output_dir: Path,
    bootstrap_repeats: int,
    confirmed: bool,
) -> None:
    if not confirmed:
        raise PermissionError("Pass --confirm-locked-test only after model_freeze and weight_freeze exist")
    freeze_path = output_dir / "model_freeze.json"
    weight_freeze_path = output_dir / "weight_freeze.json"
    result_path = output_dir / "locked_test_metrics.json"
    if not freeze_path.is_file() or not weight_freeze_path.is_file():
        raise FileNotFoundError("Both model_freeze.json and weight_freeze.json are required")
    if result_path.exists():
        raise FileExistsError("Locked test metrics already exist; a second test run is prohibited")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    test = frame.loc[
        frame["analysis_split"] == "patient_disjoint_internal_test"
    ].reset_index(drop=True)
    y = test["label_admitted"].to_numpy(dtype=np.int8)
    results = []
    for record in freeze["artifacts"]:
        artifact_path = Path(record["artifact"])
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            raise ValueError(f"Artifact hash mismatch: {artifact_path}")
        artifact = joblib.load(artifact_path)
        probability = predict_artifact(artifact, test, calibrated=True)
        point = metrics(y, probability)
        boot = bootstrap_metrics(
            y,
            probability,
            test["subject_id"].to_numpy(),
            bootstrap_repeats,
            RANDOM_SEED,
        )
        ci = {
            metric: {
                "low": float(boot[metric].quantile(0.025)),
                "high": float(boot[metric].quantile(0.975)),
            }
            for metric in ["auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]
        }
        thresholds = []
        for threshold in REPORT_THRESHOLDS:
            threshold_point = threshold_metrics(y, probability, threshold)
            prefix = f"threshold_{threshold:.2f}_"
            threshold_point["ci95"] = {
                metric_name: {
                    "low": float(boot[f"{prefix}{metric_name}"].quantile(0.025)),
                    "high": float(boot[f"{prefix}{metric_name}"].quantile(0.975)),
                }
                for metric_name in ["sensitivity", "specificity", "ppv", "npv"]
            }
            thresholds.append(threshold_point)
        results.append(
            {
                "candidate": artifact["candidate"],
                "family": artifact["family"],
                "point": point,
                "ci95": ci,
                "thresholds": thresholds,
            }
        )
        boot.to_parquet(
            output_dir / f"locked_test_bootstrap_{artifact['candidate']}.parquet",
            index=False,
            compression="zstd",
        )
    output = {
        "test_rows": int(len(test)),
        "test_subjects": int(test["subject_id"].nunique()),
        "test_events": int(y.sum()),
        "bootstrap_repeats": bootstrap_repeats,
        "results": results,
        "single_locked_test_run": True,
    }
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


def validate_test_gate(feature_path: Path, output_dir: Path, confirmed: bool) -> None:
    """Validate every frozen dependency before any locked-test row is read."""
    if not confirmed:
        raise PermissionError(
            "Pass --confirm-locked-test only after model_freeze and weight_freeze exist"
        )
    freeze_path = output_dir / "model_freeze.json"
    weight_freeze_path = output_dir / "weight_freeze.json"
    weights_path = output_dir / "grouped_weights.csv"
    result_path = output_dir / "locked_test_metrics.json"
    if not freeze_path.is_file() or not weight_freeze_path.is_file():
        raise FileNotFoundError("Both model_freeze.json and weight_freeze.json are required")
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if result_path.exists():
        raise FileExistsError("Locked test metrics already exist; a second test run is prohibited")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    weight_freeze = json.loads(weight_freeze_path.read_text(encoding="utf-8"))
    if freeze.get("locked_test_accessed") is not False:
        raise ValueError("Model freeze does not certify an untouched locked test")
    if weight_freeze.get("locked_test_accessed") is not False:
        raise ValueError("Weight freeze does not certify an untouched locked test")
    if sha256_file(feature_path) != freeze["feature_file_sha256"]:
        raise ValueError("Feature file hash differs from the model freeze")
    if sha256_file(freeze_path) != weight_freeze["model_freeze_sha256"]:
        raise ValueError("Model freeze hash differs from the weight freeze")
    if sha256_file(weights_path) != weight_freeze["weights_file_sha256"]:
        raise ValueError("Grouped weights hash differs from the weight freeze")


def main() -> None:
    args = parse_args()
    if not args.features.is_file():
        raise FileNotFoundError(args.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "test":
        validate_test_gate(args.features, args.output_dir, args.confirm_locked_test)
    frame = load_stage_frame(args.features, args.stage)
    if args.stage == "cv":
        cv_stage(frame, args.output_dir, args.n_splits, args.xgb_device)
    elif args.stage == "fit":
        fit_stage(frame, args.output_dir, args.xgb_device, args.features)
    elif args.stage == "weights":
        weights_stage(
            frame,
            args.output_dir,
            args.permutation_repeats,
            args.bootstrap_repeats,
            args.features,
        )
    elif args.stage == "test":
        test_stage(
            frame,
            args.output_dir,
            args.bootstrap_repeats,
            args.confirm_locked_test,
        )


if __name__ == "__main__":
    main()
