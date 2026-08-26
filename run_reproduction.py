"""One-command replay of the public v5.9.6 decision-layer analyses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repository>/results_rerun",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve() if args.output_dir else root / "results_rerun"
    command = [
        sys.executable,
        "-B",
        str(root / "code" / "replay_decision_layer.py"),
        "--core-dir",
        str(root / "code"),
        "--panel-a-importance",
        str(root / "data" / "panel_A_criterion_weak_orders.csv"),
        "--panel-a-ratings",
        str(root / "data" / "panel_A_linguistic_ratings.csv"),
        "--panel-b-importance",
        str(root / "data" / "panel_B_criterion_weak_orders.csv"),
        "--panel-b-ratings",
        str(root / "data" / "panel_B_linguistic_ratings.csv"),
        "--grouped-weights",
        str(root / "data" / "grouped_model_permutation_weights.csv"),
        "--group-shap-weights",
        str(root / "data" / "grouped_treeshap_weights.csv"),
        "--first-encounter-weights",
        str(root / "data" / "first_encounter_explanation_ddcdw.csv"),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
