"""One-command replay of the public aggregate decision-layer analyses."""

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
        str(root / "code" / "replay_aggregate_decision_layer.py"),
        "--core-dir",
        str(root / "code"),
        "--panel-value-weights",
        str(root / "data" / "panel_value_weights.csv"),
        "--aggregate-utility-cells",
        str(root / "data" / "aggregate_utility_cells.csv"),
        "--patient-evidence-weights",
        str(root / "data" / "patient_evidence_weight_specifications.csv"),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
