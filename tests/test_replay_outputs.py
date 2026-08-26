from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_decision_layer_replay(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "run_reproduction.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    status = json.loads(
        (tmp_path / "full_precision_sensitivity_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "PASS"

    summary = pd.read_csv(tmp_path / "alternative_attribution_sensitivity_summary.csv")
    primary = summary.loc[summary["analysis"] == "primary_ddcdw"].iloc[0]
    assert primary["panel_A_ranking"] == "A5>A1>A4>A2>A3"
    assert primary["panel_B_ranking"] == "A5>A4>A1>A2>A3"
    assert int(primary["shared_necessary_count"]) == 9

    first = summary.loc[
        summary["analysis"] == "first_encounter_explanation_ddcdw"
    ].iloc[0]
    assert first["panel_A_ranking"] == "A1>A5>A2>A4>A3"
    assert first["panel_B_ranking"] == "A5>A1>A4>A2>A3"

    tree_shap = summary.loc[summary["analysis"] == "grouped_treeshap"].iloc[0]
    assert tree_shap["panel_A_ranking"] == "A5>A1>A2>A4>A3"
    assert tree_shap["panel_B_ranking"] == "A5>A1>A4>A2>A3"
    assert int(tree_shap["shared_necessary_count"]) == 7

    utilities = pd.read_csv(tmp_path / "utility_cells_100.csv")
    assert len(utilities) == 100
    assert not utilities.duplicated(["panel", "alternative_id", "criterion_id"]).any()
    assert (utilities["u_lower"] <= utilities["u0"]).all()
    assert (utilities["u0"] <= utilities["u_upper"]).all()
