"""C0: confirm selected B0 on V2 once, preserving the frozen V1 decision."""
from __future__ import annotations

import json

from b0_v1_baseline import ROOT, main as run_baseline
from b1_v1_compare import compare_results
from s0_validation_check import digest
from validation_core import require

OUT = ROOT / "data/validation/2026-10-20_B0_V2"
REPORT = ROOT / "reports/2026-10-20_01_B0_V2_후반확인.md"


def freeze_v1(source_hashes, eda_hash, params, features, runtime):
    locations = {"B0": ROOT / "data/validation/2026-10-19_B0_V1",
                 "B1": ROOT / "data/validation/2026-10-19_B1_V1"}
    summaries = {name: json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                 for name, folder in locations.items()}
    b0, b1 = summaries["B0"], summaries["B1"]
    require(b0["status"] == b1["status"] == "passed", "V1 must be completed")
    choice = compare_results(b0["group_scores"], b1["group_scores"])
    require(choice["selected_for_V2"] == b1["comparison"]["selected_for_V2"] == "B0", "Only selected B0 allowed")
    require(b0["source_sha256"] == b1["source_sha256"] == source_hashes, "V1 source drift")
    require(b0["eda_plan_sha256"] == b1["eda_plan_sha256"] == eda_hash, "EDA drift")
    require(b0["model_parameters"] == b1["model_parameters"] == params, "V1 settings drift")
    require(b0["model_features"] == features and b0["runtime"] == b1["runtime"] == runtime, "Feature/runtime drift")
    hashes = {}
    names = ["summary.json", "model_round_10.txt", "model_round_17.txt",
             "predictions_round_10.parquet", "predictions_round_17.parquet", "predictions_mean.parquet"]
    for label, folder in locations.items():
        for name in names:
            path = folder / name
            actual = digest(path)
            hashes[str(path.relative_to(ROOT))] = actual
            if label == "B0":
                require(actual == b1["comparison"]["b0_artifact_sha256"][name], "Frozen B0 artifact changed")
    return dict(selected="B0", decision=choice, artifact_sha256=hashes,
                V1_refit_count=0, B1_V2_fit_count=0,
                rationale="Selected method and reference baseline are identical; fit B0 once per round, not twice.")


def verify_v1_frozen(frozen):
    for rel, expected in frozen["artifact_sha256"].items():
        require(digest(ROOT / rel) == expected, "V1 artifacts changed during V2")


if __name__ == "__main__":
    run_baseline("B0_V2", OUT, extra_code=[ROOT / "scripts/b0_v2_confirm.py", ROOT / "scripts/b1_v1_compare.py",
                 ROOT / "tests/test_v2_validation.py", ROOT / "tests/test_b1_validation.py"])
