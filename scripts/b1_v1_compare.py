"""One pre-registered pressure ablation in V1, compared with frozen B0 artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from b0_v1_baseline import OUT as B0_OUT, ROOT, main as run_v1, score_samples
from s0_validation_check import digest
from validation_core import require

OUT = ROOT / "data/validation/2026-10-19_B1_V1"
REPORT = ROOT / "reports/2026-10-19_21_V1_기압_포함과_제외_비교.md"


def compare_results(b0_scores, b1_scores):
    old = {(s["sample"], s["period"]): s for s in b0_scores}
    new = {(s["sample"], s["period"]): s for s in b1_scores}
    require(set(old) == set(new), "Score sample keys differ")
    rows = []
    for key, a in old.items():
        b = new[key]
        for field in ("times", "first_ts", "last_ts", "smp_sum", "zero_price_times", "price_k_sum", "R_local"):
            require(a[field] == b[field], f"Fixed evaluation sample differs: {key} {field}")
        wa, wb = a["W_local"], b["W_local"]
        rows.append(dict(sample=key[0], period=key[1], times=a["times"], price_k_sum=a["price_k_sum"],
                         B0_W=wa, B1_W=wb, delta_W=wb-wa,
                         change_pct=(wb-wa)/wa*100 if wa else None,
                         B0_weighted_group_mae_kwh=a["weighted_group_mae_kwh"],
                         B1_weighted_group_mae_kwh=b["weighted_group_mae_kwh"],
                         delta_S_local=b["S_local"]-a["S_local"]))
    annual = next(r for r in rows if r["sample"] == "fixed_148" and r["period"] == "all")
    quarters = [r for r in rows if r["sample"] == "fixed_148" and r["period"].startswith("Q")]
    require(len(quarters) == 4, "Four V1 quarters required")
    unstable = [r["period"] for r in quarters if
                (r["B0_W"] > 0 and r["B1_W"] > r["B0_W"] * 1.05
                 and not np.isclose(r["B1_W"], r["B0_W"] * 1.05, rtol=1e-12, atol=0))
                or (r["B0_W"] == 0 and r["B1_W"] > 0)]
    improved = annual["B1_W"] < annual["B0_W"] and not np.isclose(
        annual["B0_W"], annual["B1_W"], rtol=1e-12, atol=0)
    selected = "B1" if improved and not unstable else "B0"
    reason = ("annual_improved_and_quarter_guard_passed" if selected == "B1" else
              "annual_not_improved_or_tied" if not improved else "quarter_guard_failed")
    return dict(selected_for_V2=selected, reason=reason, annual_improved=bool(improved),
                unstable_quarters=unstable, rows=rows, threshold_pct=5,
                zero_quarter_policy="If B0 quarter W=0 and B1 W>0, retain B0 and report absolute change.")


def verify_saved_b0(truth, groups, price, fixed_ids, b0):
    """Re-score existing B0 predictions without refitting; inspect serialized outputs."""
    matrices = []
    for label in ("round_10", "round_17", "mean"):
        frame = pd.read_parquet(B0_OUT / f"predictions_{label}.parquet")
        require(len(frame) == len(truth.index) * 253 and not frame.duplicated(["site_id", "ts"]).any(),
                "Saved B0 grid mismatch")
        matrix = frame.pivot(index="ts", columns="site_id", values="kwh").reindex(index=truth.index, columns=truth.columns)
        require(np.isfinite(matrix.to_numpy()).all(), "Saved B0 has missing predictions")
        matrices.append(matrix)
    pd.testing.assert_frame_equal((matrices[0] + matrices[1])/2, matrices[2])
    for fit in b0["fits"]:
        require(digest(B0_OUT / f"model_round_{fit['round']}.txt") == fit["model_sha256"], "Saved B0 model changed")
    current = score_samples(truth, matrices[2], groups, price, fixed_ids)
    old = {(s["sample"], s["period"]): s for s in b0["group_scores"]}
    for item in current:
        previous = old[(item["sample"], item["period"])]
        for field in previous:
            if isinstance(previous[field], float):
                require(np.isclose(item[field], previous[field], rtol=1e-12, atol=1e-5), "Saved B0 score mismatch")
            else:
                require(item[field] == previous[field], "Saved B0 sample mismatch")
    return True


if __name__ == "__main__":
    run_v1("B1_V1", OUT, extra_code=[ROOT / "scripts/b1_v1_compare.py", ROOT / "tests/test_b1_validation.py"])
