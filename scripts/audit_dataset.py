"""Read-only audit of the supplied OIBC dataset; no model training or submission."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "OIBC_2026_DATA" / "01_participant_dataset"
OUT = ROOT / "데이터_확인"
OUT.mkdir(exist_ok=True)
EXPECTED = {
    "sites.csv": 253, "smp.csv": 8760, "train/smp.csv": 38688,
    "train/generation.parquet": 4392566,
    "train/forecast_wa.parquet": 4710356,
    "train/forecast_wb.parquet": 5621996,
    "train/observed_wa.parquet": 2359974,
    "train/observed_wb.parquet": 2824223,
    "test/forecast_wa.parquet": 1068456,
    "test/forecast_wb.parquet": 1278960,
}
result = {"checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
          "runtime": sys.version, "packages": {"pandas": pd.__version__,
          "numpy": np.__version__, "pyarrow": pyarrow.__version__}, "files": {}}
print("Runtime started:", sys.executable, flush=True)
sites = pd.read_csv(DATA / "sites.csv", encoding="utf-8-sig")
score = pd.read_csv(DATA / "smp.csv", encoding="utf-8-sig")
score_times = pd.DatetimeIndex(pd.to_datetime(score.ts))
train_end = pd.Timestamp("2025-06-01 00:00:00+09:00")
result["sites"] = {"unique": int(sites.site_id.nunique()),
    "ids_exact": set(sites.site_id) == {f"PV{i:03d}" for i in range(1, 254)},
    "capacity_kw": float(sites.capacity_kw.sum()),
    "reference_groups_kw": sites.groupby("current_group").capacity_kw.sum().to_dict(),
    "wa_stations": int(sites.wa_station.nunique()), "wb_stations": int(sites.wb_station.nunique()),
    "nonpositive_capacity": int((sites.capacity_kw <= 0).sum())}
result["scoring_axis"] = {
    "exact_hourly_axis": score_times.equals(pd.date_range("2025-06-01 01:00:00+09:00", periods=8760, freq="h")),
    "k_mismatches": int((score.k.to_numpy() != np.where((score_times.hour >= 11) & (score_times.hour <= 15), 3, 1)).sum()),
    "k_counts": {str(k): int(v) for k,v in score.k.value_counts().items()},
    "negative_smp": int((score.smp < 0).sum()), "zero_smp": int((score.smp == 0).sum())}
missing_records = []
for rel, expected_rows in EXPECTED.items():
    path = DATA / rel
    info = {"exists": path.is_file(), "expected_rows": expected_rows}
    if not path.is_file():
        result["files"][rel] = info
        continue
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    info.update(bytes=path.stat().st_size, sha256=h.hexdigest())
    if path.suffix == ".parquet":
        d = pq.read_table(path).to_pandas()
    else:
        d = pd.read_csv(path, encoding="utf-8-sig")
    info.update(rows=len(d), row_count_matches=len(d) == expected_rows,
                columns=list(d.columns), dtypes={c:str(t) for c,t in d.dtypes.items()},
                nulls={c:int(v) for c,v in d.isna().sum().items() if v})
    keys = [c for c in ("site_id", "wa_station", "wb_station", "round", "ts") if c in d]
    info["keys"] = keys
    info["duplicate_keys"] = int(d.duplicated(keys).sum())
    num = d.select_dtypes(include="number")
    info["infinite_values"] = int(np.isinf(num.to_numpy()).sum())
    info["numeric_ranges"] = {c:{"min":float(num[c].min()),"max":float(num[c].max())} for c in num}
    if "ts" in d:
        d["ts"] = pd.to_datetime(d.ts)
        info.update(ts_min=str(d.ts.min()), ts_max=str(d.ts.max()), timezone=str(d.ts.dt.tz),
                    unique_timestamps=int(d.ts.nunique()), nonhourly_rows=int(((d.ts.dt.minute != 0) | (d.ts.dt.second != 0)).sum()))
        info["outside_split"] = int((d.ts > train_end).sum()) if rel.startswith("train/") else (int((~d.ts.isin(score_times)).sum()) if rel != "sites.csv" else 0)
    if "site_id" in d:
        info["unknown_site_ids"] = sorted(set(d.site_id) - set(sites.site_id))
        info["unique_sites"] = int(d.site_id.nunique())
    station = next((c for c in ("wa_station", "wb_station") if c in d), None)
    if station and rel.endswith(".parquet"):
        info["unknown_stations"] = sorted(set(d[station]) - set(sites[station]))
        info["missing_mapped_stations"] = sorted(set(sites[station]) - set(d[station]))
    if "round" in d:
        info["round_counts"] = {str(k):int(v) for k,v in d["round"].value_counts().items()}
        info["unexpected_rounds"] = sorted(set(d["round"]) - {10,17})
    if rel.startswith("test/"):
        expected_index = pd.MultiIndex.from_product([sorted(sites[station].unique()), [10,17], score_times], names=[station,"round","ts"])
        actual = pd.MultiIndex.from_frame(d[[station,"round","ts"]])
        missing = expected_index.difference(actual)
        mf = missing.to_frame(index=False)
        info.update(expected_grid_rows=len(expected_index),missing_grid_rows=len(missing),
                    missing_grid_unique_times=int(mf.ts.nunique()),
                    station_times_missing_both_rounds=int((mf.groupby([station,"ts"]).size() == 2).sum()))
        for row in mf.to_dict("records"):
            missing_records.append({"file":rel, "station":row[station], "round":row["round"], "ts":str(row["ts"])})
    if rel == "train/generation.parquet":
        counts = d.groupby("site_id").size()
        info.update(negative_energy=int((d.energy < 0).sum()),
                    per_site_rows_min=int(counts.min()),per_site_rows_max=int(counts.max()),
                    absent_pairs_on_full_training_axis=int(len(sites)*38688-len(d)))
        d.groupby("site_id").agg(rows=("ts","size"),first_ts=("ts","min"),last_ts=("ts","max")).to_csv(OUT/"발전소별_학습기간.csv",encoding="utf-8-sig")
    result["files"][rel] = info
    print(rel, len(d), "rows; duplicates", info["duplicate_keys"], "nulls", info["nulls"], flush=True)
    del d, num
result["test_actual_or_observed_files"] = [str(p.relative_to(DATA)) for p in (DATA/"test").rglob("*") if p.is_file() and ("generation" in p.name or "observed" in p.name)]
result["auxiliary"] = {name:(DATA/name).is_file() for name in ("README.md","baseline.py","scoring.py","OIBC2026-문제정의서.md")}
(OUT/"데이터_검증결과.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
pd.DataFrame(missing_records,columns=["file","station","round","ts"]).to_csv(OUT/"평가예보_결손목록.csv",index=False,encoding="utf-8-sig")
print("Audit completed:", OUT, flush=True)
