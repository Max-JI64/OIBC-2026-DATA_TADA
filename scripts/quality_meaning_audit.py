"""Read-only data meaning and key-quality checks before relationship EDA.

No relationships between measured variables, model training, or predictions.
Hypothetical changes are used only to count structural effects in memory.
"""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from basic_shape_eda import ROOT, SOURCE, FILES, digest, fmt, md_table, log

OUT = ROOT / "data" / "eda" / "2026-10-19_quality_meaning"
REPORT = ROOT / "reports" / "2026-10-19_04_값의_의미와_데이터_품질_점검.md"
NEAR_ZERO = []
EXTREMES = []
CONCENTRATION = []
RESULT = {}
RULES = {
    "wa_wind_speed": ("풍속 >50 m/s", lambda a: a > 50, "안내문 관찰 조건"),
    "wb_wind_speed": ("풍속 >200 km/h", lambda a: a > 200, "안내문 관찰 조건"),
    "wb_cloud": ("운량 <0 또는 >100%", lambda a: (a < 0) | (a > 100), "정의 범위 밖"),
    "wb_uv_idx": ("UV >15", lambda a: a > 15, "안내문 관찰 조건"),
    "wb_vis": ("시정 >50 km", lambda a: a > 50, "안내문 관찰 조건"),
    "wb_pressure": ("기압 <900 hPa", lambda a: a < 900, "안내문 관찰 조건"),
    "wa_humidity": ("상대습도 <0 또는 >100%", lambda a: (a < 0) | (a > 100), "104까지 배포 README에 명시"),
    "energy": ("발전량 <0 kWh", lambda a: a < 0, "음수 기록"),
}


def long_runs(d, col, target):
    keys = [c for c in ("wb_station", "round") if c in d]
    z = d[keys + ["ts", col]].sort_values(keys + ["ts"]).reset_index(drop=True)
    hit = np.isclose(z[col].to_numpy(), target, rtol=0, atol=1e-9)
    same_key = np.ones(len(z), dtype=bool)
    same_key[0] = False
    for key in keys:
        same_key[1:] &= z[key].to_numpy()[1:] == z[key].to_numpy()[:-1]
    hourly = z.ts.diff().eq(pd.Timedelta(hours=1)).to_numpy()
    consecutive = hit & np.r_[False, hit[:-1]] & same_key & hourly
    groups = np.cumsum(~consecutive)
    lengths = pd.Series(groups[hit]).value_counts().to_numpy()
    return {"run_count": len(lengths), "longest_hourly_run": int(lengths.max()) if len(lengths) else 0,
            "median_run_hours": float(np.median(lengths)) if len(lengths) else 0.0}


def inspect_concentration(d, rel):
    for col, target, original, unit in [("wb_vis", 16.0934, 10, "mile"),
                                        ("wb_ceiling", 30000 * .3048, 30000, "ft")]:
        a = d[col].to_numpy(dtype=float)
        hit = np.isclose(a, target, rtol=0, atol=1e-9)
        finite = np.isfinite(a)
        mode = d[col].value_counts().iloc[:1]
        item = {"file": rel, "column": col, "target": target,
                "original_unit_value": original, "original_unit": unit,
                "standard_conversion_value": 10*1.609344 if col == "wb_vis" else 30000*.3048,
                "data_target_minus_standard_conversion": target-(10*1.609344 if col == "wb_vis" else 30000*.3048),
                "match_tolerance": 1e-9, "n": int(finite.sum()), "target_count": int(hit.sum()),
                "target_share": float(hit.sum()/finite.sum()),
                "above_target_count": int((finite & (a > target + 1e-9)).sum()),
                "below_target_count": int((finite & (a < target - 1e-9)).sum()),
                "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
                "file_mode": float(mode.index[0]), "file_mode_share": float(mode.iloc[0]/finite.sum()),
                "target_distinct_stations": int(d.loc[hit, "wb_station"].nunique()),
                **long_runs(d, col, target)}
        assert item["target_count"] + item["above_target_count"] + item["below_target_count"] == item["n"]
        CONCENTRATION.append(item)


def inspect_time(d):
    keys = ["wb_station", "round", "ts"]
    mask = d.ts.ne(d.ts.dt.floor("h"))
    suspect = d.loc[mask, keys].copy()
    source_index = pd.MultiIndex.from_frame(d.loc[~mask, keys])
    raw_suspect_index = pd.MultiIndex.from_frame(suspect)
    candidate = suspect.copy()
    candidate["ts"] = candidate.ts.dt.round("h")
    candidate_index = pd.MultiIndex.from_frame(candidate)
    collisions_existing = int(candidate_index.isin(source_index).sum())
    collisions_within = int(candidate_index.duplicated().sum())
    axis = pd.date_range(d.ts.min().floor("h"), d.ts.max().ceil("h"), freq="h")
    all_expected = len(d.wb_station.unique()) * 2 * len(axis)
    original_on_grid = len(source_index.unique())
    proposed_on_grid = len(source_index.union(candidate_index))
    delta = (candidate.ts - suspect.ts).dt.total_seconds()
    candidate_values = d.loc[mask].copy()
    candidate_values["ts"] = candidate_values.ts.dt.round("h")
    target_positions = source_index.get_indexer(candidate_index)
    matched_original_rows = np.flatnonzero(~mask.to_numpy())[target_positions[target_positions >= 0]]
    comparison = candidate_values.merge(d.iloc[matched_original_rows], on=keys, how="inner", suffixes=("_shifted", "_original"), validate="many_to_one")
    measured = [c for c in d.select_dtypes(include="number") if c not in keys]
    equal = np.ones(len(comparison), dtype=bool)
    changed_counts = {}
    for col in measured:
        matched = np.isclose(comparison[col+"_shifted"], comparison[col+"_original"], atol=0, rtol=0, equal_nan=True)
        equal &= matched
        if not matched.all():
            changed_counts[col] = int((~matched).sum())
    RESULT["time"] = {"off_hour_rows": len(suspect), "distinct_times": int(suspect.ts.nunique()),
        "stations": int(suspect.wb_station.nunique()), "round_counts": {str(k): int(v) for k, v in suspect["round"].value_counts().items()},
        "delta_seconds_counts": {str(k): int(v) for k, v in delta.value_counts().sort_index().items()},
        "candidate_collisions_with_original_keys": collisions_existing,
        "candidate_internal_duplicates": collisions_within,
        "collision_payload_all_equal": int(equal.sum()),
        "collision_payload_conflicting": int((~equal).sum()),
        "collision_conflicting_columns": changed_counts,
        "expected_hourly_grid": all_expected,
        "original_absent_hourly_keys": all_expected - original_on_grid,
        "hypothetical_absent_hourly_keys": all_expected - proposed_on_grid,
        "raw_off_grid_keys_absent_from_hourly_axis": len(raw_suspect_index),
        "correction_applied": False}
    assert delta.ne(0).all() and mask.sum() == 92
    suspect.assign(candidate_ts=candidate.ts, shift_seconds=delta).to_csv(OUT / "off_hour_candidates.csv", index=False, encoding="utf-8-sig")


def inspect_duplicates(d):
    keys = ["wa_station", "ts"]
    mask = d.duplicated(keys, keep=False)
    rows = d.loc[mask].copy()
    rows.to_csv(OUT / "conflicting_observation_rows.csv", index=False, encoding="utf-8-sig")
    numeric = list(d.select_dtypes(include="number"))
    differences = []
    per_col = {c: {"conflicting_keys": 0, "max_within_key_spread": 0.0} for c in numeric}
    for _, group in rows.groupby(keys, observed=True):
        changed = [c for c in numeric if group[c].nunique(dropna=False) > 1]
        differences.append({"rows": len(group), "different_columns": changed})
        for col in changed:
            spread = float(group[col].max() - group[col].min())
            per_col[col]["conflicting_keys"] += 1
            per_col[col]["max_within_key_spread"] = max(spread, per_col[col]["max_within_key_spread"])
    # Key-only left joins measure row expansion. No measured-variable relationships.
    unique_keys = d[keys].drop_duplicates()
    expanded = unique_keys.merge(d[keys], on=keys, how="left")
    first = d.drop_duplicates(keys, keep="first")
    last = d.drop_duplicates(keys, keep="last")
    unresolved = d.drop_duplicates(keys, keep=False)
    a, b = first.set_index(keys).loc[pd.MultiIndex.from_frame(rows[keys].drop_duplicates()), numeric], last.set_index(keys).loc[pd.MultiIndex.from_frame(rows[keys].drop_duplicates()), numeric]
    RESULT["duplicates"] = {"conflicting_keys": len(differences), "participating_rows": len(rows),
        "excess_rows": len(d) - len(unique_keys), "identical_full_row_duplicates": int(rows.duplicated().sum()),
        "differing_columns": {k: v for k, v in per_col.items() if v["conflicting_keys"]},
        "key_only_left_join_rows_before": len(unique_keys), "key_only_left_join_rows_after": len(expanded),
        "candidate_first_rows": len(first), "candidate_last_rows": len(last),
        "candidate_excluding_conflicts_rows": len(unresolved),
        "first_last_max_difference": {c: float((a[c]-b[c]).abs().max()) for c in numeric},
        "duplicate_key_numeric_cells": len(differences)*len(numeric),
        "conflicting_numeric_cells": sum(v["conflicting_keys"] for v in per_col.values()),
        "consistent_numeric_cells": len(differences)*len(numeric)-sum(v["conflicting_keys"] for v in per_col.values()),
        "ingestion_or_revision_order_column_available": False, "selection_applied": False}
    assert len(differences) == 6 and len(expanded) - len(unique_keys) == 6
    assert not first.duplicated(keys).any() and not last.duplicated(keys).any()


def inspect_generation(d):
    axis = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(SOURCE / "train/smp.csv", encoding="utf-8-sig").ts))
    records = d.groupby("site_id").agg(first=("ts", "min"), last=("ts", "max"), n=("ts", "size"))
    before = ((records["first"] - axis.min()).dt.total_seconds()/3600).sum()
    after = ((axis.max() - records["last"]).dt.total_seconds()/3600).sum()
    span = ((records["last"]-records["first"]).dt.total_seconds()/3600 + 1)
    inside = int((span-records.n).sum())
    a = d.energy.to_numpy()
    RESULT["generation"] = {"rows": len(d), "exact_zero": int((a == 0).sum()),
        "value_nulls": int(d.energy.isna().sum()), "negative": int((a < 0).sum()),
        "negative_min": float(a[a < 0].min()), "negative_max": float(a[a < 0].max()),
        "negative_sum": float(a[a < 0].sum()),
        "absent_full_grid": int(len(axis)*len(records)-len(d)),
        "before_first_record": int(before), "after_last_record": int(after),
        "within_record_span_absent": inside, "gap_reason_identified": False}
    assert RESULT["generation"]["absent_full_grid"] == int(before+after+inside)


def inspect_values(d, rel):
    for col in d.select_dtypes(include="number"):
        if col == "round":
            continue
        a = d[col].to_numpy(dtype=float, na_value=np.nan)
        a = a[np.isfinite(a)]
        item = {"file": rel, "column": col, "n": len(a), "exact_zero": int((a == 0).sum())}
        for threshold in [1e-12, 1e-9, 1e-6]:
            near = (a != 0) & (np.abs(a) <= threshold)
            item[f"nonzero_abs_le_{threshold:g}"] = int(near.sum())
        tight = (a != 0) & (np.abs(a) <= 1e-12)
        if tight.any():
            item["tight_nonzero_min"] = float(a[tight].min())
            item["tight_nonzero_max"] = float(a[tight].max())
        NEAR_ZERO.append(item)
        if col not in RULES:
            continue
        label, rule, status = RULES[col]
        bad = rule(a)
        if not bad.any():
            continue
        kept = a[~bad]
        top = pd.Series(a[bad]).value_counts().head(5)
        EXTREMES.append({"file": rel, "column": col, "condition": label, "status": status,
                         "n": len(a), "count": int(bad.sum()), "share": float(bad.mean()),
                         "raw_mean": float(a.mean()), "marked_excluded_mean": float(kept.mean()),
                         "raw_std": float(a.std()), "marked_excluded_std": float(kept.std()),
                         "raw_min": float(a.min()), "raw_max": float(a.max()),
                         "excluded_min": float(kept.min()), "excluded_max": float(kept.max()),
                         "above_readme_humidity_104": int((a > 104).sum()) if col == "wa_humidity" else None,
                         "top_flagged_values": [{"value": float(v), "count": int(n)} for v, n in top.items()]})


def report():
    t, dup, gen = RESULT["time"], RESULT["duplicates"], RESULT["generation"]
    near_rows = [r for r in NEAR_ZERO if r["nonzero_abs_le_1e-06"]]
    p = ["# 값의 의미와 데이터 품질 점검", "",
         f"작성: {RESULT['finished_at']}. 상태: 원본 읽기와 품질 점검 실행 완료.", "",
         "## 질문과 목적", "",
         "관계 EDA 전에 특정 값의 의미, 극단값, 0과 기록 부재, 비정시 시각과 중복 키를 확인한다. 확인 가능한 사실과 보류할 해석, 이후 EDA의 취급 원칙을 구분한다. 원본 파일 수정과 전처리 결과 확정은 하지 않았다.", "",
         "## 사용 범위와 수행 방법", "",
         "배포 파일 10개의 해시를 실행 전후와 앞선 EDA 결과에 대조했다. 94개 파일별 수치 열의 정확한 0과 근사 0을 전수 집계했다. B 예보와 실측의 시정, 운고 집중값을 단위 환산값과 비교하고 그보다 큰 값의 존재와 같은 값의 연속 기록을 확인했다. A 실측의 중복 값과 B 예보의 비정시 키는 정리 후보를 메모리에서만 비교했다.", "",
         "같은 변수의 반복 기록과 키 품질만 확인했다. 서로 다른 기상 변수나 발전량의 상관, 집단별 효과, 머신러닝 학습과 예측은 수행하지 않았다. 원본 행과 보정 후보 행은 로컬 data/에만 보관하고 보고서에는 복사하지 않았다.", "",
         "## 1. 시정과 운고의 특정 값은 무엇인가", "",
         "배포 README는 B 예보 기온, 시정과 운고 등이 원래 °F, mile, ft에서 환산됐다고 설명한다. 표준 환산의 10 mile은 16.09344 km다. 데이터에서 집중된 값은 16.0934 km로 0.00004 km, 즉 0.04 m 차이가 있어 근사하게 대응한다. 30,000 ft는 9,144 m와 정확히 대응한다. 빈도 집계는 이론값이 아니라 실제 데이터의 16.0934와 9,144에 절대 허용오차 1e-9를 적용했다.", "",
         md_table(["파일", "열", "확인할 값", "해당 비율", "그보다 큰 행", "파일 최댓값", "실제 최빈값 / 비율", "최장 연속 시간"],
                  [[r["file"], r["column"], fmt(r["target"]), f"{r['target_share']:.2%}", fmt(r["above_target_count"]), fmt(r["max"]), f"{fmt(r['file_mode'])} / {r['file_mode_share']:.2%}", fmt(r["longest_hourly_run"])] for r in CONCENTRATION]), "",
         "학습 예보에서는 확인할 값보다 큰 기록도 있으므로 모든 파일에 공통으로 적용된 엄격한 상한이라는 해석은 맞지 않는다. 평가 예보에서는 두 값이 실제 최댓값과 일치하지만, 관측된 최댓값이라는 사실만으로 수집 상한을 확정할 수 없다. 실측의 최빈값도 따로 확인했으며 예보 집중값과 같은 숫자라고 가정하지 않았다. 일부 원천 기록의 상한, 대표값 또는 상태 코드일 가능성은 남아 있다. 긴 연속 기록도 상태의 지속과 코드 반복을 구분하는 충분한 근거는 아니다.", "",
         "배포 README와 로컬 설명회 자료에서 이 숫자를 결측 코드나 상한으로 정의한 설명은 찾지 못했다. 따라서 '10 mile/30,000 ft 환산값에 집중한다'까지 확인했고, '결측 또는 맑음 코드다'라는 해석은 보류한다. 기상 제공사의 코드북 또는 운영진 설명이 필요한 사항이며 외부에 문의를 보내지는 않았다.", "",
         "**후속 EDA 취급:** 값을 유지하고 해당 값 여부를 별도 표시한다. 약 60%의 기록을 근거 없이 NaN으로 바꾸지 않는다. 이번 작업에서는 표시 열도 새 데이터셋에 적용하지 않았다.", "",
         "## 2. 극단값은 얼마나 영향을 주는가", "",
         "아래의 '표시값 제외'는 관찰 조건에 해당하는 해당 열의 값만 메모리에서 빼고 계산한 민감도 점검이다. 전체 행이나 원본을 삭제하지 않았다. 평균과 표준편차의 차이는 통계 요약에 미치는 영향이며 모델 성능 차이를 의미하지 않는다. 표준편차는 모집단식 ddof=0이다.", "",
         md_table(["파일", "열", "표시 조건", "건수", "비율", "평균 원본/제외", "표준편차 원본/제외", "조건의 지위"],
                  [[r["file"], r["column"], r["condition"], fmt(r["count"]), f"{r['share']:.5%}", f"{fmt(r['raw_mean'])} / {fmt(r['marked_excluded_mean'])}", f"{fmt(r['raw_std'])} / {fmt(r['marked_excluded_std'])}", r["status"]] for r in EXTREMES]), "",
         "운량의 정의 범위 밖 값, 발전량 음수, 안내문의 풍속/UV/시정/기압 관찰 조건을 서로 같은 확정 오류로 취급하지 않는다. A 예보 상대습도 104까지는 README가 원본 값으로 명시하므로 100% 초과를 새 결측 코드로 해석하지 않는다. 평가 A 예보에는 105인 값도 1행 있어 README에 기재된 상한 104를 넘는다. README 범위가 학습 구간만을 뜻하는지 명시가 충분하지 않아 이 차이도 별도 확인 대상으로 남긴다. 어떤 행을 제거할지는 아직 확정하지 않았다.", "",
         "**후속 EDA 취급:** 정의 범위 밖 값과 관찰 조건 초과를 구분해 표시하고, 해당 열의 요약은 원본 포함/표시값 제외를 함께 확인한다. 0 대체와 일괄 IQR 제거는 하지 않는다. IQR 밖이 많은 정상 규모 차이나 드문 양수도 보존한다.", "",
         "## 3. 정확한 0, 근사 0과 없는 기록", "",
         f"발전 실적 {fmt(gen['rows'])}행에는 값 결측이 {fmt(gen['value_nulls'])}개이고 정확한 0은 {fmt(gen['exact_zero'])}개다. 전체 발전소와 학습 시간축의 없는 조합은 {fmt(gen['absent_full_grid'])}개다. 최초 기록 이전 {fmt(gen['before_first_record'])}개, 마지막 기록 이후 {fmt(gen['after_last_record'])}개, 기록 범위 내부 {fmt(gen['within_record_span_absent'])}개로 구분된다. 이는 위치의 분류이며 실제 원인이나 설비 편입일을 확정한 것이 아니다.", "",
         f"발전량 음수는 {fmt(gen['negative'])}개이고 범위는 {fmt(gen['negative_min'])}~{fmt(gen['negative_max'])} kWh, 합계는 {fmt(gen['negative_sum'])} kWh다. 원인이 확인되지 않아 0으로 바꾸지 않았다.", "",
         "아래는 전체 94개 수치 열 중 비영 값의 절댓값이 1e-6 이하인 기록이 있는 열이다. 각 허용오차 건수는 정확한 0을 제외한 누적 건수다. 단위가 다른 열에 동일한 보정 임계값을 적용하자는 의미는 아니다.", "",
         md_table(["파일", "열", "정확한 0", "0<절댓값<=1e-12", "<=1e-9", "<=1e-6"],
                  [[r["file"], r["column"], fmt(r["exact_zero"]), fmt(r["nonzero_abs_le_1e-12"]), fmt(r["nonzero_abs_le_1e-09"]), fmt(r["nonzero_abs_le_1e-06"])] for r in near_rows]), "",
         "**후속 EDA 취급:** 0은 0, 없는 키는 기록 부재, NaN은 값 결측으로 따로 유지한다. 미세한 비영 값은 원래 수치로 보존하고 0 비교의 허용오차만 열별로 검토한다. 반올림이나 전 구간 0 채우기는 적용하지 않았다.", "",
         "## 4. 비정시 기록의 보정 후보는 키를 충돌시키는가", "",
         f"학습 B 예보에 비정시 {fmt(t['off_hour_rows'])}행, {fmt(t['distinct_times'])}개 시각과 {fmt(t['stations'])}개 지점이 있다. 모두 round=17이다. 가장 가까운 정시로 옮기는 후보의 이동량과 건수는 {json.dumps(t['delta_seconds_counts'], ensure_ascii=False)}초다.", "",
         f"그 후보가 기존 정시의 같은 지점/라운드/시각 키와 충돌하는 행은 {fmt(t['candidate_collisions_with_original_keys'])}개이고, 후보끼리 같은 키가 되는 추가 중복은 {fmt(t['candidate_internal_duplicates'])}개다. 완전한 정시 격자의 없는 키는 원본 {fmt(t['original_absent_hourly_keys'])}개에서 가상 보정 후 {fmt(t['hypothetical_absent_hourly_keys'])}개가 된다. 원본에서는 비정시 92행이 별도로 정시 격자 밖에 있다.", "",
         f"충돌 {fmt(t['candidate_collisions_with_original_keys'])}행 중 수치 값이 모두 같은 경우는 {fmt(t['collision_payload_all_equal'])}행, 수치 값이 다른 경우는 {fmt(t['collision_payload_conflicting'])}행이다. 키 충돌이 실제로 발생하므로 일괄 반올림은 채택할 수 없다. 충돌하지 않는 후보도 그 정시가 실제로 의도된 대상 시각이라는 근거는 없어 보정은 확정하지 않았다.", "",
         "**후속 EDA 취급:** 정확한 시각 조인을 기준으로 하고 비정시 행은 별도 표시한다. 해당 행을 정시로 옮기는 후보와 보류하는 후보를 구분하며, 값을 자동으로 반올림해 합치지 않는다. 평가 구간 예보에는 비정시가 없어 현재 공식 제출 시간축을 바꿀 근거는 없다.", "",
         "## 5. A 실측 중복 키에서 무엇이 다른가", "",
         f"충돌 키 {dup['conflicting_keys']}개에 {dup['participating_rows']}행이 있고 완전히 같은 추가 행은 {dup['identical_full_row_duplicates']}개다. 고유 키만 둔 왼쪽 표 {fmt(dup['key_only_left_join_rows_before'])}행에 원본 키를 조인하면 {fmt(dup['key_only_left_join_rows_after'])}행으로 늘어난다. 이는 키만 사용한 행 수 점검이며 측정값 사이 관계 분석이 아니다.", "",
         md_table(["값이 다른 열", "충돌 키 수", "같은 키 내부 최대 차이"], [[c, fmt(v["conflicting_keys"]), fmt(v["max_within_key_spread"])] for c, v in dup["differing_columns"].items()]), "",
         f"첫 행 또는 마지막 행을 택하면 각각 {fmt(dup['candidate_first_rows'])}행과 {fmt(dup['candidate_last_rows'])}행이 된다. 충돌 키의 행을 모두 제외한 실측 표는 {fmt(dup['candidate_excluding_conflicts_rows'])}행이다. 이 후보들은 메모리에서만 비교했으며 저장된 전처리 결과는 없다.", "",
         f"수집 또는 수정 순서를 알려 주는 열이 없어 첫 행과 마지막 행 중 더 정확한 쪽을 고를 근거는 없다. 중복 6키의 수치 셀 {dup['duplicate_key_numeric_cells']}개 중 {dup['conflicting_numeric_cells']}개는 값이 충돌하고 {dup['consistent_numeric_cells']}개는 두 행에서 같은 값이다. 따라서 일치하는 열까지 모두 버릴 필요도 없다. 풍향에는 원형 값 특성도 있어 모든 열을 기계적으로 산술평균하는 방법을 채택하지 않았다.", "",
         "**후속 EDA 취급:** 실측 조인 키의 유일성을 먼저 검사한다. 충돌하는 셀만 미확인으로 표시하고 일치하는 셀은 보존하는 방법을 우선 후보로 남긴다. 예보와 발전 실적의 원래 행을 함께 삭제하지 않는다. 충돌 선택 기준 확정 전에는 중복 실측을 그대로 다대일 조인하지 않는다.", "",
         "## 확인 수준과 남은 판단", "",
         md_table(["항목", "확인한 사실", "보류할 판단"],
                  [["특정 값 집중", "원 단위 환산값 대응, 그보다 큰 값 존재, 연속 반복", "상태 코드/부분 상한/실제 관측의 구분"],
                   ["극단값", "전수 건수와 해당 열의 요약 통계 영향", "발생 원인, 최종 대체 또는 제거 방식"],
                   ["0", "정확한 0과 근사 0, 값 결측과 없는 키 분리", "발전량 0의 물리적 원인, 열별 보정 임계값"],
                   ["비정시", "가상 정시 보정의 키 충돌과 격자 누락 변화", "원래 의도된 대상 시각과 보정 채택"],
                   ["중복", "값 충돌 열, 조인 행 증가와 선택 후보", "어느 실측이 맞는지와 최종 집계 방식"]]), "",
         "추가 의미가 미확인인 항목이 있어도 원본 보존, 주의 표시와 명시적인 조인 조건을 갖춘 뒤 관계 EDA를 별도 작업으로 진행할 수 있다. 이번에는 관계 EDA 자체를 시작하지 않았다. 학습, 예측, 제출 파일 생성과 공식 제출도 수행하지 않았다.", "",
         "## 관련 코드와 로컬 근거", "",
         "- [품질과 의미 점검 코드](../scripts/quality_meaning_audit.py)",
         "- [집계 결과 JSON](../data/eda/2026-10-19_quality_meaning/audit.json)",
         "- [집중값 집계](../data/eda/2026-10-19_quality_meaning/concentration.csv)",
         "- [근사 0 전수 집계](../data/eda/2026-10-19_quality_meaning/near_zero.csv)",
         "- [비정시 보정 후보, 로컬 전용](../data/eda/2026-10-19_quality_meaning/off_hour_candidates.csv)",
         "- [중복 충돌 원본 행, 로컬 전용](../data/eda/2026-10-19_quality_meaning/conflicting_observation_rows.csv)",
         "- [앞선 분포 형태 점검](2026-10-19_03_분포_형태_추가점검.md)", "",
         "로컬 근거와 출력은 data/ 아래에 보관해 Git에서 제외한다. 배포 README와 설명회 원문은 기존 로컬 파일을 참조했고 업로드하거나 외부 서비스로 전송하지 않았다.", ""]
    REPORT.write_text("\n".join(p), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    assert sys.version_info[:2] == (3, 13)
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "data/eda/2026-10-19_basic_shape/summary.json").read_text(encoding="utf-8"))
    if "--verify-only" in sys.argv:
        audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
        assert audit["source_unchanged"] and audit["source_sha256"] == base["source_sha256"]
        assert len(audit["near_zero"]) == 94 and len(audit["concentration"]) == 6
        for r in audit["near_zero"]:
            old = base["files"][r["file"]]["numeric"][r["column"]]
            assert r["n"] == old["count"] and r["exact_zero"] == old["zeros"]
            assert 0 <= r["nonzero_abs_le_1e-12"] <= r["nonzero_abs_le_1e-09"] <= r["nonzero_abs_le_1e-06"] <= r["n"]-r["exact_zero"]
        for r in audit["concentration"]:
            assert r["target_count"]+r["above_target_count"]+r["below_target_count"] == r["n"]
        t = audit["time"]
        assert sum(t["delta_seconds_counts"].values()) == t["off_hour_rows"]
        assert t["collision_payload_all_equal"]+t["collision_payload_conflicting"] == t["candidate_collisions_with_original_keys"]
        assert t["original_absent_hourly_keys"]-t["hypothetical_absent_hourly_keys"] == t["off_hour_rows"]-t["candidate_collisions_with_original_keys"]-t["candidate_internal_duplicates"]
        dup = audit["duplicates"]
        assert dup["consistent_numeric_cells"]+dup["conflicting_numeric_cells"] == dup["duplicate_key_numeric_cells"]
        candidates = pd.read_csv(OUT / "off_hour_candidates.csv", encoding="utf-8-sig")
        duplicates = pd.read_csv(OUT / "conflicting_observation_rows.csv", encoding="utf-8-sig")
        assert len(candidates) == t["off_hour_rows"] and len(duplicates) == dup["participating_rows"]
        assert "\u00b7" not in REPORT.read_text(encoding="utf-8")
        check = {"numeric_columns_cross_checked": 94, "concentration_checks": 6,
                 "time_collision_accounting": True, "duplicate_cell_accounting": True,
                 "local_candidate_and_conflict_row_counts": True, "source_unchanged": True}
        (OUT / "verification.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
        print(json.dumps(check, indent=2), flush=True)
        return
    before = {rel: digest(SOURCE / rel) for rel in FILES}
    assert before == base["source_sha256"]
    RESULT["started_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes")
    for rel in FILES:
        d = pd.read_csv(SOURCE / rel, encoding="utf-8-sig") if rel.endswith(".csv") else pd.read_parquet(SOURCE / rel)
        if "ts" in d:
            d["ts"] = pd.to_datetime(d.ts)
        inspect_values(d, rel)
        if rel in ("train/forecast_wb.parquet", "test/forecast_wb.parquet", "train/observed_wb.parquet"):
            inspect_concentration(d, rel)
        if rel == "train/forecast_wb.parquet":
            inspect_time(d)
        if rel == "train/observed_wa.parquet":
            inspect_duplicates(d)
        if rel == "train/generation.parquet":
            inspect_generation(d)
        del d
        gc.collect()
        log(f"Meaning and quality checks: {rel}")
    assert len(NEAR_ZERO) == 94 and len(CONCENTRATION) == 6
    after = {rel: digest(SOURCE / rel) for rel in FILES}
    assert before == after
    RESULT.update(source_unchanged=True, source_sha256=before, near_zero=NEAR_ZERO,
                  extremes=EXTREMES, concentration=CONCENTRATION,
                  finished_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes"))
    (OUT / "audit.json").write_text(json.dumps(RESULT, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(NEAR_ZERO).to_csv(OUT / "near_zero.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(CONCENTRATION).to_csv(OUT / "concentration.csv", index=False, encoding="utf-8-sig")
    report()
    assert "\u00b7" not in REPORT.read_text(encoding="utf-8")
    log(f"Completed read-only quality checks. Report: {REPORT}")


if __name__ == "__main__":
    main()
