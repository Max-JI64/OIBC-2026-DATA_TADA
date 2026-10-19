"""Execute S0 implementation checks only. All data outputs remain in ignored data/."""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import platform
import sys
import sysconfig
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow

from validation_core import (KST, KEY, ROUNDS, WINDOWS, clock, grid, group_totals,
                             join_forecasts, metadata, prepare_forecast, require,
                             times, truth_matrix, unique, weights)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "OIBC_2026_DATA/01_participant_dataset"
OUT = ROOT / "data/validation/2026-10-19_S0"
REPORT = ROOT / "reports/2026-10-19_19_S0_구현_점검.md"
INPUTS = ["sites.csv", "smp.csv", "train/smp.csv", "train/generation.parquet",
          "train/forecast_wa.parquet", "train/forecast_wb.parquet",
          "test/forecast_wa.parquet", "test/forecast_wb.parquet",
          "train/observed_wa.parquet", "train/observed_wb.parquet", "baseline.py", "README.md"]


def now():
    return datetime.now(ZoneInfo(KST)).isoformat(timespec="minutes")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def baseline_contract():
    """Read only literals and the pure solar function; never import/run baseline.py."""
    tree = ast.parse((SOURCE / "baseline.py").read_text(encoding="utf-8"))
    spec = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("WA_COLS", "WB_COLS"):
                spec[name] = ast.literal_eval(node.value)
            elif name == "FEATS":
                # This expression only concatenates the literal feature lists.
                require(isinstance(node.value, ast.BinOp), "Unexpected baseline features")
                spec[name] = eval(compile(ast.Expression(node.value), "<baseline-features>", "eval"),
                                  {"__builtins__": {}}, spec)
        if isinstance(node, ast.FunctionDef) and node.name == "solar":
            namespace = {"np": np, "pd": pd}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE / "baseline.py"), "exec"), namespace)
            spec["solar"] = namespace["solar"]
    require(set(spec) == {"WA_COLS", "WB_COLS", "FEATS", "solar"}, "Baseline contract unavailable")
    require(spec["FEATS"] == spec["WA_COLS"] + spec["WB_COLS"] +
            ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon"],
            "Baseline feature list changed")
    return spec


def monthly(axis):
    day = (axis - pd.Timedelta(hours=1)).strftime("%Y-%m")
    for month in sorted(set(day)):
        yield axis[day == month]


def load_forecasts(part, spec):
    results = []
    excluded = {}
    for prefix, label in (("wa", "WA_COLS"), ("wb", "WB_COLS")):
        station = prefix + "_station"
        raw = pd.read_parquet(SOURCE / f"{part}/forecast_{prefix}.parquet",
                              columns=[station, "round", "ts"] + spec[label])
        clean, n = prepare_forecast(raw, station, spec[label])
        excluded[prefix] = n
        results.append(clean)
        del raw
    log(f"Loaded {part} forecasts, nonhourly exclusions={excluded}")
    return (*results, excluded)


def audit_joins(label, base, fa, fb, spec, stats):
    """One monthly batch, both rounds. Never filter base on forecast availability."""
    for r in ROUNDS:
        a = fa.loc[fa.ts.isin(base.ts.unique())]
        b = fb.loc[fb.ts.isin(base.ts.unique())]
        d = join_forecasts(base, a, b, r)
        key = (label, r)
        s = stats.setdefault(key, {"scope": label, "round": r, "rows": 0,
                                   "wa_missing_rows": 0, "wb_missing_rows": 0,
                                   "rows_any_weather_nan": 0, "daylight_rows": 0,
                                   "nan_by_feature": {col: 0 for col in spec["WA_COLS"] + spec["WB_COLS"]}})
        s["rows"] += len(d)
        for prefix in ("wa", "wb"):
            s[prefix + "_missing_rows"] += int((~d[prefix + "_row_present"]).sum())
        cols = spec["WA_COLS"] + spec["WB_COLS"]
        s["rows_any_weather_nan"] += int(d[cols].isna().any(axis=1).sum())
        for col in cols:
            s["nan_by_feature"][col] += int(d[col].isna().sum())
        elev, az, cosz = spec["solar"](pd.DatetimeIndex(d.ts), d.lat.to_numpy(), d.lon.to_numpy())
        require(np.isfinite(np.stack([elev, az, cosz])).all(), "Invalid solar feature")
        d["elev"], d["azim"], d["cosz"] = elev, az, cosz
        ts = pd.DatetimeIndex(d.ts)
        d["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear.to_numpy() / 365.25).astype("float32")
        d["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear.to_numpy() / 365.25).astype("float32")
        d["hour"] = ts.hour.to_numpy().astype("int8")
        d["cap"] = d.capacity_kw.astype("float32")
        x = d[spec["FEATS"]].astype("float32")
        require(not np.isinf(x.to_numpy()).any(), "Infinite model feature")
        require(x[["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon"]]
                .notna().all().all(), "Missing derived feature")
        s["daylight_rows"] += int((elev > 0).sum())
        if "energy" in d:
            clipped = (d.energy / d.capacity_kw.astype("float32")).clip(0, 1.2)
            require(np.isfinite(clipped).all(), "Invalid clipped training target")
            s["clipped_target_rows"] = s.get("clipped_target_rows", 0) + int(
                ((d.energy / d.capacity_kw.astype("float32")) != clipped).sum())
        del d, a, b, x


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def build_report(summary):
    n = lambda value: f"{value:,}"
    lines = ["# S0 시점, 조인, 정답 가용성과 로컬 집계 구현 점검", "", f"작성: {summary['finished_at']}", "",
             "상태: 정한 S0 구현 점검을 실제 데이터와 합성 예제로 완료했다. 모델 학습, 발전량 예측, 모델 성능 점수 계산과 공식 제출은 수행하지 않았다. 기본 EDA 종료와 선택 후보 3개, 보류 5개는 유지했다.", "",
             "## 질문과 목적", "", "[검증과 데이터 처리 설계](2026-10-19_18_검증과_데이터처리_설계.md)의 시간 경계와 표본을 구현할 수 있는지, 결손을 조용히 삭제하거나 부분 정답 합계를 만들지 않는지 확인한다. S0 결과를 사용자와 확인한 뒤 B0/B1 학습 여부를 정한다.", "",
             "## 사용 범위와 수행 방법", "",
             f"원본 메타로 참조 구성과 용량 조건을 다시 검사했다. G1(A)은 {summary['capacity_kw']['A']:,.3f} kW, G2(B)는 {summary['capacity_kw']['B']:,.3f} kW이며 합계는 {sum(summary['capacity_kw'].values()):,.3f} kW다. 253개소에 누락/중복이 없고 두 집합 모두 20,000 kW 이상이다.", "",
             "발전소 메타, 학습 발전량, 학습/평가 예보와 SMP를 사용했다. 실측 기상은 특징으로 읽거나 조인하지 않았으며 원본 보존 해시 확인에만 포함했다. 원본 10개 데이터 파일과 배포 README/baseline.py의 실행 전후 SHA256이 일치했다. 입력 해시, 코드 해시, 환경, 실행 시각과 발전소별 집계는 로컬 산출물에 기록했다.", "",
             "S0는 학습 관측 행과 V1/V2/최종 평가의 253개소 전체 격자를 월 단위로 생성했다. 각 라운드별로 메타와 예보를 왼쪽 조인하고 키 유일성, 행 수와 순서를 검사했다. 비정시 예보는 반올림 없이 제외하고 없는 행과 행 안의 NaN을 따로 집계했다. 학습 마감은 설계 그대로 유지했으며 전체 기간의 평균이나 보정값을 적합하지 않았다.", "",
             "배포 baseline.py를 import하거나 main을 실행하지 않고 AST에서 특징 목록과 순수 태양 위치 함수만 읽었다. 15개 기상 특징과 9개 태양/달력/메타 특징의 입력 범위를 확인했다. 태양 고도와 학습 목표 clip의 유한성을 점검했으나 학습 행렬 저장이나 모델 적합은 하지 않았다. 태양 고도 0 이하의 행도 이번 조인 검사에는 남기고, 추후 학습에 쓰일 낮 행 수를 별도 기록했다.", "",
             "예보 발행 시각은 배포 정의에 따라 대상일 전일 10/17시로 파생해 예측 시점 전일 17시 이하인지 검사했다. 실제 수신 시각 필드가 없어 실시간 도착을 검증한 결과는 아니다.", "",
             "## 표본과 학습 마감", ""]
    rows = []
    for w in summary["windows"]:
        rows.append([w["name"], w["train_ts_min"], w["train_ts_max"], n(w["train_rows"]),
                     w["train_sites"], w["zero_train_sites"], n(w["daylight_train_rows_per_round"])])
    lines += [table(["구간", "실제 학습 시작 ts", "실제 학습 끝 ts", "관측 학습 행", "학습 기록 발전소", "학습 기록 없음", "낮 학습 후보/라운드"], rows), "",
              "학습 시작과 끝은 관측 행의 실제 경계다. V1 학습 s는 2022-01-01 이상, ts는 2023-12-31 00시 이하이고 V2 학습 s는 2023-01-01 이상, ts는 2024-12-31 00시 이하이다. 마감부터 첫 예측 시점까지 17시간의 간격이 있다. 학습 기록이 없는 발전소도 예측 격자에서 유지했다.", ""]
    rows = [[w["name"], n(w["target_times"]), n(w["prediction_rows_per_round"]), n(w["observed_target_rows"]),
             w["observed_target_sites"], n(w["common_253_times"]), n(w["fixed_148_common_times"])] for w in summary["windows"]]
    lines += [table(["구간", "예측 시각", "253개 격자/라운드", "관측 정답 행", "정답 관측 발전소", "253개 공통 시각", "고정 148개 공통 시각"], rows), "",
              "고정 148개는 2024년 전체 8,784시간 정답을 가진 EDA의 사후 진단 집단을 재구성한 것이다. V1의 253개 공통 시각이 있어도 연간 개발 표본을 바꾸지 않고, 설계대로 W_148_proxy를 별도 개발 지표로 유지한다.", "",
              "## 집합 정답 마스크와 가격 가중", ""]
    rows = []
    for w in summary["windows"]:
        for sample in w["samples"]:
            rows.append([w["name"], sample["sample"], n(sample["times"]), sample["first_ts"], sample["last_ts"],
                         f"{sample['smp_sum']:,.2f}", f"{sample['price_k_sum']:,.2f}", n(sample["zero_price_times"])])
    lines += [table(["구간", "표본", "시각 수", "최초 ts", "최종 ts", "ΣSMP", "ΣSMP×k", "SMP=0"], rows), "",
              "각 시각의 완전성 마스크로 표본을 정하고, 모든 구성원이 관측된 시각에만 집합 정답을 합산했다. V1의 전체 253개 집합 지표는 연말 178시각에만 정의되며 연간 지표가 아니다. 148개 하위집단의 지표도 제출 용량 조건을 충족한 전체 집합의 성능으로 해석하지 않는다. 음수와 큰 정답을 자르거나 결측 실적을 0으로 채우지 않았다. 실제 예측값이 없으므로 이 자료의 W/S_local이나 정확도를 보고하지 않는다.", "",
              "## 예보 조인과 전체 격자 유지", ""]
    rows = [[s["scope"], s["round"], n(s["rows"]), n(s["wa_missing_rows"]), n(s["wb_missing_rows"]),
             n(s["rows_any_weather_nan"])] for s in summary["joins"]]
    lines += [table(["범위", "라운드", "조인 후 행", "A 예보 행 부재", "B 예보 행 부재", "기상 특징 NaN 행"], rows), "",
              f"정확한 정시만 남기는 과정에서 제외한 원본 예보 행은 학습 A {summary['nonhourly']['train']['wa']}개, B {summary['nonhourly']['train']['wb']}개, 평가 A {summary['nonhourly']['test']['wa']}개, B {summary['nonhourly']['test']['wb']}개다. 이는 격자 행 삭제 수가 아니다. 한 예보 지점을 여러 발전소가 공유하므로 표의 결손 수는 원본 지점 행 수와 다르다.", "",
              f"최종 평가의 {summary['evaluation_times']:,}시각을 그대로 유지해 라운드당 {summary['evaluation_grid_rows']:,}행을 검사했다. SMP=0인 {summary['evaluation_zero_price_times']:,}시각도 유지했고 제공 k와 종료 시각 기준 k가 일치했다. 예보 행 부재와 값 NaN을 구분해 기록했으며 독립적인 EDA를 새로 열지 않았다.", "",
              "## 합성 예제와 구현 검증", "",
              f"정규 CPython {summary['runtime']['python']}에서 {summary['synthetic_tests']}개 테스트를 통과했다. 주요 검사는 다음과 같다.", "",
              "- 자정 종료 행의 대상 날짜와 전일 예측 시점, 학습 마감 포함/초과와 시작 경계, KST 종료 10/11/15/16시의 k=1/3/3/1을 확인했다.",
              "- 집합 내부 +5/-5 오차는 상쇄되고 서로 다른 집합의 +5/-5는 상쇄되지 않았다. 후자의 SMP=2, k=3 예제는 W_local=60, R_local=200, S_local=194로 손계산과 일치했다.",
              "- 결측 정답은 완전성 마스크에서 제외되며, 고정 점수 표본에 끼워 넣으면 오류로 중단했다. 결측/음수/무한 예측은 SMP=0이어도 거부했다. 원본 음수 정답은 점수에 그대로 반영했다.",
              "- 중복 조인 키, 중복/결손 가격, 예측 축 불일치를 거부하고 예보 없는 행과 값 NaN을 구분했다. 가중 분모 0에서는 가중 MAE를 정의하지 않고 빈 점수 표본은 거부했다.", "",
              "합성 예제만 로컬 점수 함수를 호출했다. 공식 scoring.py와 문제정의서가 배포 폴더에 없어 공식 채점기와 동치는 확인하지 못했다.", "",
              "## 한계와 다음 단계", "",
              "S0 종료 기준인 시점, 조인, 표본, 정답 마스크와 로컬 집계를 확인했다. 현재 확인한 범위에서 학습 전에 고쳐야 할 차단 오류는 남지 않았다. 결손 예보의 모델 내부 처리, B0/B1 모델 적합과 예측 생성, 성능 및 최종 제출 적합성은 아직 확인하지 않았다. 학습 기록이 없는 발전소의 일반화 성능도 미확인이다.", "",
              "이번 결과로 기간이나 개발 표본을 바꾸지 않았다. [EDA 진행 계획](EDA_진행계획.md)의 종료 상태와 보류 목록도 변경하지 않았다. S0 결과를 사용자가 확인한 뒤, 기존 예산과 선택 규칙에 따른 B0/B1 V1 비교를 별도 실행한다. 현재 코드는 모델을 import하거나 학습하지 않는다.", "",
              "## 관련 코드와 로컬 산출물", "",
              "- [시간, 표본, 조인과 로컬 집계](../scripts/validation_core.py)",
              "- [S0 실행과 보고서 생성](../scripts/s0_validation_check.py)",
              "- [합성 예제 테스트](../tests/test_validation_core.py)",
              "- [입력/코드 해시, 실행 환경과 집계 요약](../data/validation/2026-10-19_S0/summary.json)",
              "- [발전소별 학습/정답 가용성](../data/validation/2026-10-19_S0/site_availability.csv)",
              "- [분기별 고정 표본 가용성](../data/validation/2026-10-19_S0/sample_quarters.csv)",
              "- [일별 로그](logs/2026-10-19.md)", "",
              "실행은 정규 Python 3.13의 승인된 Start-Process 경로로 scripts/s0_validation_check.py를 실행한다. --report-only는 저장된 집계에서 보고서만 다시 만들고 합성 테스트를 재실행한다. 원본 및 개별 데이터와 집계 파일은 Git 제외 경로 data/에만 보관한다. 공개 보고서에는 원본/전처리 행을 복사하지 않았다.", ""]
    if "postrun_revision" in summary:
        lines += [f"보고서 정정: {summary['postrun_revision']['at']}에 학습 마감과 첫 예측 시점의 간격 설명을 41시간에서 17시간으로 바로잡았다. 실제 시간 경계와 표본 계산은 변경하지 않았다. 합성 테스트를 다시 통과했으며 원본 재분석과 모델 실행은 추가하지 않았다. 최초 실행 코드 해시는 summary.json에 보존했고 실행 당시 소스는 로컬 executed_code/에 보관했다. 보고서 재생성 코드 해시는 별도로 기록했다.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),
            "Regular CPython 3.13 required")
    OUT.mkdir(parents=True, exist_ok=True)
    started = now()
    code_files = [Path(__file__), ROOT / "scripts/validation_core.py", ROOT / "tests/test_validation_core.py"]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    if "--report-only" not in sys.argv:
        archive = OUT / "executed_code"
        archive.mkdir(exist_ok=True)
        for path in code_files:
            (archive / path.name).write_bytes(path.read_bytes())
    eda_hash = digest(ROOT / "reports/EDA_진행계획.md")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_validation_core.py")
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(test_result.wasSuccessful(), "Synthetic contracts failed")
    if "--report-only" in sys.argv:
        summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        require(summary["status"] == "passed", "Full S0 run must pass first")
        require(eda_hash == summary["eda_plan_sha256"], "EDA list changed")
        for rel, expected in summary["code_sha256"].items():
            require(digest(OUT / "executed_code" / Path(rel).name) == expected, "Executed source archive mismatch")
        summary["postrun_revision"] = {"at": now(), "code_sha256": code_hashes,
                                       "synthetic_tests": test_result.testsRun,
                                       "change": "Report wording and run metadata support; cutoff gap 41 -> 17 hours; sample/feature/score algorithms unchanged"}
        build_report(summary)
        require("\u00b7" not in REPORT.read_text(encoding="utf-8"), "Forbidden punctuation")
        write_json(OUT / "summary.json", summary)
        log("Report regenerated from saved aggregates; synthetic tests passed; no data rerun")
        return
    before = {f: digest(SOURCE / f) for f in INPUTS}
    spec = baseline_contract()
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig")
    capacities = metadata(sites)
    gen = pd.read_parquet(SOURCE / "train/generation.parquet")
    gen["ts"] = times(gen.ts)
    unique(gen, KEY)
    require(gen.site_id.isin(sites.site_id).all() and np.isfinite(gen.energy).all(), "Invalid generation")
    require(gen.ts.eq(gen.ts.dt.floor("h")).all(), "Nonhourly generation")
    price = pd.read_csv(SOURCE / "train/smp.csv", encoding="utf-8-sig")
    price["ts"] = times(price.ts)
    unique(price, ["ts"])
    test_price = pd.read_csv(SOURCE / "smp.csv", encoding="utf-8-sig")
    test_price["ts"] = times(test_price.ts)
    unique(test_price, ["ts"])
    eval_axis = pd.date_range("2025-06-01 01:00", "2026-06-01 00:00", freq="h", tz=KST)
    require(times(test_price.ts).equals(eval_axis), "Evaluation axis mismatch")
    require(np.array_equal(test_price.k.to_numpy(), weights(eval_axis)), "Provided k mismatch")
    require(np.isfinite(test_price.smp).all() and test_price.smp.ge(0).all(), "Invalid evaluation price")
    fixed = truth_matrix(gen, sites.site_id, WINDOWS[0].axis())
    fixed_ids = fixed.columns[fixed.notna().all(axis=0)].tolist()
    require(len(fixed_ids) == 148, "Fixed EDA diagnostic cohort changed")
    groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    fa, fb, train_nonhourly = load_forecasts("train", spec)
    joins, windows, availability, quarters = {}, [], [], []
    for window in WINDOWS:
        log(f"Checking {window.name}")
        axis = window.axis()
        train = window.training(gen)
        target = truth_matrix(gen, sites.site_id, axis)
        target_groups, complete = group_totals(target, groups)
        fixed_groups, fixed_complete = group_totals(target[fixed_ids], groups)
        sample_list = []
        for label, mask in (("common_253", complete), ("fixed_148", fixed_complete)):
            selected = axis[mask]
            p = price.set_index("ts").reindex(selected).smp
            require(np.isfinite(p).all() and p.ge(0).all(), "Missing sample prices")
            item = {"sample": label, "times": len(selected),
                    "first_ts": str(selected.min()) if len(selected) else None,
                    "last_ts": str(selected.max()) if len(selected) else None,
                    "smp_sum": float(p.sum()), "price_k_sum": float((p * weights(selected)).sum()),
                    "zero_price_times": int(p.eq(0).sum())}
            sample_list.append(item)
            for quarter in sorted(set((axis - pd.Timedelta(hours=1)).quarter)):
                selected_q = selected[(selected - pd.Timedelta(hours=1)).quarter == quarter]
                quarters.append({"window": window.name, "sample": label, "quarter": int(quarter),
                                 "times": len(selected_q), "price_k_sum": float(
                                     (price.set_index("ts").reindex(selected_q).smp * weights(selected_q)).sum())})
        per_site = train.groupby("site_id").agg(train_rows=("ts", "size"), first_train_ts=("ts", "min"), last_train_ts=("ts", "max"))
        per_site = per_site.reindex(sites.site_id)
        per_site["train_rows"] = per_site.train_rows.fillna(0).astype(int)
        per_site["observed_target_rows"] = target.notna().sum().reindex(per_site.index)
        per_site["window"] = window.name
        per_site["fixed_148"] = per_site.index.isin(fixed_ids)
        availability.append(per_site.reset_index())
        for batch_axis in monthly(pd.DatetimeIndex(train.ts.unique()).sort_values()):
            part = train.loc[train.ts.isin(batch_axis)].merge(sites, on="site_id", how="left", validate="many_to_one")
            audit_joins(window.name + "_train", part, fa, fb, spec, joins)
        for batch_axis in monthly(axis):
            audit_joins(window.name + "_grid", grid(sites, batch_axis), fa, fb, spec, joins)
        for r in ROUNDS:
            require(joins[(window.name + "_train", r)]["rows"] == len(train), "Training row mismatch")
            require(joins[(window.name + "_grid", r)]["rows"] == len(axis) * len(sites), "Grid row mismatch")
        windows.append({"name": window.name, "train_rows": len(train),
                        "train_ts_min": str(train.ts.min()), "train_ts_max": str(train.ts.max()),
                        "cutoff": str(pd.Timestamp(window.cutoff, tz=KST)),
                        "train_sites": int(per_site.train_rows.gt(0).sum()), "zero_train_sites": int(per_site.train_rows.eq(0).sum()),
                        "daylight_train_rows_per_round": joins[(window.name + "_train", 10)]["daylight_rows"],
                        "target_times": len(axis), "prediction_rows_per_round": len(axis) * len(sites),
                        "observed_target_rows": int(target.notna().to_numpy().sum()),
                        "observed_target_sites": int(target.notna().any().sum()), "common_253_times": int(complete.sum()),
                        "fixed_148_common_times": int(fixed_complete.sum()), "samples": sample_list})
        log(f"{window.name}: train={len(train):,}, grid={len(axis)*len(sites):,}, common253={int(complete.sum()):,}")
    del fa, fb, train, target, fixed, part
    gc.collect()
    fa, fb, test_nonhourly = load_forecasts("test", spec)
    for batch_axis in monthly(eval_axis):
        audit_joins("evaluation_grid", grid(sites, batch_axis), fa, fb, spec, joins)
    for r in ROUNDS:
        require(joins[("evaluation_grid", r)]["rows"] == 253 * 8760, "Evaluation grid row mismatch")
    require(before == {f: digest(SOURCE / f) for f in INPUTS}, "Source changed")
    require(eda_hash == digest(ROOT / "reports/EDA_진행계획.md"), "EDA list changed")
    require(code_hashes == {str(p.relative_to(ROOT)): digest(p) for p in code_files}, "Code changed during run")
    require("lightgbm" not in sys.modules, "Model library unexpectedly imported")
    summary = {"experiment": "S0", "status": "passed", "started_at": started, "finished_at": now(),
               "runtime": {"executable": sys.executable, "python": platform.python_version(), "pandas": pd.__version__,
                           "numpy": np.__version__, "pyarrow": pyarrow.__version__, "free_threaded": False},
               "source_sha256": before, "code_sha256": code_hashes, "source_unchanged": True,
               "eda_plan_sha256": eda_hash, "eda_state_unchanged": True, "model_fit_count": 0,
               "model_prediction_count": 0, "model_scores_computed": False, "synthetic_tests": test_result.testsRun,
               "capacity_kw": {k: float(v) for k, v in capacities.items()}, "fixed_148_sites": fixed_ids,
               "weather_features": spec["WA_COLS"] + spec["WB_COLS"], "windows": windows,
               "model_features": spec["FEATS"],
               "joins": list(joins.values()), "nonhourly": {"train": train_nonhourly, "test": test_nonhourly},
               "evaluation_times": len(eval_axis), "evaluation_grid_rows": len(eval_axis) * 253,
               "evaluation_zero_price_times": int(test_price.smp.eq(0).sum())}
    pd.concat(availability).to_csv(OUT / "site_availability.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quarters).to_csv(OUT / "sample_quarters.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "summary.json", summary)
    build_report(summary)
    require("\u00b7" not in REPORT.read_text(encoding="utf-8"), "Forbidden punctuation")
    log(f"S0 passed: {test_result.testsRun} tests; no model fit/predict; report={REPORT.name}")


if __name__ == "__main__":
    main()
