"""Report completed B0 V2 confirmation without fitting or selecting candidates."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from b0_v2_confirm import OUT, REPORT, ROOT, verify_v1_frozen
from report_b0_v1 import f
from s0_validation_check import digest, now, table, write_json
from validation_core import require


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    require(s["status"] == "passed" and s["experiment"] == "B0_V2" and s["V2_executed"], "Completed B0 V2 required")
    require(s["model_fit_count"] == 2 and not s["B1_executed"] and not s["final_evaluation_executed"], "Scope mismatch")
    require(s["v1_frozen"]["selected"] == "B0", "V1 decision changed")
    verify_v1_frozen(s["v1_frozen"])
    require(digest(ROOT / "reports/EDA_진행계획.md") == s["eda_plan_sha256"], "EDA plan changed")
    for rel, expected in s["code_sha256"].items():
        require(digest(OUT / "executed_code" / Path(rel).name) == expected, "Executed code archive mismatch")
    matrices = []
    for label in ("round_10", "round_17", "mean"):
        frame = pd.read_parquet(OUT / f"predictions_{label}.parquet")
        require(len(frame) == 916872 and not frame.duplicated(["site_id", "ts"]).any(), "Saved V2 grid mismatch")
        matrix = frame.pivot(index="ts", columns="site_id", values="kwh")
        require(matrix.shape == (3624, 253) and np.isfinite(matrix.to_numpy()).all()
                and (matrix.to_numpy() >= 0).all(), "Invalid saved V2 predictions")
        matrices.append(matrix)
    pd.testing.assert_frame_equal((matrices[0] + matrices[1])/2, matrices[2])
    for fit in s["fits"]:
        require(digest(OUT / f"model_round_{fit['round']}.txt") == fit["model_sha256"], "Saved V2 model changed")
    scores, individual = s["group_scores"], s["individual_scores"]
    overall = next(r for r in scores if r["sample"] == "common_253" and r["period"] == "all")
    fixed = next(r for r in scores if r["sample"] == "fixed_148" and r["period"] == "all")
    require(overall["times"] == 3624 and np.isclose(overall["price_k_sum"], 616533.54), "S0 V2 score sample drift")
    indiv = {r["scope"]: r for r in individual}
    require(s["train_sites"] == 253 and s["zero_train_sites"] == 0 and "no_train" not in indiv,
            "Empty no-training cohort must remain undefined")
    def group_rows(sample, period_filter):
        return [[({"all": "전체", "Q1": "Q1 (1~3월)", "Q2": "Q2 일부 (4~5월)"}.get(r["period"], r["period"])),
                 f(r["times"], 0), f(r["price_k_sum"]), f(r["W_local"]), f(r["weighted_group_mae_kwh"]),
                 f(r["group_mae_kwh"])] for r in scores if r["sample"] == sample and period_filter(r["period"])]
    # Period labels describe the actual five-month window, never a full-year score.
    headers = ["기간", "시각 수", "ΣSMP×k", "W_local", "가중 집합 MAE (kWh)", "비가중 집합 MAE (kWh)"]
    dnames = dict(all="전체 관측 행", solar_day="태양 고도 > 0", solar_night="태양 고도 <= 0", has_train="학습 기록 있음")
    drows = [[dnames.get(r["scope"], r["scope"]), f(r["rows"], 0), r["sites"], f(r["cf_mae"], 6),
              f(r["kwh_mae"], 4), f(r["bias_kwh"], 4), f(r["cf_bias"], 6)]
             for r in individual if r["scope"] in ("all", "solar_day", "solar_night", "has_train", "k=1", "k=3")]
    months = [r for r in scores if r["sample"] == "common_253" and r["period"].startswith("M")]
    worst = max(months, key=lambda r: r["weighted_group_mae_kwh"])
    lines = ["# B0 V2 후반 확인", "", f"작성: {now()}", "",
             "상태: C0의 B0 V2 재학습, 전체 예측과 성능 평가를 완료했다. V1에서 고정한 선택과 설정을 유지했으며 V2 결과에 따른 후보 추가, 튜닝, 최종 평가 예측과 공식 제출은 수행하지 않았다. EDA 종료와 선택 후보 3개/보류 5개도 유지했다.", "",
             "## 질문과 목적", "",
             "[V1 기압 비교](2026-10-19_21_V1_기압_포함과_제외_비교.md)에서 유지한 기압 포함 B0가 더 최근의 V2에서 어느 정도 오차를 보이는지 확인한다. [설계](2026-10-19_18_검증과_데이터처리_설계.md)의 C0에 해당한다. 선택안과 참조 기준선이 같은 B0이므로 같은 학습을 중복 실행하지 않고 10시/17시 각 1회, 총 2회 적합했다.", "",
             f"전체 253개소, 3,624시간의 **W_local은 {f(overall['W_local'])}**, **가중 집합 MAE는 {f(overall['weighted_group_mae_kwh'])} kWh**다. 전체 관측 cf MAE는 **{f(indiv['all']['cf_mae'], 6)}**, 낮 cf MAE는 **{f(indiv['solar_day']['cf_mae'], 6)}**다. 이는 2025년 1~5월의 로컬 확인 성능이다.", "",
             "## 사용 범위와 수행 방법", "",
             "발전소 메타, 학습 발전량, 학습 A/B 예보와 학습 SMP만 모델 적합과 검증에 사용했다. 기상 실측과 최종 평가 자료는 특징으로 읽지 않았고 원본 보존 해시에만 포함했다. 원본은 수정하지 않았다.", "",
             "| 항목 | 실제 적용 |", "|---|---|",
             "| 학습 범위 | 구간 시작 s >= 2023-01-01 00:00, 종료 ts <= 2024-12-31 00:00 KST |",
             f"| 실제 관측 학습 경계 | {s['fits'][0]['train_ts_min']}~{s['fits'][0]['train_ts_max']} |",
             "| 관측 학습 행 | 2,852,271행, 253개소 모두 학습 기록 있음 |",
             "| 낮 학습 행 | 라운드별 1,430,959행, 태양 고도 > 0 |",
             "| 예측 구간 시작 s | 2025-01-01 00:00 이상, 2025-06-01 00:00 미만 KST |",
             "| 예측 종료 ts | 2025-01-01 01:00~2025-06-01 00:00 KST, 3,624시간 |",
             "| 예측 격자/관측 정답 | 라운드별 253 × 3,624 = 916,872행, 정답 결손 없음 |",
             "| 참조 구성 | current_group A→G1, B→G2. 각각 20,465.655 kW, 22,189.715 kW |", "",
             "각 구간의 정보 시점과 학습 마감은 S0와 같다. 모델은 검증 5개월 동안 갱신하지 않았다. 평가 기간의 실적 시차나 기상 실측을 입력하지 않았다. 전체 24개 특징에 wa_ground_press를 포함하고 기상 결측은 LightGBM 기본 처리로 유지했다. 정확한 정시 키만 조인하며 비정시 예보를 반올림하지 않았다.", "",
             "배포 태양 계산과 float32 특징/목표를 사용했다. 학습 목표 energy/cap은 0~1.2로 자르되 평가 정답은 음수와 큰 값까지 그대로 유지했다. 태양 고도 0 이하의 예측과 음수 출력은 0, 양의 예측에는 상한을 적용하지 않았다. 라운드별 kWh 예측의 산술평균을 최종 예측으로 사용했다.", "",
             "LightGBM은 V1과 같은 n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=None이다. 조기 종료와 설정 탐색은 하지 않았다. 같은 라이브러리 환경을 검사했고 난수를 바꾼 반복 실험은 수행하지 않았다.", "",
             table(["라운드", "낮 학습 행", "기상 NaN 행", "목표 clip 행", "트리 수", "학습 초"],
                   [[r["round"], f(r["daylight_rows"], 0), f(r["daylight_rows_any_weather_nan"], 0),
                     r["clipped_daylight_targets"], r["trees"], f(r["fit_seconds"])] for r in s["fits"]]), "",
             "## 성능 평가 결과", "",
             "W는 두 집합 각각의 예측 합과 정답 합 차이에 절댓값을 취해 SMP×k로 가중한 합이다. k는 KST 종료 11~15시가 3, 나머지가 1이다. R_local=ΣSMP×실제 발전량 합, S_local=R_local-0.10×W_local이며 가중 집합 MAE는 W_local/(2×ΣSMP×k)다. 모든 구성원 정답이 있는 시각에서만 집계한다.", "",
             table(["표본", "시각 수", "ΣSMP×k", "W_local", "R_local", "S_local", "가중 MAE (kWh)"],
                   [[label, f(r["times"], 0), f(r["price_k_sum"]), f(r["W_local"]), f(r["R_local"]),
                     f(r["S_local"]), f(r["weighted_group_mae_kwh"])]
                    for label, r in (("전체 253개소, 주 확인", overall), ("고정 148개소, 진단", fixed))]), "",
             f"주 확인 표본은 V2 전체 3,624시간이다. SMP=0인 {overall['zero_price_times']}시간도 예측과 비가중 진단에서 유지했다. 고정 148개소는 V1의 사후 진단 집단을 그대로 사용했으며 V2에서 253개로 다시 선정하지 않았다. 이 부분집합 지표는 제출 용량 조건을 갖춘 전체 집합 점수가 아니다.", "",
             "### 월/분기 진단, 전체 253개소", "",
             table(headers, group_rows("common_253", lambda p: p.startswith("Q") or p.startswith("M"))), "",
             f"가중 집합 MAE가 가장 큰 월은 {worst['period']}의 {f(worst['weighted_group_mae_kwh'])} kWh다. Q2는 4~5월만 포함하며 6월은 없다. 월/분기 W의 크기는 가격 가중과 시간 수에도 영향을 받으므로 가중 MAE를 함께 본다. 이 진단을 보고 설정을 바꾸지 않았다.", "",
             "### k별 진단, 전체 253개소", "",
             table(headers, group_rows("common_253", lambda p: p.startswith("k="))), "",
             "### 개별 관측 행 진단", "",
             table(["조건", "관측 행", "발전소", "cf MAE", "kWh MAE", "kWh 편향", "cf 편향"], drows), "",
             "cf는 energy/capacity_kw이며 관측 행을 같은 비중으로 집계한다. cf MAE×100은 설비용량으로 정규화한 발전 비율의 %p 오차로 실제 발전량 대비 상대오차가 아니다. 편향은 예측-정답이다. 학습 기록 없음 집단은 0개소이므로 지표를 정의하지 않았다. 비어 있는 월이나 조건도 가짜 0으로 기록하지 않았다. 발전소별, 월/분기 및 고정 148개소 진단은 로컬 산출물에 보존했다.", "",
             "## V1과 연결한 판단", "",
             "V1에서 선택한 B0를 고정해 V2 실행과 점수 산출을 완료했다. 선택안과 참조가 같아 서로 다른 모델 간의 상대 개선이나 방향 역전을 평가할 대상은 없다. B1을 V2에서 추가 실행하지 않았으므로 기압 포함/제외의 V2 우열을 새로 검증했다고 표현하지 않는다.", "",
             "V1의 연간 148개소/일부 시각 253개소와 V2의 253개소 5개월은 기간, 구성과 학습량이 다르다. 두 결과의 W나 MAE 차이를 모델 개선률로 계산하지 않았다. V2에서 나온 오차를 근거로 V1 선택이나 설정을 수정하지 않았다.", "",
             "## 실행 검증", "",
             f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}, PyArrow {s['runtime']['pyarrow']}에서 {s['started_at']}~{s['finished_at']}에 실행했다. 원본 보존 검사와 학습/예측/집계를 포함한 경과 시간은 {f(s['wall_seconds'])}초다.", "",
             f"- 합성 테스트 {s['synthetic_tests']}개를 통과했다. V2 학습/자정 경계, 24개 특징 고정, B1 V2 거부, 학습 기록 없음/없는 월의 지표 미정의와 기존 집계 계약을 확인했다.",
             "- S0와 학습 행/발전소, 낮 행, 정답 행 및 공통 시각 수가 일치했다. 실제 학습 행의 특징과 목표를 원본 배포 순수 함수와 대조했다.",
             "- 라운드별 예측 916,872행을 유지하고 유한/비음수를 검사했다. 저장된 두 라운드와 평균을 다시 읽어 키 유일성, 전체 격자와 평균의 일치를 확인했다.",
             "- 별도 NumPy 집합 누적으로 모든 W를 대조하고 월/분기 W/R/S/가격 가중 합이 전체와 일치함을 확인했다. 저장된 모델과 실행 코드 사본의 해시도 확인했다.",
             "- 원본 10개 데이터 파일과 배포 README/baseline.py, EDA 계획 파일 및 V1 B0/B1 요약/모델/예측 12개 파일의 실행 전후 해시가 일치했다. V1 선택 규칙을 다시 적용해 B0 유지도 확인했다.",
             "- 이번 V2 적합은 2회이며 중복 적합, V1 재적합과 B1 V2 적합은 없다. 앞선 V1의 저장 오류 재적합 1회까지 실제 누적 적합은 총 7회다. 이번 실행에서는 오류 재시도가 없다.", "",
             "## 한계 및 다음 단계", "",
             "정한 C0 범위는 완료했다. 이 기간의 자료 특성은 이미 EDA에서 살펴봤으므로 완전히 미관찰된 독립 시험은 아니다. 일부 계절인 1~5월의 결과이며 연간 공식 성능을 확인한 것도 아니다. 공식 scoring.py와 문제정의서가 없어 공식 채점기 동치는 미확인으로 남긴다. 제공 정적 용량과 현재 그룹, 익명 좌표의 한계도 유지한다.", "",
             "V2는 2026년 A 기압 변화 이전이므로 그 변화에 대한 대응 성능을 입증하지 못한다. 최종 평가 구간에는 발전 실적 정답이 없어 이번 로컬 성능으로 최종 점수를 보장할 수 없다. 최종 학습 마감/범위와 제출 적합성은 후속 설계가 필요하며 이번에는 정하거나 실행하지 않았다. EDA 종료와 후보/보류 목록은 그대로 유지한다.", "",
             "## 관련 코드와 로컬 산출물", "",
             "- [V2 실행과 V1 동결 검사](../scripts/b0_v2_confirm.py), [공통 학습/평가](../scripts/b0_v1_baseline.py)",
             "- [보고서 생성](../scripts/report_b0_v2.py), [V2 테스트](../tests/test_v2_validation.py)",
             "- [실행 환경, 해시와 전체 집계](../data/validation/2026-10-20_B0_V2/summary.json)",
             "- [집합 집계](../data/validation/2026-10-20_B0_V2/group_scores.csv), [관측 행 집계](../data/validation/2026-10-20_B0_V2/individual_scores.csv), [발전소별 진단](../data/validation/2026-10-20_B0_V2/individual_sites.csv)",
             "- [평균 예측](../data/validation/2026-10-20_B0_V2/predictions_mean.parquet), [10시 예측](../data/validation/2026-10-20_B0_V2/predictions_round_10.parquet), [17시 예측](../data/validation/2026-10-20_B0_V2/predictions_round_17.parquet)",
             "- [10시 모델](../data/validation/2026-10-20_B0_V2/model_round_10.txt), [17시 모델](../data/validation/2026-10-20_B0_V2/model_round_17.txt)",
             "- [일별 로그](logs/2026-10-20.md), [EDA 상태](EDA_진행계획.md)", "",
             "모델, 데이터가 담긴 출력과 실행 코드 사본은 Git 제외 경로 data/에만 보관했다. 공개 보고서에는 원본/전처리 행과 개별 예측을 복사하지 않았다. 저장된 집계에서 보고서를 재생성할 때는 모델을 다시 적합하지 않는다.", ""]
    text = "\n".join(lines)
    require(not any(c in text for c in ("\u00b7", "\u2014", "\u2013")), "Forbidden punctuation")
    REPORT.write_text(text, encoding="utf-8")
    write_json(OUT / "report_manifest.json", dict(generated_at=now(), generator_sha256=digest(Path(__file__)),
               report_sha256=digest(REPORT), summary_sha256=digest(OUT / "summary.json")))
    print(f"Report generated: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
