"""Build the V1 pressure comparison report from frozen B0 and completed B1."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from b1_v1_compare import B0_OUT, OUT, REPORT, ROOT, compare_results
from report_b0_v1 import f
from s0_validation_check import digest, now, table, write_json
from validation_core import require


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    b0 = json.loads((B0_OUT / "summary.json").read_text(encoding="utf-8"))
    require(s["status"] == "passed" and s["experiment"] == "B1_V1" and s["model_fit_count"] == 2,
            "Completed B1 V1 required")
    require(not s["V2_executed"] and not s["final_evaluation_executed"], "Scope mismatch")
    require(digest(ROOT / "reports/EDA_진행계획.md") == s["eda_plan_sha256"], "EDA plan changed")
    for rel, expected in s["code_sha256"].items():
        require(digest(OUT / "executed_code" / Path(rel).name) == expected, "Code archive mismatch")
    for name, expected in s["comparison"]["b0_artifact_sha256"].items():
        require(digest(B0_OUT / name) == expected, "Frozen B0 artifact changed")
    matrices = []
    for label in ("round_10", "round_17", "mean"):
        frame = pd.read_parquet(OUT / f"predictions_{label}.parquet")
        require(len(frame) == 2222352 and not frame.duplicated(["site_id", "ts"]).any(), "B1 saved grid mismatch")
        matrix = frame.pivot(index="ts", columns="site_id", values="kwh")
        require(matrix.shape == (8784, 253) and np.isfinite(matrix.to_numpy()).all()
                and (matrix.to_numpy() >= 0).all(), "Invalid B1 saved predictions")
        matrices.append(matrix)
    pd.testing.assert_frame_equal((matrices[0] + matrices[1])/2, matrices[2])
    for fit in s["fits"]:
        require(digest(OUT / f"model_round_{fit['round']}.txt") == fit["model_sha256"], "B1 saved model changed")
    comparison = compare_results(b0["group_scores"], s["group_scores"])
    for key in comparison:
        require(comparison[key] == s["comparison"][key], "Decision changed during reporting")
    pd.DataFrame(comparison["rows"]).to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")
    chosen = comparison["selected_for_V2"]
    annual = next(r for r in comparison["rows"] if r["sample"] == "fixed_148" and r["period"] == "all")
    reasons = dict(annual_improved_and_quarter_guard_passed="B1의 연간 대리지표가 개선됐고 모든 분기의 악화가 5% 이내다.",
                   annual_not_improved_or_tied="B1의 연간 대리지표가 개선되지 않았거나 동률이므로 B0를 유지한다.",
                   quarter_guard_failed="B1의 연간 대리지표는 개선됐지만 분기별 악화 기준을 넘겨 B0를 유지한다.")
    if comparison["reason"] == "annual_not_improved_or_tied" and annual["delta_W"] > 0:
        reasons["annual_not_improved_or_tied"] = f"B1의 연간 대리지표가 B0보다 {annual['change_pct']:.4f}% 악화돼 B0를 유지한다."
    def group_rows(period_filter, sample="fixed_148"):
        return [[r["period"], f(r["times"], 0), f(r["B0_W"]), f(r["B1_W"]),
                 f"{r['change_pct']:+.4f}%" if r["change_pct"] is not None else "정의하지 않음",
                 f(r["B0_weighted_group_mae_kwh"]), f(r["B1_weighted_group_mae_kwh"])]
                for r in comparison["rows"] if r["sample"] == sample and period_filter(r["period"])]
    group_headers = ["기간", "시각 수", "B0 W", "B1 W", "W 변화율", "B0 가중 MAE (kWh)", "B1 가중 MAE (kWh)"]
    individual_old = {r["scope"]: r for r in b0["individual_scores"]}
    names = dict(all="전체 관측", solar_day="태양 고도 > 0", solar_night="태양 고도 <= 0",
                 has_train="학습 기록 있음", no_train="학습 기록 없음")
    individual_rows = []
    for r in s["individual_scores"]:
        if r["scope"] not in ("all", "solar_day", "solar_night", "has_train", "no_train", "k=1", "k=3"):
            continue
        old = individual_old[r["scope"]]
        require(r["rows"] == old["rows"] and r["sites"] == old["sites"], "Individual sample mismatch")
        individual_rows.append([names.get(r["scope"], r["scope"]), f(r["rows"], 0),
                                f(old["cf_mae"], 6), f(r["cf_mae"], 6), f(old["kwh_mae"], 4),
                                f(r["kwh_mae"], 4), f(old["bias_kwh"], 4), f(r["bias_kwh"], 4)])
    lines = ["# V1 기압 포함과 제외 비교", "", f"작성: {now()}", "",
             "상태: B1 V1의 두 라운드 학습과 평균 예측을 완료하고 저장된 B0와 같은 표본에서 비교했다. V2, 튜닝과 공식 제출은 실행하지 않았다. EDA 종료와 선택 후보 3개/보류 5개는 유지했다.", "",
             "## 질문과 목적", "",
             "A 지표기압 wa_ground_press만 제외했을 때 V1 성능이 개선되는지 확인하고 [기존 설계](2026-10-19_18_검증과_데이터처리_설계.md)의 연간/분기 선택 규칙을 적용한다. [B0 결과](2026-10-19_20_B0_V1_기준선_학습과_평가.md)는 그대로 보존했다.", "",
             f"**V1 규칙에 따른 V2 확인 대상은 {chosen}이다.** {reasons[comparison['reason']]} 이 판단은 이번 V1 비교의 선택 결과이며 V2 성능 확인이나 최종 제출 방법 확정을 뜻하지 않는다.", "",
             "## 사용 범위와 수행 방법", "",
             "B0와 같은 원본, 구성, 시간 경계, 학습 행, 정답, SMP와 가격 가중을 사용했다. V1 학습은 구간 시작 s >= 2022-01-01 00:00, 종료 ts <= 2023-12-31 00:00 KST다. 검증 s는 2024-01-01 00:00 이상, 2025-01-01 00:00 미만이며 종료 ts는 2024-01-01 01:00~2025-01-01 00:00 KST의 8,784시간이다.", "",
             "공통 학습 코드는 B0를 기본값으로 유지하면서 실험 ID에 따른 특징 선택과 출력 위치만 받도록 확장했다. 기존 B0 실행 코드 사본은 이전 로컬 산출물에 유지했다. B1은 24개 특징을 동일하게 생성한 뒤 wa_ground_press 한 열만 제외한 23개를 같은 순서로 사용했다. B0와 환경 및 LightGBM 설정이 정확히 같은지 실행 전에 검사했다.", "",
             "학습 기록 1,442,111행 중 태양 고도 > 0인 719,235행을 라운드별로 사용했다. 148개소에 학습 기록이 있고 105개소에는 없다. 목표 energy/cap을 0~1.2로 자르고, 예측의 야간 0/음수 0/상한 미적용과 10시/17시 kWh의 산술평균을 그대로 유지했다. 결측 예보는 LightGBM 기본 처리에 맡기고 비정시 예보는 반올림하지 않았다. 원본 정답을 자르거나 없는 실적을 채우지 않았다.", "",
             "LightGBM은 n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1이며 random_state=None도 B0와 같다. 조기 종료, 특징 추가, 손실/가중 변경과 난수 탐색은 하지 않았다. 제외로 인한 특징 수 변경은 모델 내부 특징 추출에도 영향을 줄 수 있어 기압의 독립적인 인과 효과로 해석하지 않는다.", "",
             table(["라운드", "낮 학습 행", "남은 기상 NaN 행", "목표 clip 행", "트리 수", "학습 초"],
                   [[r["round"], f(r["daylight_rows"], 0), f(r["daylight_rows_any_weather_nan"], 0),
                     r["clipped_daylight_targets"], r["trees"], f(r["fit_seconds"])] for r in s["fits"]]), "",
             "## 결과", "",
             "W 변화율은 (B1 W-B0 W)/B0 W×100이다. 음수는 개선, 양수는 악화다. 같은 표본에서는 가중 MAE 변화율도 같다. W는 두 집합의 오차를 각각 합산한 SMP×k 가중 절대오차이며 낮을수록 좋다. S_local=R_local-0.10×W_local이고 같은 정답의 R은 두 방법에서 같다.", "",
             "### 연간 개발 대리지표와 분기", "",
             table(group_headers, group_rows(lambda p: p == "all" or p.startswith("Q"))), "",
             "이 표는 고정 148개소의 W_148_proxy다. 가격 가중 합 ΣSMP×k는 연간 1,600,600.36이며 B0/B1이 동일하다. 148개소는 2024년 전체 정답 가용성으로 선정한 사후 진단 집단으로 전체 253개소의 연간 제출 성능을 대표하지 않는다.", "",
             "### 월별 진단", "",
             table(group_headers, group_rows(lambda p: p.startswith("M"))), "",
             "### 전체 253개소의 공통 시각", "",
             table(group_headers, group_rows(lambda p: p == "all", sample="common_253")), "",
             "전체 집합 표본은 2024-12-24 15:00~2025-01-01 00:00 KST의 178시간이며 ΣSMP×k는 28,013.78이다. 연간 고정 148개소 지표와 기간 및 구성원이 다르다. 이 짧은 구간의 개선/악화로 개발 표본이나 사전 선택 규칙을 바꾸지 않았다.", "",
             "### k별 고정 148개소 진단", "",
             table(group_headers, group_rows(lambda p: p.startswith("k="))), "",
             "k는 KST 종료 11~15시에 3, 나머지에 1이다. SMP=0인 6시간도 예측과 비가중 진단에 남겼다. 정확한 월/분기/가격 가중과 S_local 차이는 로컬 comparison.csv에 보존했다.", "",
             "### 개별 관측 행 진단", "",
             table(["조건", "관측 행", "B0 cf MAE", "B1 cf MAE", "B0 kWh MAE", "B1 kWh MAE", "B0 kWh 편향", "B1 kWh 편향"], individual_rows), "",
             "cf는 energy/capacity_kw이며 관측 행마다 같은 비중으로 집계한다. cf MAE×100은 설비용량으로 정규화한 발전 비율의 %p 오차이며 실제 발전량 대비 상대오차가 아니다. 편향은 예측-정답이다. 학습 기록 유무별 표본의 기간과 발전소가 달라 집단 간 차이를 신규 발전소 효과로 해석하지 않는다. 발전소별 진단은 로컬 파일에 보존했다.", "",
             "## 선택 판단", "",
             f"기존 규칙에 따라 {chosen}를 V2 확인 대상으로 남긴다. {reasons[comparison['reason']]}", "",
             "연간 W_148_proxy가 감소해야 개선으로 판단한다. 분기 악화가 5%를 넘으면 자동 채택하지 않고 불안정 후보로 남긴다. 정확히 5%는 초과가 아니며, 수치상 동률은 상대 허용오차 1e-12로 구분해 B0를 유지한다. B0 분기 W가 0이면 비율 대신 절대 차이를 보고 B1이 양수일 경우 자동 채택하지 않는 예외를 실행 전에 명시했다. 실제 분기 분모는 모두 양수다. 5%는 운영용 기준이며 유의성 문턱이 아니다.", "",
             f"고정 148개소의 분기 기준 초과 항목: {', '.join(comparison['unstable_quarters']) if comparison['unstable_quarters'] else '없음'}.", "",
             "## 실행 검증", "",
             f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}, PyArrow {s['runtime']['pyarrow']}에서 {s['started_at']}~{s['finished_at']}에 실행했다. 학습/예측/점수 재검증과 원본 보존 검사 경과 시간은 {f(s['wall_seconds'])}초다.", "",
             f"- 합성 테스트 {s['synthetic_tests']}개를 통과했다. 시간, 조인, 점수, 목표/예측 처리와 함께 기압 한 열만 제외되는지, 5% 경계, 동률, 악화와 0 분모 처리도 검사했다.",
             "- B0를 재학습하지 않고 저장된 두 라운드/평균 예측을 읽어 평균 일치를 검사했다. B0를 같은 원본 정답과 가격으로 다시 채점해 모든 기존 집합 지표를 재현했다.",
             "- B0/B1의 각 점수 표본 키, 시각 수, 최초/최종 시각, SMP 합, SMP=0 수, 가격 가중 합과 R_local이 정확히 일치했다.",
             "- 라운드별 2,222,352행을 유지했다. B1의 저장된 세 예측을 다시 읽어 253×8,784 격자, 유한/비음수와 산술평균의 일치를 확인했다. 두 모델 해시와 실행 코드 사본도 확인했다.",
             "- 월/분기 합계와 별도 NumPy 집합 누적의 점수가 일치했다. 원본 10개 데이터 파일과 배포 README/baseline.py, 기존 B0 결과/모델/예측과 EDA 계획 파일을 변경하지 않았음을 해시로 확인했다.",
             "- 이번 B1 적합은 2회다. B0/B1의 완료 모델은 합계 4개이며 앞선 B0 저장 오류의 재적합 1회까지 실제 적합 이력은 총 5회다. 이번에는 오류 재시도와 추가 후보 실행이 없다.", "",
             "## 한계 및 다음 단계", "",
             "이번 한 차례 V1 비교와 선택 판단으로 요청 범위를 종료했다. 난수를 바꾼 반복 실험이나 불확실성 구간은 산출하지 않았고 V2 점수도 보지 않았다. 공식 채점기와의 동치를 검증하지 못했으므로 점수는 로컬 W/S다. 정적 용량과 현재 그룹을 과거에 동일 적용한 참조 구성, 익명 좌표와 사후 진단 집단의 한계는 B0와 같다.", "",
             "V1 결과만으로 2026년 A 기압 변화의 생성 원인이나 그 구간의 예측 성능을 판단할 수 없다. 기압 포함/제외의 최종 운영 판단은 확정하지 않았다. 후속 V2 확인은 선택안을 고정해 별도 실행하며, 이번에는 착수하지 않았다. 기본 EDA 종료 및 후보/보류 목록은 그대로 유지했다.", "",
             "## 관련 코드와 로컬 산출물", "",
             "- [B1 실행과 선택 규칙](../scripts/b1_v1_compare.py), [공통 V1 학습](../scripts/b0_v1_baseline.py)",
             "- [보고서 생성](../scripts/report_b1_v1.py), [선택 규칙 테스트](../tests/test_b1_validation.py)",
             "- [입력/코드 해시, 환경과 결과](../data/validation/2026-10-19_B1_V1/summary.json)",
             "- [B0/B1 비교 집계](../data/validation/2026-10-19_B1_V1/comparison.csv), [B1 집합 집계](../data/validation/2026-10-19_B1_V1/group_scores.csv)",
             "- [B1 관측 행 진단](../data/validation/2026-10-19_B1_V1/individual_scores.csv), [발전소별 진단](../data/validation/2026-10-19_B1_V1/individual_sites.csv)",
             "- [B1 평균 예측](../data/validation/2026-10-19_B1_V1/predictions_mean.parquet), [10시 모델](../data/validation/2026-10-19_B1_V1/model_round_10.txt), [17시 모델](../data/validation/2026-10-19_B1_V1/model_round_17.txt)",
             "- [EDA 종료 상태와 보류 목록](EDA_진행계획.md), [일별 로그](logs/2026-10-19.md)", "",
             "데이터가 담긴 출력, 모델과 실행 코드 사본은 Git 제외 경로 data/에만 둔다. 공개 보고서에는 원본/전처리 행과 개별 예측을 복사하지 않았다. 보고서 재생성은 저장된 집계만 사용하며 재학습하지 않는다.", ""]
    text = "\n".join(lines)
    require(not any(c in text for c in ("\u00b7", "\u2014", "\u2013")), "Forbidden punctuation")
    REPORT.write_text(text, encoding="utf-8")
    write_json(OUT / "report_manifest.json", dict(generated_at=now(), generator_sha256=digest(Path(__file__)),
               report_sha256=digest(REPORT), summary_sha256=digest(OUT / "summary.json")))
    print(f"Report generated: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
