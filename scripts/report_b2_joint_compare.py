"""Report the frozen B2 V1/T0 comparison without further model execution."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from b2_joint_compare import OUT, REPORT, ROOT, compare
from report_b0_v1 import f
from s0_validation_check import digest, now, table, write_json
from validation_core import require


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(s["status"] == "passed" and s["model_fit_count"] == 4 and not s["V2_executed"],"Completed scoped comparison required")
    require(digest(OUT/"precheck.json") == s["precheck_sha256"],"Precheck changed")
    require(digest(OUT/"preregistered_report.md") == pre["prereg_report_sha256"],"Preregistered report archive changed")
    for rel,h in pre["code_sha256"].items():
        require(digest(OUT/"executed_code"/Path(rel).name) == h,"Executed code archive changed")
    for rel,h in pre["protected_sha256"].items():
        require(digest(ROOT/rel) == h,"Baseline/EDA changed")
    all_comparisons=[]
    for period in s["periods"]:
        name=period["window"]["name"]
        methods=("B2",) if name == "V1" else ("B0","B2")
        for method in methods:
            folder=OUT/name/method
            modes=(10,17) if method == "B0" else ("joint",)
            matrices=[]
            for mode in (*modes,"mean"):
                flat=pd.read_parquet(folder/f"predictions_{mode}.parquet")
                ntimes=8784 if name == "V1" else 8760
                require(len(flat) == ntimes*253 and not flat.duplicated(["site_id","ts"]).any(),"Serialized grid mismatch")
                mat=flat.pivot(index="ts",columns="site_id",values="kwh")
                require(mat.shape == (ntimes,253) and np.isfinite(mat.to_numpy()).all() and mat.ge(0).all().all(),"Invalid serialized predictions")
                matrices.append(mat)
            expected=sum(matrices[:-1])/(len(matrices)-1)
            pd.testing.assert_frame_equal(expected,matrices[-1])
            fit=json.loads((folder/"fit_summary.json").read_text(encoding="utf-8"))
            for item in fit:
                require(digest(folder/f"model_{item['mode']}.txt") == item["model_sha256"],"Model archive changed")
        decision=compare(period["baseline_scores"],period["joint_scores"],period["cohort_name"])
        require(decision == period["comparison"],"Report decision differs")
        all_comparisons.extend(dict(window=name,**row) for row in decision["rows"])
    pd.DataFrame(all_comparisons).to_csv(OUT/"comparison_all.csv",index=False,encoding="utf-8-sig")
    selected="B2를 유망 후보로 유지" if s["selection"] == "B2_promising_candidate" else "B0 유지"
    def result_rows(period,include_months=False):
        return [[r["sample"],r["period"],f(r["times"],0),f(r["B0_W"]),f(r["B2_W"]),
                 f"{r['change_pct']:+.4f}%" if r["change_pct"] is not None else "정의하지 않음",
                 f(r["B0_mae"]),f(r["B2_mae"])] for r in period["comparison"]["rows"]
                if (r["period"] == "all" or r["period"].startswith("Q") or
                    (include_months and r["period"].startswith("M"))) and
                   (r["sample"] == period["cohort_name"] or r["period"] == "all")]
    headers=["표본","기간","시각 수","B0 W","B2 W","W 변화율","B0 가중 MAE (kWh)","B2 가중 MAE (kWh)"]
    fits=[]
    for p in s["periods"]:
        for item in p["fits"]:
            fits.append([p["window"]["name"],"B2" if item["mode"] == "joint" else "B0",item["mode"],
                         f(item["train_rows"],0),f(item["daylight_rows"],0),len(item["features"]),
                         f(item["any_feature_nan_rows"],0),f(item["fit_seconds"])])
    main_items=[next(r for r in p["comparison"]["rows"] if r["sample"] == p["cohort_name"] and r["period"] == "all") for p in s["periods"]]
    lines=["# B2 공동 입력 검증 설계와 비교","",f"작성: {now()}","",
        "상태: 사전 설계와 가용성 점검을 동결한 뒤 B2 V1 개발 비교와 추가 과거 재현 T0를 완료했다. 모델 적합은 계획한 4회이며 기존 V2, EDA 종료 및 후보/보류 목록을 보존했다. 튜닝, 추가 후보, 최종 학습/평가 예측과 제출은 실행하지 않았다.","",
        "## 질문과 목적","",
        "[D1 잔차 진단](2026-10-20_02_V1_집합잔차와_라운드_진단.md)에서 제안한 두 라운드 공동 입력 B2가 기존 B0보다 집합 오차를 줄이는지 확인한다. 같은 발전소-시각 행의 10시/17시 기상예보 15개씩과 기존 태양/달력/메타 9개로 39개 특징을 만든다. 두 기압 열을 유지하고 예측 모델 한 개를 사용한다. 기준 B0는 라운드별 24개 특징 모델 두 개의 kWh 평균이다.","",
        f"사전 판단 규칙의 결과는 **{selected}**다. V1 연간 대리지표 변화는 **{main_items[0]['change_pct']:+.4f}%**, T0의 변화는 **{main_items[1]['change_pct']:+.4f}%**다. W 변화가 음수면 개선, 양수면 악화다. 각 기간의 표본이 달라 두 기간의 W 절댓값을 직접 비교하지 않는다.","",
        "## 점수 확인 전에 고정한 검증 설계","",
        table(["구분","학습 구간 시작 s 하한","학습 종료 ts 상한 (KST)","예측 대상 s","역할"],
              [["V1","2022-01-01 00:00","2023-12-31 00:00","2024-01-01 이상, 2025-01-01 미만","기존 고정 148개소 개발 비교"],
               ["T0","2022-01-01 00:00","2022-12-31 00:00","2023-01-01 이상, 2024-01-01 미만","추가 과거 강건성 재현"]]),"",
        "V1 B0는 저장된 평균 예측과 점수를 그대로 사용하고 B2만 1회 적합했다. T0에서는 같은 경계로 B0의 10시/17시 각 1회와 B2 1회를 적합했다. 예산은 총 4회로 점수 확인 전에 정했으며 첫 기간 결과와 무관하게 두 기간을 모두 완료했다.","",
        "T0 진단 집단은 기존 고정 148개소 중 T0의 8,760시간 정답이 모두 존재하는 발전소로 정의했다. 두 그룹 모두 존재하고 낮 학습 행이 10,000개 이상이면 실행한다는 가용성 조건도 사전 기록했다. 대상 기간의 정답 가용성을 보고 정한 사후 진단 집단이라는 한계가 있다. 전체 253개 예측은 유지하고 결측 실적을 0으로 채우지 않았다.","",
        "각 기간에서 B2가 연간 W를 줄이고 어느 분기도 B0 대비 5%를 넘게 악화하지 않아야 통과로 판단했다. 두 기간을 모두 통과하면 B2를 유망 후보로 남기고, 하나라도 미개선/동률/불명확/분기 기준 초과이면 B0를 유지한다. B0 W=0인 분기에 B2 W가 양수면 상대 비율 대신 절대 차이를 보고 통과로 처리하지 않는다. 동률 상대 허용오차는 1e-12다. 5%는 운영 기준이며 통계적 유의성 문턱이 아니다.","",
        f"사전 가용성 점검은 {pre['at']}에 끝났고 그때까지 모델 적합과 성능 점수 계산은 0회였다. 설계 원문, 코드, 원본과 기존 결과 파일의 해시를 동결했다. 실행 때 설계 원문과 코드가 같은지 확인했고 최초 설계 문서는 로컬 preregistered_report.md에 보존했다. 이 결과 보고서는 실행 후 정리본이다.","",
        "## 자료 가용성과 실제 수행 방법","",
        table(["T0 사전 점검 항목","확인 결과"],
              [["관측 학습 행",f(pre["T0"]["train_rows"],0)],
               ["낮 학습 행",f(pre["T0"]["daylight_rows"],0)],
               ["학습 기록 발전소",pre["T0"]["train_sites"]],
               ["실제 학습 ts",pre["T0"]["first_train_ts"]+"~"+pre["T0"]["last_train_ts"]],
               ["완전 정답 진단 발전소",pre["T0"]["cohort_sites"]],
               ["그중 학습 기록 발전소",pre["T0"]["cohort_train_sites"]],
               ["진단 G1/G2 발전소",f"{pre['T0']['group_sites']['A']} / {pre['T0']['group_sites']['B']}"],
               ["진단 G1/G2 용량 kW",f"{pre['T0']['group_capacity_kw']['A']:,.3f} / {pre['T0']['group_capacity_kw']['B']:,.3f}"],
               ["전체 253개 공통 정답 시각",pre["T0"]["common253_times"]],
               ["전체 예측 시각/행",f"{pre['T0']['prediction_times']:,} / {pre['T0']['prediction_rows']:,}"],
               ["관측 정답 행",f(pre["T0"]["observed_truth_rows"],0)],
               ["진단 기간 ΣSMP×k",f(pre["T0"]["price_k_sum"])]]),"",
        "T0는 2022년 자료만 사용할 수 있어 기존 약 2년 학습과 다른 1년 학습 조건이다. T0는 V1 학습 구간의 일부를 이전 시점에서 재현한다. 기존 EDA와 V1 학습에 이미 사용한 기간이므로 미관찰 독립 시험이나 미래 일반화 보장은 아니다. T0 집단은 G1 74개소/G2 6개소로 매우 불균형하며 제출 용량 조건을 충족한 전체 구성이 아니다.","",
        "두 라운드 특징을 같은 키/행 순서로 만들고 태양/달력/메타 특징과 목표가 일치하는지 검사한 뒤 열로 결합했다. 두 라운드 예보가 대상일 전일 17시까지 사용 가능하다는 배포 정의를 따른다. 실제 수신 시각을 검증한 것은 아니다. 특징은 float32이며 원래 기상 단위를 유지했다. 결손 예보는 NaN으로 남겨 LightGBM에 전달하고 행을 삭제하지 않았다. 비정시 예보는 반올림하지 않았다.","",
        "학습 목표는 energy/float32 cap을 0~1.2로 자르고 float32로 변환했다. 평가 정답은 원본 그대로 사용했다. 태양 고도 0 이하와 음수 예측은 0, 양의 출력에는 상한을 적용하지 않았다. B2는 공동 모델의 kWh를 직접 사용하고 B0만 두 kWh 출력을 평균했다. 각 모델을 검증 기간 동안 고정했고 실적 시차, 기상 실측과 가격을 입력으로 추가하지 않았다.","",
        "LightGBM 설정은 n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=None으로 기존 B0와 같다. 조기 종료와 설정/난수 탐색은 하지 않았다. 모델 수와 기상 특징 수 변경이 포함된 처리안 전체를 비교한 것이며 특정 정보의 독립적인 인과 효과를 분리한 실험은 아니다.","",
        table(["기간","방법","모드","학습 행","낮 행","특징 수","NaN 특징 행","학습 초"],fits),"",
        "## 결과","",
        "W는 집합별 발전량을 먼저 합산한 후 절대오차에 SMP×k를 곱한 합이다. 종료 시각 11~15시의 k는 3, 나머지는 1이다. 가중 집합 MAE는 W/(2×ΣSMP×k)다. 실제 집합 합계는 모든 구성원 정답이 있는 시각에서만 정의했다. W 변화율은 (B2/B0-1)×100이다.","",
        "### V1 개발 비교","",table(headers,result_rows(s["periods"][0],True)),"",
        "V1의 고정 148개소는 연간 8,784시간이며 전체 253개 공통 표본은 연말 178시간이다. B0의 W/R/S/가격 가중 합과 시각 수가 기존 저장 점수와 일치함을 확인했다. 전체 253개 짧은 표본의 결과로 사전 주 표본이나 판단 규칙을 바꾸지 않았다.","",
        "### T0 과거 재현","",table(headers,result_rows(s["periods"][1],True)),"",
        "T0의 전체 253개 공통 시각이 없으면 common_253 지표는 출력하지 않는다. 기간/정답 부재를 성능 0으로 해석하지 않는다. 같은 기간 안에서 두 방법의 표본, 실제 발전량 가격 수입 R과 가격 가중 합이 정확히 같음을 검사했다. k별 점수와 모든 W/R/S는 로컬 집계에 저장했다.","",
        "## 판단과 한계","",
        f"사전 규칙에 따른 현재 판단은 {selected}다. V1 분기 기준 초과는 {s['periods'][0]['comparison']['unstable_quarters'] or '없음'}, T0는 {s['periods'][1]['comparison']['unstable_quarters'] or '없음'}이다. 기간별 연간 개선과 분기 기준을 함께 적용했으며 결과를 보고 다른 열이나 설정을 추가하지 않았다.","",
        "다만 V1의 전체 253개소 공통 178시간에서는 B2 W가 9.0231% 악화됐다. 이는 제출 대상에 가까운 전체 구성에서 반대 결과가 나온 한계로 남긴다. 짧은 연말 표본을 연간 성능으로 일반화하거나 사전 규칙을 바꾸지는 않지만, B2가 전체 발전소에서도 개선됐다고 결론 내릴 수 없다. T0는 80개소이며 G2가 6개소뿐이어서 이 한계를 해소하지 못한다. T0의 10월은 5.7775% 악화됐으나 사전 기준은 분기 단위였고 분기 악화는 5%를 넘지 않았다. 월별 악화를 숨기거나 이를 보고 문턱을 변경하지 않았다.","",
        "V1은 이미 B0/B1 선택과 잔차 진단에 사용한 개발 구간이다. T0도 기존 EDA/학습에 노출된 과거 자료이므로 추가 독립 검증을 확보했다고 표현하지 않는다. 두 기간의 학습량과 완전 관측 집단이 달라 개선률이 달라도 단일 원인으로 설명할 수 없다. 난수 반복이나 통계적 불확실성 구간은 계산하지 않았다.","",
        "기존 V2는 파일 바이트 해시만 보존 검사했고 정답/예측 내용이나 점수를 새로운 비교에 사용하지 않았다. V2를 보고 B2를 반복 튜닝하지 않았다. 제공 평가 기간의 2026년 기압 변화에 대한 실제 발전량 정답이 없으므로 그 변화 대응 성능은 여전히 미확인이다. 정적 용량/현재 그룹/익명 좌표와 공식 채점기 동치 미확인의 한계도 유지한다.","",
        "이 비교와 판단 정리로 이번 작업을 종료했다. 유망 후보 판단이 나와도 최종 운영 방법이나 제출 성능의 보장은 아니며, 최종 학습 범위와 제출 적합성은 별도 설계가 필요하다. 기존 V1의 B0 선택 이력과 V2 결과는 보존했다. 기본 EDA 종료 및 후보 3개/보류 5개도 변경하지 않았다.","",
        "## 실행 검증","",
        f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']}에 실행했다. 사전/실행 테스트 {s['synthetic_tests']}개를 통과했고 실행 종료 코드 0을 확인했다.","",
        "- 39개 특징, 두 기압 열과 라운드별 값/NaN 유지, 키 순서/목표 불일치 거부, 예보 부재에도 전체 행 보존, T0 학습/자정 마감과 17시간 간격, 선택 규칙을 합성 예제로 검사했다.",
        "- 각 모델의 전체 253개 예측을 유지하고 유한/비음수를 확인했다. 저장 파일을 다시 읽어 V1 2,222,352행, T0 2,216,280행과 키 유일성을 검사하고 B0 두 출력의 평균, B2 단일 출력과 저장된 최종 출력의 일치를 확인했다.",
        "- 독립 집합 오차 계산과 월/분기/k 분할 합계가 원 점수와 일치했다. 같은 기간의 B0/B2 가격 가중 합, 실제 발전량 수입과 점수 표본도 일치했다.",
        "- 모델 파일과 실행 코드 사본 해시, 사전 설계와 가용성 점검 해시를 확인했다. 원본 10개 데이터 및 배포 README/baseline.py, 기존 B0/B1/V2 결과/모델/예측과 EDA 계획 파일을 보존했다.",
        "- 실제 적합은 계획한 4회이며 이번 실행의 오류 재시도는 없다. 앞선 V1/V2와 저장 오류 재적합 이력까지 포함한 누적 적합은 총 11회다. 기존 B0 V1 재학습과 B2 V2 실행은 하지 않았다.","",
        "## 관련 코드와 로컬 산출물","",
        "- [실행 및 사전 가용성 점검](../scripts/b2_joint_compare.py), [보고서 생성](../scripts/report_b2_joint_compare.py), [B2 테스트](../tests/test_b2_validation.py)",
        "- [사전 점검과 입력/코드/설계 해시](../data/validation/2026-10-20_B2_joint/precheck.json), [사전 설계 원문](../data/validation/2026-10-20_B2_joint/preregistered_report.md)",
        "- [실행 요약과 전체 점수](../data/validation/2026-10-20_B2_joint/summary.json), [통합 비교 집계](../data/validation/2026-10-20_B2_joint/comparison_all.csv)",
        "- [V1 B2 예측](../data/validation/2026-10-20_B2_joint/V1/B2/predictions_mean.parquet), [V1 B2 모델](../data/validation/2026-10-20_B2_joint/V1/B2/model_joint.txt)",
        "- [T0 B0 예측](../data/validation/2026-10-20_B2_joint/T0/B0/predictions_mean.parquet), [T0 B2 예측](../data/validation/2026-10-20_B2_joint/T0/B2/predictions_mean.parquet)",
        "- [오늘 작업 로그](logs/2026-10-20.md), [EDA 종료와 보류 목록](EDA_진행계획.md)","",
        "개별 데이터/예측과 모델, 집계 및 실행 코드 사본은 Git 제외 경로 data/에만 저장했다. 공개 보고서에는 원본/전처리 행과 발전소 식별자를 복사하지 않았다. 보고서 재생성은 기존 집계와 저장 파일 검증만 수행하며 재학습하지 않는다.",""]
    text="\n".join(lines)
    require(not any(c in text for c in ("\u00b7","\u2014","\u2013")),"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(generated_at=now(),generator_sha256=digest(Path(__file__)),
        report_sha256=digest(REPORT),summary_sha256=digest(OUT/"summary.json")))
    print(f"Report generated: {REPORT}",flush=True)


if __name__ == "__main__":
    main()
