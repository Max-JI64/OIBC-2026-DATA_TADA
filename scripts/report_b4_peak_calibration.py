"""Verify persisted B4 outputs and report the single frozen correction comparison."""
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from b4_peak_calibration import OUT, B3, REPORT, C0, adjust, direction, load_prediction
from s0_validation_check import ROOT, SOURCE, digest, now, table, write_json
from validation_core import WINDOWS, require, weights
from b2_joint_compare import T0


def f(value,digits=2):
    return f"{value:,.{digits}f}" if value is not None else "정의하지 않음"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(s["status"]=="passed" and s["model_fit_count"]==1 and s["calibration_fit_count"]==2 and not s["V2_executed"],"Completed B4 required")
    require(digest(OUT/"precheck.json")==s["precheck_sha256"],"Precheck changed")
    require(digest(OUT/"preregistered_report.md")==pre["prereg_report_sha256"],"Preregistered design changed")
    for rel,h in pre["protected_sha256"].items(): require(digest(ROOT/rel)==h,"Prior artifact changed")
    for rel,h in pre["code_sha256"].items():
        require(digest(OUT/"executed_code"/Path(rel).name)==h,"Executed code archive changed")
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig")
    capacity=sites.set_index("site_id").capacity_kw
    g1=sites.loc[sites.current_group.eq("A"),"site_id"].tolist()
    g2=sites.loc[sites.current_group.eq("B"),"site_id"].tolist()
    model=OUT/"C0/B3_calibration/model_joint.txt"
    require(digest(model)==s["calibration_base_fits"][0]["model_sha256"],"Calibration model changed")
    booster=lgb.Booster(model_str=model.read_text(encoding="utf-8"))
    require(booster.num_trees()==600 and booster.dump_model()["objective"]=="regression_l1","Unexpected calibration model")
    hashes={str(model.relative_to(ROOT)):digest(model)}
    # Reconstruct deterministic postprocessing only; no model inference or coefficient fitting.
    for p,window in zip(s["periods"],(T0,WINDOWS[0])):
        blank=pd.DataFrame(index=window.axis(),columns=sites.site_id)
        base=load_prediction(B3/window.name/"B3/predictions_mean.parquet",blank)
        path=OUT/window.name/"predictions_calibrated.parquet"
        saved=load_prediction(path,blank)
        expected=adjust(base,g1,capacity,p["calibration"]["coefficient"])
        pd.testing.assert_frame_equal(saved,expected)
        pd.testing.assert_frame_equal(base[g2],saved[g2])
        pd.testing.assert_frame_equal(base.loc[weights(base.index)==1],saved.loc[weights(saved.index)==1])
        hashes[str(path.relative_to(ROOT))]=digest(path)
    c0blank=pd.DataFrame(index=C0.axis(),columns=sites.site_id)
    c0pred=OUT/"C0/B3_calibration/predictions_mean.parquet"
    load_prediction(c0pred,c0blank)
    hashes[str(c0pred.relative_to(ROOT))]=digest(c0pred)
    pd.testing.assert_frame_equal(load_prediction(OUT/"C0/B3_calibration/predictions_joint.parquet",c0blank),load_prediction(c0pred,c0blank))
    coefficients=[]
    for scope,cal in zip(pre["scopes"],s["calibrations"]):
        coefficients.append([cal["window"],scope["calibration_window"]["start"][:4],cal["sites"],f(cal["capacity_kw"],3),
            f(cal["coefficient"],8),*[f(v,8) for v in cal["quarter_medians"]],"통과" if cal["stable"] else "불통과"])
    comparison_rows=[];mae=[]
    for p in s["periods"]:
        ref={(r["sample"],r["period"]):r for r in p["baseline_scores"]}
        for r in p["corrected_scores"]:
            if r["period"]=="all" or (r["sample"]==p["cohort_name"] and (r["period"].startswith("Q") or r["period"]=="M08")):
                old=ref[(r["sample"],r["period"])]
                comparison_rows.append([p["window"]["name"],r["sample"],r["period"],r["times"],f(old["W_local"]),f(r["W_local"]),
                    f((r["W_local"]/old["W_local"]-1)*100,4)+"%"])
                if r["sample"]==p["cohort_name"] and r["period"]=="all":
                    mae.append([p["window"]["name"],f(old["weighted_group_mae_kwh"]),f(r["weighted_group_mae_kwh"])])
    peak=[]
    for p in s["periods"]:
        a,b=p["August_G1_peak_before"],p["August_G1_peak_after"]
        peak.append([p["window"]["name"],f(a["W"]),f(b["W"]),f((b["W"]/a["W"]-1)*100,4)+"%",
            f(b["over_W"]-a["over_W"]),f(b["under_W"]-a["under_W"])])
    gate_names={"stable":"과거 계수의 연도/분기 방향 안정성","annual_and_quarters":"두 기간 연간 개선 및 분기 악화 5% 이내",
                "V1_August_G1_peak_improved":"V1 8월 G1/k=3 W 감소","V1_Q3_not_worse":"V1 3분기 전체 W 비증가"}
    gates=[[gate_names[k],"통과" if v else "불통과"] for k,v in s["decision"].items()]
    retained=s["selection"]=="retain_B3"
    choice="B4 보정 미채택, B3 유지" if retained else "B4 개발 후보 채택"
    annual={p["window"]["name"]:next(r for r in p["comparison"]["rows"] if r["sample"]==p["cohort_name"] and r["period"]=="all") for p in s["periods"]}
    lines=["# B4 G1 피크 과거 잔차 보정 비교","",f"작성: {now()}","",
        f"상태: 사용자 요청에 따라 D3-H1의 보정안 하나를 실행했고 **{choice}**로 종료했다. 보정용 기초 모델 적합은 1회, 과거 연간 보정계수는 2개다. 기존 V1/T0 B3를 재학습하지 않았고 V2 재평가와 최종 평가 예측/제출은 하지 않았다. 기본 EDA 종료와 선택 3개/보류 5개도 유지했다.","",
        "## 질문과 목적","",
        "발견일은 2026-10-20이고 부모 작업은 [D3 여름철 진단](2026-10-20_07_B3_여름철_오차악화_진단.md)의 D3-H1이다. G1/k=3 과대예측을 학습 구간의 별도 미래 예측 잔차로 추정한 단일 보정이 개선하는지 비교했다. V1 8월 정답에서 보정값을 직접 구하지 않았다.","",
        f"B3 대비 연간 W 변화는 T0 {annual['T0']['change_pct']:+.4f}%, V1 {annual['V1']['change_pct']:+.4f}%다. W 감소가 개선이며 정확도 또는 정산금의 같은 비율 증가를 뜻하지 않는다.","",
        "## 사전 고정과 실제 방법","",
        f"사전 점검은 {pre['at']}에 종료했고 모델 적합/보정계수 적합/성능 계산은 모두 0회였다. 설계 원문과 코드/원본/기존 결과 해시를 고정했다. 보정 연도와 집단은 성능을 보지 않고 시간 경계와 정답/기초 모델 학습 기록 가용성으로 정했다.","",
        "| 적용 대상 | 보정용 기초 모델 학습 | 보정 잔차 정답 구간 | 실제 실행 |",
        "|---|---|---|---|",
        "| T0, 2023년 | s >= 2021-01-01, ts <= 2021-12-31 00:00 | 2022-01-01 01:00~2022-12-31 00:00 | C0 모델 1회 적합, 2022년 예측 |",
        "| V1, 2024년 | s >= 2022-01-01, ts <= 2022-12-31 00:00 | 2023-01-01 01:00~2023-12-31 00:00 | 기존 T0 B3 예측 재사용 |","",
        "모든 시각은 KST다. 보정 잔차는 기초 모델 학습 이후의 정답이며 각 적용 대상 B3의 학습 마감 이전에 끝난다. 보정 연도 마지막 하루는 제외해 8,736시간으로 고정했다. 대상 검증의 고정 G1 가운데 보정 기간 전체 정답과 기초 모델 학습 기록이 있는 발전소만 사용했다.","",
        f"C0는 관측 {pre['C0_train_rows']:,}행, 낮 {pre['C0_daylight_rows']:,}행, {pre['C0_train_sites']}개소로 학습했다. 보정 집단은 2022년 31개소/2,125.645 kW, 2023년 74개소/5,999.445 kW다. 각 보정 연도의 분기별 양의 가격 가중 k=3 시각 수는 450/455/460/455개다. 사전의 최소 20개소, 1,000 kW와 분기 400개 기준을 통과했다. 이는 가용성 문턱이며 통계적 신뢰도 보장은 아니다.","",
        "C0도 기존 B3와 같은 39개 공동 특징, 두 기압 열, float32 처리와 목표 clip 및 태양 고도/음수 예측 처리를 사용했다. objective=regression_l1, 600개 트리와 나머지 LightGBM 설정을 유지했다. SMP를 모델 입력이나 새 가중 학습으로 추가하지 않았고 기존 비정시 B 예보 92행 제외 규칙도 유지했다.","",
        "### 단일 보정식","",
        "보정 집단의 용량 합을 C, 집합 예측을 P, 실적을 A라고 할 때 z=(A-P)/C다. 연간 k=3 시각에서 q=SMP×k로 가중한 z의 중앙값 b 하나를 계산했다. 누적 가중 50%에 처음 도달한 값을 선택했다. b의 부호는 강제하지 않았으며 음수면 하향, 양수면 상향 보정이다.","",
        "보정은 전체 253개소 중 G1 134개소의 합에 적용했다. P'=max(P+b×G1 전체 용량 20,465.655 kW,0)로 바꾸고 G1 각 발전소에 P'/P를 곱했다. 기존 합이 0이면 용량 비중을 사용했다. 그 뒤 기존 T0 80개소/V1 148개소의 고정 부분집합을 그대로 채점했다. 따라서 부분집합의 실제 보정량은 그 시각 예측이 전체 G1에서 차지하는 비중에도 좌우된다. G2와 k=1은 바꾸지 않았다. 이 처리는 사전 동결 코드에 기록한 방식이며 결과를 본 뒤 변경하지 않았다. 월/계절별 계수와 축소율, 상한, 새 특징과 다른 보정안을 탐색하지 않았다.","",
        "## 보정계수와 안정성","",
        table(["적용 대상","보정 연도","집단 수","용량 kW","연간 b","Q1 중앙값","Q2 중앙값","Q3 중앙값","Q4 중앙값","분기 방향 안정성"],coefficients),"",
        "b와 분기 중앙값은 용량으로 나눈 발전 비율 단위다. 네 분기의 값은 진단용이며 적용 계수는 연간 b 하나다. 두 보정 연도의 연간 계수와 각 네 분기의 중앙값 모두 같은 비영 방향이어야 사전 안정성 조건을 통과한다. 0의 방향 허용오차는 cf 1e-8이다.","",
        "실제로 2022년 잔차는 상향, 2023년 잔차는 하향 보정을 요구했다. 2022년은 상반기와 하반기의 분기 중앙값 부호가 달랐고 2023년은 3분기가 다른 분기와 반대였다. 이 두 과거 집단에서는 일정한 방향의 보정이 반복된다고 확인하지 못했다. 시기와 모델 학습량, 집단이 함께 달라 단일 원인으로 설명하지 않는다.","",
        "## 기존 B3와 비교 결과","",
        "기존 T0 80개소와 V1 148개소 표본, 시간과 가격을 그대로 사용했다. 전체 253개소 공통 표본은 V1의 기존 연말 178시간만 있다. T0에는 전체 공통 시각이 없어 해당 지표를 출력하지 않았다.","",
        table(["검증","표본","기간","시각 수","B3 W","B4 W","변화율"],comparison_rows),"",
        "계산은 집합 안에서 잔차를 합친 뒤 절댓값을 취하고 SMP×k를 곱한다. 서로 다른 G1/G2의 잔차를 합쳐 상쇄하지 않는다. 같은 정답/표본의 R이 같아 ΔS_local=-0.10×ΔW다. 공식 채점기와 동치를 확인한 점수는 아니다.","",
        "가중 집합 MAE는 W/(2×ΣSMP×k)이며 단위는 kWh다. 같은 표본 안에서 W와 순위가 같다.","",
        table(["검증","B3 가중 집합 MAE","B4 가중 집합 MAE"],mae),"",
        "### 8월 G1/k=3","",table(["검증","B3 W","B4 W","변화율","과대 비용 증감","과소 비용 증감"],peak),"",
        "과대/과소 비용은 해당 시각의 양/음 잔차 절댓값에 가격 가중을 곱한 합이다. 8월은 수정 목표의 진단이며 계수를 맞추는 자료가 아니다. G2와 k=1은 저장 파일에서도 그대로임을 확인했다.","",
        "V1의 8월 G1/k=3 W는 4.1962% 줄었고 전체 8월 W도 2.4820% 줄었다. 반면 T0는 네 분기 모두 W가 증가해 연간 개선 조건을 충족하지 못했다. 분기 악화는 모두 5% 이내지만 연간 조건과 안정성 조건은 별개다. V1 연말 전체 253개소의 짧은 표본에서도 W가 0.9508% 늘었다. 여름 한 구간의 개선만으로 보정을 채택하지 않는다.","",
        "## 사전 기준에 따른 판단","",table(["고정한 조건","결과"],gates),"",
        "자료 적합성, 계수 안정성, 두 기간 연간 W 감소 및 모든 분기 5% 초과 악화 없음, V1 8월 G1/k=3 개선과 V1 3분기 전체 비증가를 모두 충족해야 채택하도록 정했다. 동률 상대 허용오차는 1e-12다. 5%와 방향 안정성은 운영용 선택 기준이며 유의성 검정 문턱이 아니다.","",
        f"판단은 {choice}다. "+("D3-H1의 이 보정안은 채택하지 않고 이번 한 가지 비교로 종료한다. 보정 부호를 뒤집거나 연도/월을 골라 다시 적합하지 않는다. 기존 B3 최종 학습 설계를 유지한다." if retained else "개발 후보 판단이며 독립 시험 통과를 뜻하지 않는다. 최종 실행 전에 기존 최종 학습 설계에 보정 추정 범위를 별도로 반영해야 하며 이번에는 해당 수정과 최종 적합을 하지 않았다."),"",
        "## 한계와 다음 단계","",
        "보정용 기초 모델과 적용 대상 B3는 학습량과 집단이 다르다. 특히 C0는 초기에 기록된 31개소만 학습했으며 31개소의 용량 정규화 편향을 더 큰 G1으로 옮기는 가정이 있다. 각 계수의 분기 진단은 같은 과거 자료에서의 안정성을 보는 것이며 그 자체가 독립 성능 검증은 아니다.","",
        "T0/V1은 이미 개발과 진단에 사용했고 T0의 과거 예측은 뒤의 V1 보정 자료로도 사용했다. 시간 순서는 지켰지만 두 비교를 독립 시험으로 표현하지 않는다. 전체 253개소의 여름/연간 정답 부족과 2026년 기압 변화 대응 미확인도 남는다.","",
        "최종 학습과 평가 예측은 이번 범위 밖이다. 다음 작업은 채택 판단을 반영한 최종 학습/로컬 예측 생성이며 공식 제출은 별개다. EDA 종료와 선택/보류 목록을 변경하지 않았다.","",
        "## 실행 검증","",
        f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']}에 실행했고 종료 코드 0을 확인했다. 합성 테스트 {s['synthetic_tests']}개를 통과했다.","",
        "- 시간 경계와 계수 자료 가용성을 사전 고정했고 실제 1개 기초 모델, 과거 연간 계수 2개만 적합했다. 현재 8월 잔차로 계수를 추정하지 않았다.",
        "- 기존 B3의 두 기간 W/R/S/시각 수/가격 가중 합을 재현했고 새 점수의 월/분기/k 합계와 별도 집합 계산을 검사했다.",
        "- 저장된 C0 전체 253개소 예측 2,216,280행과 V1/T0 보정 격자 2,222,352/2,216,280행을 다시 읽어 키/유한/비음수를 확인했다. 보정식을 재구성해 저장 결과와 일치했고 G2/k=1 불변 및 G1 합계 보존을 검사했다.",
        "- C0 모델을 UTF-8 문자열로 다시 읽어 objective=regression_l1과 600개 트리 및 파일 해시를 검사했다. 보고서 검증의 보정식 재구성은 저장 값의 확인이며 모델 추론/계수 재적합은 아니다.",
        "- 원본 10개 데이터와 배포 README/baseline.py, 기존 모델/예측/진단과 P1 최종 설계 및 EDA 계획의 해시가 유지됐다. V2는 내용 해석 없이 파일 해시만 확인했다.",
        "- 이번 새 기초 모델 적합은 1회이고 오류 재시도는 없다. 앞선 모델 적합 이력까지 누적 실제 적합은 14회이며 별도로 보정계수 2개를 이번에 추정했다.","",
        "## 관련 코드와 로컬 산출물","",
        "- [B4 실행/사전 점검](../scripts/b4_peak_calibration.py), [보정 계약 테스트](../tests/test_peak_calibration.py), [보고서 생성/저장 검증](../scripts/report_b4_peak_calibration.py)",
        "- [사전 점검과 코드/자료 해시](../data/validation/2026-10-20_B4_peak_calibration/precheck.json), [사전 설계 원문](../data/validation/2026-10-20_B4_peak_calibration/preregistered_report.md)",
        "- [실행 요약과 모든 점수](../data/validation/2026-10-20_B4_peak_calibration/summary.json), [C0 기초 모델](../data/validation/2026-10-20_B4_peak_calibration/C0/B3_calibration/model_joint.txt)",
        "- [V1 보정 예측](../data/validation/2026-10-20_B4_peak_calibration/V1/predictions_calibrated.parquet), [T0 보정 예측](../data/validation/2026-10-20_B4_peak_calibration/T0/predictions_calibrated.parquet)",
        "- [일별 로그](logs/2026-10-20.md), [기존 최종 학습 설계](2026-10-20_06_B3_최종학습과_평가절차_설계.md), [EDA 종료/보류](EDA_진행계획.md)","",
        "모델, 예측과 계수 및 요약/실행 사본은 Git 제외 data/에 보관했다. 공개 문서에는 원본/전처리 행이나 개별 발전소 예측/식별자를 복사하지 않았다.",""]
    text="\n".join(lines)
    require("\u00b7" not in text,"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(generated_at=now(),report_sha256=digest(REPORT),
        generator_sha256=digest(Path(__file__)),summary_sha256=digest(OUT/"summary.json"),output_sha256=hashes))
    print(f"B4 report generated: {REPORT}",flush=True)


if __name__=="__main__":main()
