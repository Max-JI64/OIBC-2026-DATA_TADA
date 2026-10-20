"""Verify the completed local B3 bundle and generate its execution report."""
import json
from pathlib import Path
import sys

import lightgbm as lgb
import pandas as pd

from b3_final_fit_predict import OUT, REPORT, validate_outputs
from s0_validation_check import ROOT, SOURCE, digest, now, table, write_json
from validation_core import require


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(summary["status"]=="passed" and summary["model_fit_count"]==1 and not summary["calibration_applied"],"Completed uncorrected B3 required")
    require(not any(summary[k] for k in ("performance_scores_computed","V2_executed","official_submission_executed")),"Scope mismatch")
    require(digest(OUT/"precheck.json")==summary["precheck_sha256"],"Precheck changed")
    require(digest(OUT/"preregistered_report.md")==pre["prereg_report_sha256"],"Scope archive changed")
    for rel,h in pre["protected_sha256"].items():require(digest(ROOT/rel)==h,"Previous artifact changed")
    for rel,h in pre["code_sha256"].items():require(digest(OUT/"executed_code"/Path(rel).name)==h,"Executed source archive changed")
    for name,h in summary["output_sha256"].items():require(digest(OUT/name)==h,"Completed output changed")
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig")
    price=pd.read_csv(SOURCE/"smp.csv",encoding="utf-8-sig")
    composition=pd.read_csv(OUT/"composition.csv",encoding="utf-8-sig")
    forecast=pd.read_csv(OUT/"forecast.csv",encoding="utf-8-sig",float_precision="round_trip")
    flat=pd.read_parquet(OUT/"site_predictions.parquet")
    check=validate_outputs(composition,forecast,flat,sites,price)
    require(check==summary["output_validation"],"Saved output validation changed")
    for name,lines in (("composition.csv",254),("forecast.csv",17521)):
        require((OUT/name).read_bytes().startswith(b"\xef\xbb\xbf"),"Missing UTF-8 BOM")
        require(len((OUT/name).read_text(encoding="utf-8-sig").splitlines())==lines,"CSV line count mismatch")
    booster=lgb.Booster(model_str=(OUT/"model_joint.txt").read_text(encoding="utf-8"))
    require(booster.num_trees()==600 and booster.feature_name()==pre["features"] and booster.dump_model()["objective"]=="regression_l1","Saved model contract changed")
    month_rows=[]
    for info in summary["monthly_predictions"]:
        path=OUT/"months"/(info["month"]+".parquet")
        require(digest(path)==info["sha256"],"Monthly archive changed")
        month_rows.append([info["month"],f"{info['rows']:,}",f"{info['feature_nan_rows']:,}"])
    require(len(month_rows)==12 and sum(r["rows"] for r in summary["monthly_predictions"])==2216280,"Monthly count mismatch")
    if "--check-only" in sys.argv:
        print("Final saved bundle passed; no fit, inference, score or submission",flush=True)
        return
    fit=summary["fit"]
    sizes=[[name,f"{(OUT/name).stat().st_size:,}",summary["output_sha256"][name]]
           for name in ("composition.csv","forecast.csv","site_predictions.parquet","model_joint.txt")]
    lines=["# B3 최종 학습과 평가 예측 생성","",f"작성: {now()}","",
        "상태: 보정 없는 B3 모델 1회 적합과 평가 1년의 로컬 예측 생성을 완료하고 저장 파일 검증을 통과했다. 공식 제출은 하지 않았다. 평가 정답이 없어 W/S/MAE 성능을 계산하지 않았으며 V2 재평가도 없다. EDA 종료와 선택 3개/보류 5개를 유지했다.","",
        "## 질문과 목적","",
        "[최종 학습 설계](2026-10-20_06_B3_최종학습과_평가절차_설계.md)에 따라 모델을 고정하고 전체 발전소의 평가 격자와 참조 집합 출력을 생성했다. [B4 과거 잔차 보정](2026-10-20_08_B4_G1피크_과거잔차_보정비교.md)이 미채택돼 보정/혼합을 적용하지 않은 B3를 사용했다. 이번 결과는 로컬 생성 및 형식 검증 완료이며 대회 성능 확인과 구분한다.","",
        "## 실제 학습 범위와 처리","",
        "모든 시각은 KST이며 대상 날짜는 s=ts-1시간이다. 학습은 s >= 2023-06-01, ts <= 2025-05-31 00:00이다. 첫 예측 시점 2025-05-31 17:00보다 17시간 앞서 마감했고 모델은 평가 1년 동안 고정했다. 실제 수신 지연을 확인한 마감이 아니라 기존 보수적 정책이다.","",
        table(["항목","실행 확인"],[["관측 학습 행",f"{fit['train_rows']:,}"],["낮 학습 행",f"{fit['daylight_rows']:,}"],
            ["관측 대상 발전소","253개소"],["학습 ts 시작",fit["train_ts_min"]],["학습 ts 끝",fit["train_ts_max"]],
            ["낮 학습 중 특징 NaN 포함 행",f"{fit['any_feature_nan_rows']:,}"],["모델 적합 시간",f"{fit['fit_seconds']:.2f}초"]]),"",
        "P1의 관측 3,421,168행 및 낮 1,707,268행과 일치했다. 학습에는 기존 V2 기간의 일부 관측 실적도 들어갔지만 각 최종 예측의 이용 가능 시점 이전 자료를 적합에 사용한 것이다. V2를 재채점해 방법을 고르는 절차는 수행하지 않았다. 마감 이후 6,072행과 시작 이전 965,326행은 이전 설계대로 제외했다.","",
        "학습 특징을 24개 월 묶음으로 조립해 원래 관측 행 순서를 나타내는 키로 다시 정렬했다. 특징 39개의 순서와 목표 행 정렬을 확인한 뒤 LightGBM을 적합했다. 월별 조립이 학습 표본 순서를 바꾸지 않도록 했다.","",
        "각 라운드의 기상 15개를 결합하고 태양/달력/메타 9개를 한 번 사용했다. 기압 두 열을 포함하며 특징과 학습 목표는 기존 float32 처리를 유지했다. 목표는 energy/float32 cap을 0~1.2로 자르고 태양 고도 > 0인 관측 행만 학습했다. 없는 실적은 만들지 않았고 낮의 원본 0도 별도로 삭제하지 않았다.","",
        "objective=regression_l1, n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1을 유지했다. random_state=None의 기존 동작을 바꾸지 않았다. sample_weight, 조기 종료, 새 변수, 보정과 추가 설정 탐색은 없다.","",
        "학습은 train 예보만 조인하고 그 자료를 해제한 뒤 평가에는 test 예보만 사용했다. 실측 기상과 SMP, 시차 실적을 모델 특징으로 추가하지 않았다. 원본 비정시 B 학습 예보 92행은 반올림 없이 정시 연결에서 제외했고 학습 목표/예측 격자는 결손 예보 때문에 삭제하지 않았다.","",
        "## 평가 예측 생성","",
        "예측 대상 s는 2025-06-01 00:00 이상, 2026-06-01 00:00 미만이다. 종료 시각 ts는 제공 smp.csv의 2025-06-01 01:00~2026-06-01 00:00 8,760개와 일치한다. 저장된 모델을 문자열로 다시 불러 월별로 추론했으며 중간에 재학습하지 않았다.","",
        "태양 고도 0 이하의 예측은 0, 음수 cf 출력도 0으로 만든 뒤 float32 cap을 곱했다. 양의 출력 상한과 B4 보정은 추가하지 않았다. 두 라운드는 단일 모델의 공동 입력이며 라운드별 두 모델을 따로 적합하거나 예측 평균을 내지 않았다.","",
        table(["대상 월 s","발전소 예측 행","특징 NaN 포함 행"],month_rows),"",
        "NaN 행 수는 두 라운드 공동 특징 중 하나라도 값이 비어 있는 행 수다. 원본 기상 지점의 결손 행 수나 발전량 누락 수와 다르다. LightGBM의 기존 결손 처리를 사용했고 그 행도 최종 격자에 남겼다. 제공 가격 0인 18시간도 두 집합 출력에서 유지했다.","",
        "## 고정 구성과 생성 파일","",
        table(["집합","발전소 수","설비용량 kW"],[["G1",134,"20,465.655"],["G2",119,"22,189.715"],["전체",253,"42,655.370"]]),"",
        "current_group A/B를 G1/G2로 대응시켰고 평가 기간 전체에 같은 구성을 적용한다. 모든 발전소를 정확히 한 번 포함하고 각 집합 20,000 kW 이상을 만족한다. 구성 최적화는 수행하지 않았다.","",
        table(["파일","열","데이터 행","헤더 포함 행"],[["composition.csv","site_id, group","253","254"],
            ["forecast.csv","group, ts, forecast_kwh","17,520","17,521"],
            ["site_predictions.parquet","site_id, ts, kwh","2,216,280","해당 없음"]]),"",
        "CSV는 UTF-8 BOM으로 저장했고 ts는 +09:00이 있는 문자열이다. 발전소별 kWh를 같은 구성의 집합별로 합산했다. 저장한 CSV/Parquet를 다시 읽어 전체 시간축과 키, 유한/비음수 및 집합 합계 보존을 확인했다.","",
        f"집합별 저장 값과 별도 NumPy 발전소 합계의 최대 절대 차이는 {check['max_group_sum_abs_difference']:.12g} kWh다. 비교 허용오차는 상대 1e-12, 절대 1e-7 kWh이며 두 집합을 합친 예측 총합도 발전소 총합과 일치했다. 이 검사는 정답과의 오차가 아닌 출력 사이의 보존 검사다.","",
        "모든 산출물은 로컬 data/final/2026-10-20_B3/에 있다. 아래 두 CSV가 구성/예측 파일이며 모델과 발전소별 Parquet는 재현 및 로컬 검증 자료다.","",
        "- [composition.csv](../data/final/2026-10-20_B3/composition.csv)",
        "- [forecast.csv](../data/final/2026-10-20_B3/forecast.csv)",
        "- [발전소별 예측](../data/final/2026-10-20_B3/site_predictions.parquet), [고정 모델](../data/final/2026-10-20_B3/model_joint.txt)","",
        table(["파일","크기 bytes","SHA256"],sizes),"",
        "## 검증과 실행 이력","",
        f"사전 검사는 {pre['at']}에 학습/예측/성능 계산 없이 완료했다. 정규 CPython {summary['runtime']['python']}, LightGBM {summary['runtime']['lightgbm']}, pandas {summary['runtime']['pandas']}, NumPy {summary['runtime']['numpy']}에서 {summary['started_at']}~{summary['finished_at']}에 실행했고 종료 코드 0을 확인했다. 합성 계약 테스트 {summary['synthetic_tests']}개를 통과했다.","",
        "- 39개 특징과 두 라운드/기압 유지, 결손 예보에도 격자 보존, 키/목표/행 순서, 학습 마감과 L1 설정 계약을 확인했다.",
        "- 출력 테스트는 누락/중복, 잘못된 구성 용량, 타임존 없는 시각, 가격 0인 시각의 NaN과 집합 합계 불일치를 거부했다.",
        "- 실제 학습 표본/낮 행 수와 경계를 기존 설계에 대조했고 모델을 저장한 뒤 다시 읽어 L1 손실, 600개 트리와 특징 순서를 확인했다.",
        "- 두 CSV의 UTF-8 BOM과 실제 줄 수, 전체 발전소/집합 예측 격자, 원본 메타 구성과 정확한 용량 및 가격 시간축을 저장 파일에서 재검증했다.",
        "- 원본 10개 데이터 파일과 배포 README/baseline.py, 모든 보존 대상 이전 모델/예측/진단 및 P1 최종 설계와 EDA 계획의 해시가 유지됐다. V2는 내용 해석 없이 파일 해시로 보존만 확인했다.",
        f"- 이번 모델 적합은 1회이고 오류 재시도는 없다. 이미 완료된 월별 예측을 재사용한 횟수는 {summary['reused_prediction_months']}회다. 앞선 이력까지 누적 실제 기초 모델 적합은 15회이며 B4의 보정계수 적합은 별도 이력이다.","",
        "학습 후 모델은 즉시 UTF-8 문자열로 저장했다. 완료 최종 실행은 덮어쓰지 않으며 예측 도중 중단된 경우에만 --resume-predictions로 저장 모델과 월별 해시를 확인해 이어갈 수 있다. 이번 실행에는 재개가 필요하지 않았다. 보고서 검증은 저장 값을 읽는 작업으로 모델 적합/새 추론을 하지 않는다.","",
        "저장 파일만 다시 확인하려면 보고서 생성 코드의 --check-only 모드를 사용한다. 이 모드는 학습/추론/성능 계산과 보고서 재작성을 하지 않는다. 실행 시 정규 CPython 3.13과 프로젝트의 승인된 Start-Process 경로를 따른다.","",
        "## 성능과 공식 제출의 한계","",
        "**평가 1년의 발전량 정답이 없어 이 최종 모델의 W, S_local과 MAE를 계산하지 않았다.** 새 독립 미래 시험도 확보하지 못했다. 파일 검증 통과를 예측 성능 통과나 실제 운영 성과로 표현하지 않는다.","",
        "기존 개발 비교의 B3 개선, 여름 일부 악화와 B4 미채택 근거는 각 보고서에 보존했다. 최종 모델은 학습량과 발전소 가용성이 달라졌으며 개발 개선률을 그대로 평가 성능으로 옮겨 쓰지 않는다. 2026년 기압 변화 대응 성능도 미확인이다.","",
        "현재 배포 폴더에는 공식 scoring.py와 문제정의서가 없어 공식 채점기와의 동치를 검증하지 못했다. 이번에는 로컬 배포 README의 열/행/시각/구성 규칙을 검사했다. 실제 제출 시 최신 공식 규격과 제출 절차를 확인해야 하며, 이 작업에서 웹 조회나 공식 제출은 하지 않았다.","",
        "이번 범위는 최종 학습, 로컬 파일 생성과 검증 및 보고서 정리로 종료한다. 다음 단계는 필요할 때 두 CSV의 제출 전 정본 규격 대조이며 외부 제출은 별도 작업이다. EDA 보류 질문과 추가 모델 탐색을 재개하지 않았다.","",
        "## 관련 코드와 재현 기록","",
        "- [최종 실행](../scripts/b3_final_fit_predict.py), [출력/행 순서 테스트](../tests/test_final_outputs.py), [저장 결과 검증과 보고서 생성](../scripts/report_b3_final.py)",
        "- [사전 검사와 입력/코드 해시](../data/final/2026-10-20_B3/precheck.json), [사전 범위 원문](../data/final/2026-10-20_B3/preregistered_report.md)",
        "- [학습 요약](../data/final/2026-10-20_B3/fit_summary.json), [전체 실행과 출력 해시](../data/final/2026-10-20_B3/summary.json)",
        "- [일별 로그](logs/2026-10-20.md), [EDA 종료/보류 목록](EDA_진행계획.md)","",
        "모델, 원본/전처리 데이터, 예측과 기타 출력 파일은 Git 제외 data/에만 저장했다. 공개 보고서에는 원본/전처리 행이나 개별 발전소 예측/식별자를 복사하지 않았다. 코드와 중간보고서 및 로그만 로컬 커밋한다.",""]
    text="\n".join(lines)
    require("\u00b7" not in text,"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(generated_at=now(),report_sha256=digest(REPORT),
        summary_sha256=digest(OUT/"summary.json"),generator_sha256=digest(Path(__file__))))
    print(f"Final B3 report generated: {REPORT}",flush=True)


if __name__=="__main__":main()
