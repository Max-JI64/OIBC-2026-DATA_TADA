"""Verify persisted four-fit weighting ablation and render results without refitting."""
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from b5_loss_weighting import OUT,REPORT,load_prediction
from b2_joint_compare import T0
from s0_validation_check import ROOT,digest,now,table,write_json
from validation_core import WINDOWS,require


def f(value,digits=2):
    return f"{value:,.{digits}f}" if value is not None else "정의하지 않음"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(s["status"]=="passed" and s["model_fit_count"]==4 and not s["V2_executed"] and not s["final_evaluation_executed"],"Completed weighting comparison required")
    require(digest(OUT/"precheck.json")==s["precheck_sha256"],"Precheck changed")
    require(digest(OUT/"preregistered_report.md")==pre["prereg_report_sha256"] and digest(OUT/"performance_plan.md")==pre["performance_plan_sha256"],"Registered scope changed")
    for rel,h in pre["protected_sha256"].items():require(digest(ROOT/rel)==h,"Previous artifact drift")
    for rel,h in pre["code_sha256"].items():require(digest(OUT/"executed_code"/Path(rel).name)==h,"Executed source archive drift")
    ids=pd.read_csv(ROOT/"OIBC_2026_DATA/01_participant_dataset/sites.csv",encoding="utf-8-sig").site_id
    artifacts={};fits=[];audits=[];rows=[];mae=[];diag=[];contrast=[]
    for period,window in zip(s["periods"],(WINDOWS[0],T0)):
        scope=period["cohort_name"]
        blank=pd.DataFrame(index=window.axis(),columns=ids)
        baseline={(r["sample"],r["period"]):r for r in period["baseline_scores"]}
        b3main=baseline[(scope,"all")]
        main_scores={}
        for method in ("B5","B6"):
            candidate=period["candidates"][method];fit=candidate["fit"];folder=OUT/window.name/method
            model_path=folder/"model_joint.txt";pred_path=folder/"predictions.parquet"
            require(digest(model_path)==fit["model_sha256"],"Weighted model changed")
            booster=lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
            require(booster.num_trees()==600 and booster.feature_name()==fit["features"] and booster.dump_model()["objective"]=="regression_l1","Stored weighted model mismatch")
            pm=json.loads((folder/"prediction_manifest.json").read_text(encoding="utf-8"))
            require(digest(pred_path)==pm["sha256"] and fit["model_sha256"]==pm["model_sha256"],"Prediction manifest drift")
            load_prediction(pred_path,blank)
            artifacts[str(model_path.relative_to(ROOT))]=digest(model_path)
            artifacts[str(pred_path.relative_to(ROOT))]=digest(pred_path)
            fit_disk=json.loads((folder/"fit_summary.json").read_text(encoding="utf-8"))
            require(fit_disk==fit,"Stored fit summary mismatch")
            fits.append([window.name,method,f(fit["train_rows"],0),f(fit["daylight_rows"],0),f(fit["fit_seconds"])])
            w=fit["weights"]
            audits.append([window.name,method,f(w["normalizer"],6),f(w["min"],4),f(w["median"],4),f(w["max"],4),
                f(w["effective_rows"],0),w["zero_rows"],f(w["G1_weight_share"]*100)+"%",f(w["clip_weight_share"]*100,4)+"%"])
            lookup={(r["sample"],r["period"]):r for r in candidate["scores"]}
            main_scores[method]=lookup[(scope,"all")]
            for key,r in lookup.items():
                if key[1]=="all" or (key[0]==scope and (key[1].startswith("Q") or key[1]=="M08")):
                    b=baseline[key]
                    rows.append([window.name,method,key[0],key[1],r["times"],f(b["W_local"]),f(r["W_local"]),f((r["W_local"]/b["W_local"]-1)*100,4)+"%"])
            mae.append([window.name,method,f(b3main["weighted_group_mae_kwh"]),f(lookup[(scope,"all")]["weighted_group_mae_kwh"])])
            for r in candidate["diagnostics"]:
                if r["month"]=="8" and r["k"]=="3":
                    diag.append([window.name,method,r["group"],f(r["B3_W"]),f(r["W"]),f((r["W"]/r["B3_W"]-1)*100,4)+"%",f(r["over_W"]),f(r["under_W"])])
        contrast.append([window.name,f(main_scores["B5"]["W_local"]),f(main_scores["B6"]["W_local"]),
                         f((main_scores["B6"]["W_local"]/main_scores["B5"]["W_local"]-1)*100,4)+"%"])
    selected=s["selection"]
    choice="B3 참조 유지" if selected=="B3" else selected+" 개발 우선 후보"
    ranking=[[r["method"],"통과" if r["passes"] else "불통과",f(r["geometric_ratio"],8),f((r["geometric_ratio"]-1)*100,4)+"%"] for r in s["ranking"]]
    failures=[]
    for p in s["periods"]:
        for m in ("B5","B6"):
            d=p["candidates"][m]["comparison"]
            failures.append([p["window"]["name"],m,"통과" if d["annual_improved"] else "불통과",", ".join(d["unstable_quarters"]) or "없음"])
    lookup_period={p["window"]["name"]:p for p in s["periods"]}
    def change(period,method,part="all",sample=None):
        p=lookup_period[period]
        sample=sample or p["cohort_name"]
        return next(r["change_pct"] for r in p["candidates"][method]["comparison"]["rows"] if r["sample"]==sample and r["period"]==part)
    interpretation=[
        f"B5는 B3 대비 연간 W를 T0에서 {f(-change('T0','B5'),4)}%, V1에서 {f(-change('V1','B5'),4)}% 줄였다. T0의 네 분기 모두 개선됐고 V1은 1분기 {f(change('V1','B5','Q1'),4)}% 악화, 나머지 세 분기 개선이었다. 두 기간의 분기 악화는 사전 한도 5% 안에 있어 B5가 기준을 통과했다.",
        f"V1의 기존 여름 약점도 B3 대비 3분기 {f(-change('V1','B5','Q3'),4)}%, 8월 {f(-change('V1','B5','M08'),4)}% 감소했다. 다만 전체 253개소의 연말 178시간 W는 {f(change('V1','B5',sample='common_253'),4)}% 증가했다. 용량 가중이 모든 계절과 전체 구성에서 개선됐다는 결론은 내리지 않는다.",
        f"B6는 V1 연간 W가 {f(-change('V1','B6'),4)}% 줄었지만 T0는 {f(change('T0','B6'),4)}% 늘었다. 연간 비율 기하평균은 1보다 작아도 두 기간 각각 개선이라는 사전 조건을 만족하지 않아 채택하지 않았다. 짧은 전체 253개소 구간의 큰 개선으로 이 판단을 바꾸지 않는다.",
        "다음 우선 작업은 P-004의 추가 순차 검증 설계와 가용성 점검이다. B3와 B5를 고정해 더 많은 발전소가 관측되는 후반 시점에서도 비교할 수 있는지 먼저 확인하고, 실행 전에 기간/표본/학습 마감/적합 예산/판단 기준을 고정한다. 이미 본 2023~2024년의 추가 분할은 독립 시험으로 부르지 않으며 V2나 제공 평가 구간을 새 선택 자료로 사용하지 않는다. P-002의 기상 조건 잔차 EDA와 P-003의 설정 탐색은 후보로 남긴다.",
    ]
    require(selected=="B5" and all(p["candidates"]["B5"]["comparison"]["passes"] for p in s["periods"]),"Interpretation requires the verified B5 decision")
    lines=["# B5/B6 학습 가중치 비교","",f"작성: {now()}","",
        f"상태: 학습 가중치 두 단계의 V1/T0 4회 적합과 저장 파일 검증을 완료했고 **{choice}**로 판단했다. 기존 B3 출력은 고정 비교 자료로 보존했다. V2 재평가, 최종 평가 예측 갱신과 공식 제출은 하지 않았다.","",
        "## 질문과 목적","",
        "발견일은 2026-10-20이고 부모 작업은 [성능 개선 계획 P-001](성능개선_진행계획.md)이다. 사용자 요청으로 제출 준비 중심의 흐름에서 성능 개선으로 돌아와 첫 가중 손실 실험을 실행했다. 기존 [B3 예측 생성](2026-10-20_09_B3_최종학습과_평가예측_생성.md)은 중간 후보 산출물이며 분석 종료나 최종 방법 확정을 뜻하지 않는다.","",
        "기존 모델은 개별 발전 비율 L1을 학습하지만 대회는 집합 발전량의 SMP×k 가중 절대오차 W를 본다. 용량 비중과 가격/시간대 비중을 학습에 반영하면 실제 집합 W가 줄어드는지 비교했다. [D3](2026-10-20_07_B3_여름철_오차악화_진단.md)의 피크 비용 집중과 [B4](2026-10-20_08_B4_G1피크_과거잔차_보정비교.md)의 보정 불안정을 배경으로 삼았으며 새 원본 EDA를 열지는 않았다.","",
        "## 사전 설계와 실행 범위","",
        f"사전 검사는 {pre['at']}에 끝났고 그때 모델 적합과 비교 점수 계산은 0회였다. 코드와 설계 원문/성능 계획, 원본과 이전 결과 해시를 고정했다. 실행 예산은 B5/B6를 V1/T0에서 각각 1회, 총 4회이며 첫 결과와 무관하게 모두 완료했다.","",
        "| 방법 | 낮 학습 행의 정규화 전 가중치 | 변경한 항목 |",
        "|---|---|---|",
        "| B3 참조 | 동일 비중 | 기존 저장 예측 재사용 |",
        "| B5 | float32 cap을 float64로 바꾼 값 | sample_weight |",
        "| B6 | B5의 cap × 학습 시각 SMP × 종료 시각 k | sample_weight |","",
        "가중치는 해당 실험의 낮 학습 행 평균으로 나눠 평균 1로 정규화했다. 검증이나 평가 기간의 가격으로 학습 가중치를 구하지 않았다. B6의 과거 SMP는 학습 손실 가중에만 사용하고 예측 특징에는 넣지 않았다. 가격 0인 행은 가중치 0으로 남기며 행을 삭제하지 않았다.","",
        "V1은 s >= 2022-01-01, ts <= 2023-12-31 00:00 학습으로 2024년을 예측하며 고정 148개소를 채점한다. T0는 s >= 2022-01-01, ts <= 2022-12-31 00:00 학습으로 2023년을 예측하며 기존 완전 관측 80개소를 채점한다. 시간은 KST이고 기존 관측 행 순서/낮 행/특징/NaN 행 수/학습 경계를 B3와 대조했다.","",
        "공동 기상 30개와 태양/달력/메타 9개, 기압 두 열과 float32 처리, 목표 clip(energy/cap,0,1.2), 야간/음수 예측 0 및 상한 미적용 처리를 유지했다. objective=regression_l1, 600개 트리, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1과 random_state=None의 기존 동작을 유지했다. 새로운 조기 종료/난수 탐색/보정은 없다.","",
        table(["기간","후보","관측 학습 행","낮 학습 행","학습 초"],fits),"",
        "## 목표와 실제 가중치의 관계","",
        "이상적인 cf 잔차 e에서 집합 W=Σq×|Σcap×e| <= Σq×Σcap×|e|다. B6는 개별 절대오차의 상한 쪽에 가중을 맞추지만 집합 잔차 상쇄를 직접 최적화하지 않는다. 학습 목표 clip, 낮 제한과 원본 정답 유지 때문에 실제 평가 W와 학습 손실의 등식도 성립하지 않는다.","",
        table(["기간","후보","정규화 전 평균","최소","중앙값","최대","가중치 집중 유효 행","가중 0행","G1 가중 비중","목표 clip 행의 가중 비중"],audits),"",
        "가중치 집중 유효 행은 (Σw)^2/Σw²이며 독립 표본 수나 통계적 유의성의 추정치가 아니다. 서로 다른 용량/가격 비중이 어느 정도 집중되는지 요약한다. clip 행은 원본 cf가 0 미만 또는 1.2 초과인 낮 학습 행이다. 해당 원본 실적을 로컬 채점에서 자르지 않았다.","",
        "## 동일 표본 비교 결과","",
        "W는 낮을수록 좋다. 집합 안에서 잔차를 합친 뒤 절댓값을 취하고 SMP×k를 곱한다. 서로 다른 G1/G2 잔차를 합쳐 상쇄하지 않는다. 정답/구성/기간이 같으면 R은 같아 ΔS_local=-0.10×ΔW다. 변화율은 정확도 상승이나 정산금의 같은 비율 변화가 아니다.","",
        table(["기간","후보","표본","세부 기간","시각 수","B3 W","후보 W","변화율"],rows),"",
        "전체 253개소 common_253은 V1 연말 178시간이며 연간 148개소와 다른 표본이다. T0에는 전체 공통 정답 시각이 없어 해당 지표를 출력하지 않았다. 개발 표본은 제출 용량 조건을 충족한 전체 구성 성능을 대표하지 않는다.","",
        "가중 집합 MAE=W/(2×ΣSMP×k), 단위 kWh를 함께 기록한다. 같은 표본에서는 W와 순위가 같다.","",
        table(["기간","후보","B3 가중 집합 MAE","후보 가중 집합 MAE"],mae),"",
        "### 용량 가중에 가격/시간대 가중을 더한 비교","",table(["기간","B5 W","B6 W","B6 대 B5"],contrast),"",
        "두 후보의 설정은 같은 가중치 정규화 계약 아래 학습 행별 비중만 다르다. 한 번씩의 적합 결과로 용량 또는 가격의 인과 효과를 확정하거나 다른 설정에도 같은 결과를 보장하지 않는다.","",
        "### 8월 집합별 k=3","",table(["기간","후보","집합","B3 W","후보 W","변화율","후보 과대 비용","후보 과소 비용"],diag),"",
        "이 표는 여름 약점 진단이다. 해당 월에 맞춰 학습 가중치나 계수를 다시 정하지 않았다. 나머지 집합/k 진단과 모든 월별 W/R/S는 로컬 실행 요약에 남겼다.","",
        "## 사전 기준에 따른 판단","",
        "각 후보는 B3 대비 두 기간 연간 W 감소와 어느 분기도 5% 초과 악화 없음이 채택 조건이다. 상대 동률 허용오차는 1e-12다. 통과 후보가 둘이면 두 연간 W 비율의 기하평균이 작은 쪽을 우선하며 동률은 B5를 먼저 둔다. 조건을 통과하지 못하면 연간 두 기간의 평균 개선만으로 채택하지 않는다.","",
        table(["기간","후보","연간 개선","5% 초과 악화 분기"],failures),"",
        table(["후보","두 기간 기준","연간 W 비율 기하평균","기하평균 변화"],ranking),"",
        f"이번 판단은 **{choice}**다. 이는 이미 본 T0/V1에서의 개발 선택이며 독립 시험이나 공식 평가의 성능 보장이 아니다. 전체 253개소 연간 성능과 새로운 미래 정답은 여전히 없다.","",
        "### 결과 해석과 다음 우선순위","",
        *[v for paragraph in interpretation for v in (paragraph,"")],
        "## 한계와 후속 작업","",
        "T0와 V1은 앞선 선택과 잔차 진단에 반복 사용했다. 이번에도 독립 검증이라고 부르지 않는다. T0의 발전소/그룹 불균형, 짧은 연말 전체 집합 표본과 난수 반복 미실행의 한계를 유지한다. V2는 보존 해시만 확인하고 내용/점수를 새 후보 선택에 사용하지 않았다.","",
        "B5/B6는 개별 발전소의 가중 L1이며 집합 직접 손실 학습이 아니다. 가격과 용량 가중이 강해도 집합 상쇄, 기록 가용성 차이, 여름 기상 조건과 미래 기압 변화 문제가 해결됐다고 해석하지 않는다.","",
        "이 비교를 끝낸 뒤 [성능 개선 계획](성능개선_진행계획.md)의 다음 후보 우선순위를 검토한다. 필요한 추가 EDA는 기존 EDA 완료 이력과 별도로 질문/범위/종료 기준을 등록해 진행한다. 이번 비교가 좋거나 나쁘다는 이유로 제출 준비나 최종 평가 예측 재생성으로 자동 이동하지 않는다.","",
        "## 실행 검증","",
        f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']}에 실행했고 종료 코드 0을 확인했다. 가중치 계약 테스트 {s['synthetic_tests']}개를 통과했다.","",
        "- 용량 L1과 kWh L1의 단위 대응, k의 종료 10/11시 경계, 중복 대상 시각의 가격 정렬, 가격 0행 유지와 가격 결손/전체 가격 0/용량 0 거부 및 집중도/clip 집계를 합성 예제로 검사했다.",
        "- 실제 B3의 학습 관측/낮/NaN 행 수, 시간 경계, 특징 순서를 유지했다. 가중치를 해당 행의 cap/시각에서 만들어 목표와 같은 순서로 전달했다.",
        "- 기존 B3의 동일 표본 W/R/S/가격 가중 합/시각 수를 재현했다. 각 후보의 별도 집합 W, 월/분기/k 분할 합계 및 진단의 집합/방향/k 합계가 일치했다.",
        "- 저장된 네 모델을 문자열로 다시 불러 L1/600개 트리/특징 순서를 검사했다. V1 후보별 2,222,352행과 T0 후보별 2,216,280행의 저장 예측을 다시 읽어 전체 키/시각/유한/비음수를 확인했다.",
        "- 원본 10개 데이터와 배포 README/baseline.py 및 보존 대상 기존 모델/예측/진단과 B3 고정 산출물 및 EDA 계획의 해시가 유지됐다. 보고서의 저장 검증은 모델 재적합/새 추론을 하지 않는다.",
        "- 이번 실제 새 기초 모델 적합은 4회이고 오류 재시도는 없다. 이전 이력까지 누적 적합은 19회이며 B4 보정계수 2개는 별도 이력이다. V2/최종 평가 새 실행과 공식 제출은 없다.","",
        "공식 scoring.py와 문제정의서가 없어 채점기 동치는 미확인이다. 현재 값은 제공 식을 구현한 로컬 집합 W/S이며 대회 제출 점수가 아니다.","",
        "## 관련 코드와 로컬 산출물","",
        "- [실행/가중치 계약](../scripts/b5_loss_weighting.py), [가중치 테스트](../tests/test_loss_weighting.py), [저장 검증/보고서 생성](../scripts/report_b5_loss_weighting.py)",
        "- [사전 검사/코드/입력 해시](../data/validation/2026-10-20_B5_B6_loss_weighting/precheck.json), [사전 설계 원문](../data/validation/2026-10-20_B5_B6_loss_weighting/preregistered_report.md)",
        "- [전체 실행 요약과 점수](../data/validation/2026-10-20_B5_B6_loss_weighting/summary.json)",
        "- [V1 B5 예측](../data/validation/2026-10-20_B5_B6_loss_weighting/V1/B5/predictions.parquet), [V1 B6 예측](../data/validation/2026-10-20_B5_B6_loss_weighting/V1/B6/predictions.parquet)",
        "- [T0 B5 예측](../data/validation/2026-10-20_B5_B6_loss_weighting/T0/B5/predictions.parquet), [T0 B6 예측](../data/validation/2026-10-20_B5_B6_loss_weighting/T0/B6/predictions.parquet)",
        "- [일별 로그](logs/2026-10-20.md), [EDA 종료와 추가 질문](EDA_진행계획.md)","",
        "원본/전처리 행과 개별 발전소 예측/식별자는 공개 보고서에 복사하지 않았다. 모델/예측/집계/코드 및 설계 사본은 Git 제외 data/에 보관한다. 코드와 중간보고서 및 로그만 로컬 커밋하며 push하지 않는다.",""]
    text="\n".join(lines)
    require("\u00b7" not in text,"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(generated_at=now(),report_sha256=digest(REPORT),summary_sha256=digest(OUT/"summary.json"),
        generator_sha256=digest(Path(__file__)),verified_artifacts=artifacts))
    print(f"Weighting comparison report generated: {REPORT}",flush=True)


if __name__=="__main__":main()
