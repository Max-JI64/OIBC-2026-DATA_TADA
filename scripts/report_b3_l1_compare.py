"""Build the B3 objective-only comparison report from completed artifacts."""
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from b3_l1_compare import OUT,REPORT,ROOT
from report_b0_v1 import f
from s0_validation_check import digest,now,table,write_json
from validation_core import require


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(s["status"]=="passed" and s["model_fit_count"]==2 and not s["V2_executed"],"Completed scoped B3 required")
    require(digest(OUT/"precheck.json")==s["precheck_sha256"] and digest(OUT/"preregistered_report.md")==pre["prereg_report_sha256"],"Preregistration changed")
    for rel,h in pre["protected_sha256"].items():
        require(digest(ROOT/rel)==h,"Protected result changed")
    for rel,h in pre["code_sha256"].items():
        require(digest(OUT/"executed_code"/Path(rel).name)==h,"Executed code archive changed")
    for p in s["periods"]:
        folder=OUT/p["window"]["name"]/"B3"
        mats=[]
        for name in ("joint","mean"):
            flat=pd.read_parquet(folder/f"predictions_{name}.parquet")
            n=8784 if p["window"]["name"]=="V1" else 8760
            require(len(flat)==253*n and not flat.duplicated(["site_id","ts"]).any(),"Saved grid mismatch")
            mat=flat.pivot(index="ts",columns="site_id",values="kwh")
            require(mat.shape==(n,253) and np.isfinite(mat.to_numpy()).all() and mat.ge(0).all().all(),"Invalid saved predictions")
            mats.append(mat)
        pd.testing.assert_frame_equal(*mats)
        require(digest(folder/"model_joint.txt")==p["fits"][0]["model_sha256"],"Model changed")
        booster=lgb.Booster(model_str=(folder/"model_joint.txt").read_text(encoding="utf-8"))
        require(booster.dump_model()["objective"] in ("regression_l1","l1") and booster.num_trees()==600,"Saved objective mismatch")
    comparisons=[]
    def summary_rows(period,monthly=False):
        lookup={label:{(r["sample"],r["period"]):r for r in period[label+"_scores"]} for label in ("B0","B2","B3")}
        output=[]
        for key,r3 in lookup["B3"].items():
            if not (key[1]=="all" or (key[0]==period["cohort_name"] and (key[1].startswith("Q") or (monthly and key[1].startswith("M"))))):
                continue
            r0,r2=lookup["B0"][key],lookup["B2"][key]
            output.append([key[0],key[1],r3["times"],f(r0["W_local"]),f(r2["W_local"]),f(r3["W_local"]),
                           f"{(r3['W_local']/r2['W_local']-1)*100:+.4f}%",f"{(r3['W_local']/r0['W_local']-1)*100:+.4f}%"])
        for key,r in lookup["B3"].items():
            comparisons.append(dict(window=period["window"]["name"],sample=key[0],period=key[1],
                B0_W=lookup["B0"][key]["W_local"],B2_W=lookup["B2"][key]["W_local"],B3_W=r["W_local"]))
        return output
    v1rows=summary_rows(s["periods"][0],True)
    t0rows=summary_rows(s["periods"][1],True)
    pd.DataFrame(comparisons).to_csv(OUT/"comparison_all.csv",index=False,encoding="utf-8-sig")
    headers=["표본","기간","시각 수","B0 W","B2 W","B3 W","B3 대 B2","B3 대 B0"]
    choice="B3 개발 기준 통과" if s["selection"]=="B3_development_pass" else "B3 미채택, 기존 B0 참조/B2 후보 유지"
    annual=[]
    for p in s["periods"]:
        annual.append(next(r for r in p["vs_B2"]["rows"] if r["sample"]==p["cohort_name"] and r["period"]=="all"))
    diag=[]
    for g in ("G1","G2"):
        for method in ("B0","B2","B3"):
            r=next(r for r in s["yearend"] if r["group"]==g and r["method"]==method)
            diag.append([g,r["method"],f(r["W_all"]),f(r["W_fixed"]),f(r["W_rest"]),
                         f(r["weighted_bias_all"]),f(r["cross_cancellation"])])
    totals={m:sum(r["W_all"] for r in s["yearend"] if r["method"]==m) for m in ("B0","B2","B3")}
    components={m:{k:sum(r[k] for r in s["yearend"] if r["method"]==m)
                    for k in ("W_fixed","W_rest","cross_cancellation")} for m in ("B2","B3")}
    mae_rows=[]
    for p in s["periods"]:
        rows=[]
        for method in ("B0","B2","B3"):
            row=next(r for r in p[method+"_scores"] if r["sample"]==p["cohort_name"] and r["period"]=="all")
            rows.append(f(row["weighted_group_mae_kwh"]))
        mae_rows.append([p["window"]["name"],p["cohort_sites"],*rows])
    fits=[[p["window"]["name"],f(p["fits"][0]["train_rows"],0),f(p["fits"][0]["daylight_rows"],0),
           len(p["fits"][0]["features"]),f(p["fits"][0]["fit_seconds"])] for p in s["periods"]]
    lines=["# B3 L1 손실 비교","",f"작성: {now()}","",
        "상태: B2 공동 입력의 학습 손실만 L1으로 바꾼 B3를 V1/T0에서 각 한 번 적합하고 비교를 완료했다. V2 재평가와 추가 설정 탐색은 하지 않았다. EDA 종료와 후보 3개/보류 5개 및 기존 결과를 보존했다.","",
        "## 질문과 목적","",
        "[D2 악화 분해](2026-10-20_04_B2_연말_전체집합_악화분해.md)의 후속 후보로, 개별 cf의 절대오차 손실을 쓰면 [B2](2026-10-20_03_B2_공동입력_검증설계와_비교.md)보다 실제 집합 W가 줄어드는지 확인한다. 개별 L1은 SMP×k 가중 집합 W와 같지 않으므로 최종 판단은 집합 점수로 했다.","",
        f"사전 규칙의 결과는 **{choice}**다. 같은 표본의 W 변화율은 아래 표에 기록했다. W 감소는 개선이며 정산금 S의 같은 비율 증가를 뜻하지 않는다.","",
        f"B2 대비 연간 W는 V1에서 {annual[0]['change_pct']:+.4f}%, T0에서 {annual[1]['change_pct']:+.4f}% 변했다. 이번 결과에서는 B3를 개발 우선 후보로 두고 B0 참조와 B2 비교 결과를 보존한다. 최종 방법 확정과 독립 시험 통과를 뜻하지 않는다.","",
        "## 점수 확인 전에 고정한 설계와 실제 방법","",
        "실행 예산은 V1 B3 1회와 T0 B3 1회, 총 2회다. B0/B2는 저장 예측으로 재채점하고 재학습하지 않았다. 두 기간을 결과와 무관하게 모두 실행했다. objective=regression_l1 외에는 B2의 설정을 바꾸지 않았다. 표본/경계, 39개 특징 순서, 두 기압 열, 자료형, 학습 목표와 출력 처리를 유지했다.","",
        "V1은 학습 s >= 2022-01-01, ts <= 2023-12-31 00:00 KST에서 2024년 8,784시간을 예측한다. 주 개발 표본은 기존 고정 148개소다. T0는 학습 s >= 2022-01-01, ts <= 2022-12-31 00:00 KST로 2023년 8,760시간을 예측하며 기존 완전 관측 80개소를 유지한다. T0의 G1/G2는 74/6개소로 불균형하고 약 1년 학습 조건이다. 두 표본 모두 전체 253개소를 대표하거나 제출 용량 조건을 충족한 구성은 아니다.","",
        "두 기간 모두 B2 대비 연간 W가 줄고 분기별 W 악화가 5%를 넘지 않아야 개발 기준 통과로 판단한다. 동률은 상대 허용오차 1e-12이며 B2 분기 W=0에서 B3가 양수면 불통과다. 한 조건이라도 충족하지 못하면 B3를 채택하지 않고 B0 참조/B2 후보 상태를 유지한다. 5%는 운영 기준이며 유의성 문턱이 아니다. 같은 178시간의 전체 253개소와 두 집단/G1/G2 점수 및 상쇄는 별도 진단으로 기록하고 사후 문턱을 바꾸지 않는다.","",
        f"사전 검사는 {pre['at']}에 점수/학습 없이 완료했고 설계, 코드와 입력 해시를 동결했다. 사전 원문은 로컬 preregistered_report.md에 보존했다. 실행 코드에서 B2와의 차이가 objective 하나임을 검사했다.","",
        "LightGBM은 n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1이며 random_state=None을 유지했다. sample_weight와 alpha를 지정하지 않았고 조기 종료나 난수 탐색도 없다. 각 모델을 대상 연도 전체에서 고정했다.","",
        "기상 15개를 라운드별 열로 결합하고 태양/달력/메타 9개를 한 번 사용했다. 학습 목표 energy/float32 cap을 0~1.2로 자르고 float32로 변환했다. 원본 정답은 자르지 않았다. 태양 고도 0 이하/음수 예측은 0이며 양의 예측 상한은 없다. 결손 예보는 NaN으로 유지하고 행을 삭제하지 않았다. 두 라운드와 키/목표/공통 특징이 같은지 검사했고 실제 B2의 특징/학습 행/낮 행/마감/NaN 행 수가 일치했다.","",
        table(["기간","학습 행","낮 학습 행","특징 수","학습 초"],fits),"",
        "## 결과","",
        "W=Σ집합,시각 SMP×k×|집합 예측-집합 정답|이다. k는 KST 종료 11~15시가 3, 나머지가 1이다. 같은 기간/구성에서는 R이 같으므로 ΔS_local=-0.10×ΔW다. 아래 변화율은 W 기준이며 MAE나 정확도 비율이 아니다.","",
        "공식 채점 코드가 없어 제공 안내의 식을 구현한 로컬 값이다. 공식 제출 점수와의 일치는 확인하지 않았다.","",
        "W를 2×Σ시각 SMP×k로 나눈 가중 집합 MAE도 함께 기록한다. 단위는 kWh이고 같은 표본에서는 W와 순위가 같다. 개별 발전소 MAE 및 대회의 정산금 S와는 구분한다.","",
        table(["기간","발전소 수","B0 가중 집합 MAE","B2 가중 집합 MAE","B3 가중 집합 MAE"],mae_rows),"",
        "### V1 개발 비교","",table(headers,v1rows),"",
        "연간 고정 148개소와 연말 전체 253개소 178시간은 기간과 구성원이 다르다. 이번 연말 표본에서도 개선했지만 짧은 겨울 구간이므로 전체 253개소의 연간 성능은 여전히 미확인이다. V1 Q3는 B2 대비 2.4389% 악화했고 특히 8월은 7.8283% 악화했다. 월별 결과도 남겼으며 사전의 분기 기준을 월 기준으로 바꾸지는 않았다.","",
        "### T0 과거 재현","",table(headers,t0rows),"",
        "T0에는 전체 253개소 공통 정답 시각이 없어 그 지표를 정의하지 않았다. 월/분기/k와 모든 W/R/S 집계는 로컬 요약 및 comparison_all.csv에 보존했다.","",
        "### 동일 연말 178시간의 방향과 상쇄","",
        table(["집합","방법","전체 W","148개소 W","105개소 W","전체 가중 편향 (kWh)","집단 사이 상쇄"],diag),"",
        f"두 집합 W를 합친 B3의 변화는 B2 대비 {(totals['B3']/totals['B2']-1)*100:+.4f}%, B0 대비 {(totals['B3']/totals['B0']-1)*100:+.4f}%다. 편향은 예측-정답을 SMP×k로 가중한 평균이다. 집단 사이 상쇄 C를 사용해 W전체=W148+W105-C를 검사했고 서로 다른 G1/G2의 잔차는 상쇄하지 않았다. 이 구간에 맞춰 보정값을 적합하지 않았다.","",
        f"B2 대비 148개소의 두 집합 W 합은 {(components['B3']['W_fixed']/components['B2']['W_fixed']-1)*100:+.4f}%, 나머지 105개소는 {(components['B3']['W_rest']/components['B2']['W_rest']-1)*100:+.4f}% 변했다. 집단 사이 상쇄는 {f(components['B2']['cross_cancellation']-components['B3']['cross_cancellation'])} 줄어 전체 W 개선을 일부 상쇄했다. 두 집단의 W 감소가 상쇄 감소의 불리한 영향을 웃돌았다. 다만 나머지 105개소 중 G1의 W는 증가했으므로 모든 하위 집단이 개선했다고 말할 수 없다.","",
        "같은 연말 표본에서 G1/G2 가중 편향은 B2의 -135.47/-274.58kWh에서 B3의 -14.00/-129.89kWh로 바뀌어 과소예측이 줄었다. 이 관측을 다른 계절이나 일반적인 L1의 효과로 확대하지 않는다.","",
        "## 판단과 한계","",
        f"{choice}. V1에서 B2 대비 분기 기준 초과는 {s['periods'][0]['vs_B2']['unstable_quarters'] or '없음'}, T0에서는 {s['periods'][1]['vs_B2']['unstable_quarters'] or '없음'}이다. 연간 개선과 분기 조건을 함께 적용했다. 이미 확인한 기존 후보 판단의 이력은 변경하지 않았다.","",
        "L1을 사용했다는 사실만으로 대회 손실을 최적화한 것은 아니다. 개별 발전소의 조건부 중심, 발전소 잔차 상쇄와 집합 W는 다른 대상이다. 특히 L1이 과소예측을 줄인다고 가정하지 않았다. 이 실험은 목표함수 하나의 변경 결과이며 가격/용량 가중이나 집합 직접 학습의 효과를 확인하지 않았다.","",
        "V1은 후보 선택과 잔차 진단에 사용했고 T0도 이미 EDA와 학습에 노출된 과거 구간이다. 독립 시험이나 최종 일반화 보장으로 표현하지 않는다. 난수 반복과 통계적 불확실성 구간도 계산하지 않았다. V2는 파일 바이트 해시로 보존만 확인하고 내용/점수를 새 비교에 사용하지 않았다. 제공 평가 구간의 발전량 정답이 없어 2026년 기압 변화 대응 성능은 여전히 미확인이다.","",
        "이번 두 적합과 사전 판단 기록으로 종료한다. 결과를 보고 후속 후보나 설정을 자동으로 추가하지 않았다. 최종 방법 확정/학습 설계와 최종 평가 예측/제출은 미실행이며 EDA 종료와 보류 목록을 유지한다.","",
        "다음 단계로는 B3의 현재 설정을 우선 후보로 놓고 최종 학습 범위와 평가 절차를 정리하는 작업을 제안한다. 이미 확인한 V2를 반복해서 튜닝 근거로 쓰지 않고, 미관측 검증 구간을 확보할 수 있는지와 확보하지 못할 때의 해석 한계를 먼저 정해야 한다. 이 절차는 아직 설계하거나 실행하지 않았다.","",
        "## 실행 검증","",
        f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']}에 실행했다. 사전/실행 테스트 {s['synthetic_tests']}개를 통과했고 실행 종료 코드 0을 확인했다.","",
        "- 손실함수만 바뀌는지와 기존 목표/예측 처리, 공동 입력/결손/키 순서/시점 및 판단 규칙을 검사했다.",
        "- B0/B2를 재채점해 기존 모든 W/R/S/가격 가중 합/시간 수를 재현했다. 같은 기간의 B2/B3 점수 표본과 정답 기반 R_local이 동일했다.",
        "- 별도 집합 오차 계산과 월/분기/k 분할 합계를 검사했다. 저장된 B3 두 기간의 전체 253개 예측 격자와 키 유일성/유한/비음수, 단일 출력과 최종 파일의 일치를 확인했다.",
        "- 저장된 모델을 문자열로 다시 불러 objective=regression_l1 및 600개 트리를 확인했다. 모델 파일과 실행 코드/사전 설계 사본의 해시도 검사했다.",
        "- 원본 10개 데이터 파일과 배포 README/baseline.py 및 기존 B0/B1/B2/V2, D1/D2 관련 보존 결과와 EDA 계획의 해시가 유지됐다. 모델 적합은 계획한 2회이며 오류 재시도는 없다. 앞선 이력까지 누적 실제 적합은 13회다.","",
        "## 관련 코드와 로컬 산출물","",
        "- [B3 실행](../scripts/b3_l1_compare.py), [공동 학습 계약](../scripts/b2_joint_compare.py), [보고서 생성](../scripts/report_b3_l1_compare.py), [B3 테스트](../tests/test_b3_validation.py)",
        "- [사전 검사/코드/입력 해시](../data/validation/2026-10-20_B3_L1/precheck.json), [사전 설계 원문](../data/validation/2026-10-20_B3_L1/preregistered_report.md)",
        "- [실행 요약과 점수](../data/validation/2026-10-20_B3_L1/summary.json), [비교 집계](../data/validation/2026-10-20_B3_L1/comparison_all.csv), [연말 방향/상쇄](../data/validation/2026-10-20_B3_L1/yearend.csv)",
        "- [V1 모델](../data/validation/2026-10-20_B3_L1/V1/B3/model_joint.txt), [V1 예측](../data/validation/2026-10-20_B3_L1/V1/B3/predictions_mean.parquet)",
        "- [T0 모델](../data/validation/2026-10-20_B3_L1/T0/B3/model_joint.txt), [T0 예측](../data/validation/2026-10-20_B3_L1/T0/B3/predictions_mean.parquet)",
        "- [오늘 로그](logs/2026-10-20.md), [EDA 종료와 보류 목록](EDA_진행계획.md)","",
        "개별 데이터/예측, 모델과 집계 및 코드 사본은 Git 제외 경로 data/에 저장했다. 공개 보고서에 원본/전처리 행과 개별 발전소 식별자를 복사하지 않았다. 보고서 재생성은 저장된 결과만 사용하며 재학습/새 예측은 하지 않는다.",""]
    text="\n".join(lines)
    require(not any(c in text for c in ("\u00b7","\u2014","\u2013")),"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(generated_at=now(),generator_sha256=digest(Path(__file__)),report_sha256=digest(REPORT),summary_sha256=digest(OUT/"summary.json")))
    print(f"Report generated: {REPORT}",flush=True)


if __name__=="__main__":
    main()
