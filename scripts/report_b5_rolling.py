"""Verify the four stored replay outputs and write the P-004 report, without fits."""
from pathlib import Path
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from b5_rolling_compare import OUT,REPORT,REPLAYS,read_json,source_hashes,protected,code_paths,runtime,inputs,decision
from b2_joint_compare import scores
from b5_loss_weighting import load_prediction
from s0_validation_check import ROOT,SOURCE,digest,now,table,write_json
from validation_core import require,truth_matrix


def f(x,n=2):return f"{x:,.{n}f}" if x is not None else "정의하지 않음"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s=read_json(OUT/"summary.json");pre=read_json(OUT/"precheck.json")
    require(s["status"]=="passed" and s["model_fit_count"]==4 and not s["V2_executed"] and not s["final_evaluation_executed"],"Completed replay required")
    require(s["runtime"]==runtime() and s["precheck_sha256"]==digest(OUT/"precheck.json"),"Replay summary drift")
    require(pre["source_sha256"]==source_hashes() and pre["protected_sha256"]==protected(),"Protected input drift")
    require(pre["availability_sha256"]==digest(OUT/"availability.json"),"Availability drift")
    require(pre["report_sha256"]==digest(OUT/"preregistered_report.md") and pre["plan_sha256"]==digest(OUT/"performance_plan.md"),"Preregistration drift")
    for p in code_paths():
        rel=str(p.relative_to(ROOT));require(digest(p)==pre["code_sha256"][rel] and digest(OUT/"executed_code"/p.name)==pre["code_sha256"][rel],"Executed replay code drift")
    sites,gen,price=inputs();ids=sites.site_id
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    fits=[];results=[];mae=[];sub=[];group=[];gate=[];artifacts={}
    for p,w in zip(s["periods"],REPLAYS):
        av=p["availability"];truth=truth_matrix(gen,ids,w.axis())
        score_maps={}
        for method in ("B3","B5"):
            value=p["methods"][method];fit=value["fit"];folder=OUT/w.name/method
            require(fit==read_json(folder/"fit_summary.json"),"Replay fit summary changed")
            path=folder/"model.txt";require(digest(path)==fit["model_sha256"],"Stored replay model changed")
            booster=lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            require(booster.num_trees()==600 and booster.feature_name()==fit["contract"]["features"] and booster.dump_model()["objective"]=="regression_l1","Replay booster mismatch")
            pm=read_json(folder/"prediction_manifest.json");pred=folder/"predictions.parquet"
            require(pm["sha256"]==digest(pred) and pm["model_sha256"]==fit["model_sha256"],"Stored replay predictions changed")
            prediction=load_prediction(pred,truth)
            recomputed=scores(truth,prediction,groups,price,av["cohort_ids"],"complete_quarter")
            original={(r["sample"],r["period"]):r for r in value["scores"]}
            require(set(original)=={(r["sample"],r["period"]) for r in recomputed},"Stored replay score scope drift")
            for r in recomputed:
                old=original[(r["sample"],r["period"])]
                for key in ("W_local","R_local","S_local","price_k_sum","weighted_group_mae_kwh","times"):
                    require(np.isclose(r[key],old[key],rtol=1e-12,atol=1e-5),"Stored replay score mismatch")
                require(r["first_ts"]==old["first_ts"] and r["last_ts"]==old["last_ts"],"Stored replay score time drift")
            del prediction
            artifacts[str(path.relative_to(ROOT))]=digest(path);artifacts[str(pred.relative_to(ROOT))]=digest(pred)
            c=fit["contract"]
            fits.append([w.name,method,f(c["train_rows"],0),f(c["daylight_rows"],0),f(c["nan_rows"],0),f(fit["fit_seconds"])])
            score_maps[method]={(r["sample"],r["period"]):r for r in value["scores"]}
            del booster
        for r in p["comparison"]["rows"]:
            if r["sample"]=="complete_quarter" and (r["period"]=="all" or r["period"].startswith("M")) or r["sample"]=="common_253" and r["period"]=="all":
                key=(r["sample"],r["period"]);a=score_maps["B3"][key];b=score_maps["B5"][key]
                results.append([w.name,r["sample"],r["period"],a["times"],f(r["B3_W"]),f(r["B5_W"]),f(r["change_pct"],4)+"%"])
                if r["period"]=="all":mae.append([w.name,r["sample"],f(a["weighted_group_mae_kwh"]),f(b["weighted_group_mae_kwh"])])
        for name in ("fixed_148","added_complete"):
            a=next((r for r in p["methods"]["B3"]["diagnostics"] if r.get("sample")==name),None)
            b=next((r for r in p["methods"]["B5"]["diagnostics"] if r.get("sample")==name),None)
            if a is not None and b is not None:
                sub.append([w.name,name,a["sites"],f(a["W_local"]),f(b["W_local"]),f((b["W_local"]/a["W_local"]-1)*100,4)+"%"])
        for g in ("G1","G2"):
            for k in ("all","1","3"):
                a=next(r for r in p["methods"]["B3"]["diagnostics"] if r.get("group")==g and r["k"]==k)
                b=next(r for r in p["methods"]["B5"]["diagnostics"] if r.get("group")==g and r["k"]==k)
                group.append([w.name,g,k,f(a["W"]),f(b["W"]),f((b["W"]/a["W"]-1)*100,4)+"%",f(b["over_W"]),f(b["under_W"])])
        cp=p["comparison"]
        require(cp==decision(p["methods"]["B3"]["scores"],p["methods"]["B5"]["scores"],"complete_quarter"),"Replay decision drift")
        gate.append([w.name,"통과" if cp["improved"] else "불통과",", ".join(cp["unstable_months"]) or "없음","통과" if cp["passes"] else "불통과"])
    chosen=s["selection"]
    require(chosen==("B5" if all(p["comparison"]["passes"] for p in s["periods"]) else "B3"),"Overall replay decision drift")
    judgment="B5 개발 우선 유지" if chosen=="B5" else "B5의 추가 검증 미통과, B3 우선 참조로 복귀"
    interpretation=[]
    for p in s["periods"]:
        av=p["availability"];cp=p["comparison"]
        annual=next(r for r in cp["rows"] if r["sample"]=="complete_quarter" and r["period"]=="all")
        interpretation.append(f"{p['window']['name']}의 {av['cohort_sites']}개소에서 B5의 W 변화는 B3 대비 {f(annual['change_pct'],4)}%다. 월별 5% 초과 악화는 {', '.join(cp['unstable_months']) or '없다'}. 이 수치는 해당 고정 구성의 분기 성능이며 전체 253개소의 연간 성능으로 확대하지 않는다.")
    next_step=("다음 우선 후보는 P-003의 제한된 모델 용량/정규화 비교다. B5의 특징과 손실을 고정하고 변경 축과 예산, 동일 개발 비교 기준을 먼저 정한다. 이번 작업에서는 실행하지 않았다." if chosen=="B5" else
               "다음 우선 후보는 P-002의 예보 조건별 잔차 진단이다. 후반 표본에서 남은 집합 오차를 출발점으로 조건과 기간을 먼저 정해 EDA 계획에 등록한다. 기존 EDA 질문과 중복 여부를 확인하고 한 항목만 진행한다. 이번 작업에서는 추가 EDA와 다른 모델 탐색을 실행하지 않았다.")
    text=(OUT/"preregistered_report.md").read_text(encoding="utf-8")
    text=text.replace("상태: 진행. P-004의 가용성 점검과 사전 설계를 시작했다. 아직 이번 모델 학습과 비교 점수 계산은 하지 않았다.",f"작성: {now()}\n\n상태: 고정한 4회 학습과 저장 검증을 완료했다. 판단은 **{judgment}**다. 아래 학습 전 가용성 문단은 사전 단계의 기록이며 실행 결과와 구분한다.")
    lines=[text,"","## 실제 학습과 검증 결과","",
        f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']} 실행한 완료 요약 status=passed를 확인했다. 계약 테스트 {s['synthetic_tests']}개를 통과했으며 모델 적합은 사전 예산대로 4회다. 가용성 저장 실패 때는 모델을 적합하지 않았고 학습 오류 재시도는 없다.","",
        "장기 실행 대기 중 대화가 중단돼 원래 프로세스의 최종 종료 코드는 회수하지 못했다. 재개 시 실행 종료와 네 저장 모델/예측, 완료 요약을 확인했으며 새 프로세스로 저장 검증과 W/R/S 재계산을 수행했다. 해당 검증 프로세스의 종료 코드로 검증 완료를 확인하며 원래 학습 프로세스의 종료 코드와 구분한다. 완료 모델을 다시 학습하지 않았다.","",
        table(["기간","방법","관측 학습 행","낮 학습 행","특징 NaN 행","학습 초"],fits),"",
        "두 방법은 해당 시점의 학습 행 수/낮 선택/경계/39개 특징 순서와 float32 자료형/NaN 행 수가 같음을 대조했다. 학습의 energy/cap은 [0,1.2]로 자르고 실제 정답을 채점할 때는 자르지 않았다. 야간과 음수 예측은 0, 상한 미적용 처리를 유지했다. 기압 두 열도 유지했다.","",
        "### 고정 표본과 월별 W","","W는 집합별 잔차를 합한 뒤 절댓값을 취해 SMP×k를 곱한 합이며 낮을수록 좋다. k는 KST 종료 시각 11~15시에 3, 나머지는 1이다. 같은 비교 안에서 R과 가격/시간 가중 합/시각을 일치시켰다. 월/분기/k 분할 합계와 직접 집합 W 계산을 대조했다.","",
        table(["기간","표본","구간","시각 수","B3 W","B5 W","변화율"],results),"",
        "### 가중 집합 MAE","","가중 집합 MAE=W/(2×ΣSMP×k), 단위 kWh다. 같은 표본에서는 W와 순위가 같다.","",
        table(["기간","표본","B3 가중 집합 MAE","B5 가중 집합 MAE"],mae),"",
        "### 기존 표본과 추가 완전 관측 표본","","fixed_148은 기존 연간 고정 발전소 중 이번 분기의 완전 관측 발전소다. added_complete는 이번 분기 완전 관측 표본에서 기존 148개소를 제외한 구성이다. 각 소집합의 내부 잔차를 별도로 합산했으므로 두 W를 더해 전체 W로 해석하지 않는다. 그룹 구성이 양쪽 모두 있는 표본만 집합 지표를 기록했다.","",
        table(["기간","표본","발전소","B3 W","B5 W","변화율"],sub),"",
        "### 집합과 시간대 진단","","이 진단에 맞춰 추가 가중치나 보정 계수를 정하지 않았다. 과대/과소 비용 합과 k 분할 합계가 일치했다.","",
        table(["기간","집합","k","B3 W","B5 W","변화율","B5 과대 비용","B5 과소 비용"],group),"",
        "## 사전 기준에 따른 판단과 다음 단계","",table(["기간","분기 W 개선","월 5% 초과 악화","기간 기준"],gate),"",
        *[v for p in interpretation for v in (p,"")],f"최종 개발 판단은 **{judgment}**다. 앞선 두 연간 개발 비교와 이번 후반기 비교는 학습 마감과 구성이 다르므로 개선율을 합치거나 평균 정확도로 제시하지 않는다.","",
        next_step,"",
        "R4에서 B5는 기존 148개소 소집합의 W를 줄였지만 추가 104개소 소집합에서는 늘렸다. 집합별로는 G1이 개선되고 G2가 악화했다. 소집합 W는 합산할 수 없으므로 추가 발전소가 전체 악화의 원인이라고 단정하지 않는다. R4 전체 차이 0.0809%는 작으며 단일 적합 결과의 통계적 유의성이나 B5의 일반적 열세를 입증하지 않는다. 사전의 엄격한 개선 조건에 따라 B3로 복귀한 개발 판단이다.","",
        "## 해석 한계와 보존 검증","",
        "이번 분기 정답은 이전 V1에서 이미 본 2024년 자료다. 시점별로 다시 학습했지만 독립 시험이나 새 미래 정답은 아니다. 이전 연간 표본에 비해 학습 자료와 발전소 구성이 함께 달라졌으므로 특정 변화 하나의 인과 효과로 설명하지 않는다. 고정 완전 관측 표본 자체의 선택 편향, 한 쌍씩의 적합과 난수 반복 미실행 한계도 유지한다.","",
        "R4의 용량 조건 충족은 전체 253개소와 동일하다는 뜻이 아니며 R3는 용량 조건을 충족하지 않는다. 전체 253개소의 R4 공통 표본은 기존과 같은 겨울 178시간뿐이다. 이를 연간이나 계절 일반화 근거로 삼지 않았다.","",
        "공식 scoring.py와 문제정의서가 없어 로컬 채점의 공식 구현 동치는 미확인이다. 결과는 로컬 개발 집합 W/S이며 공식 제출 점수가 아니다.","",
        "- 원본 데이터/배포 파일과 기존 모델/예측/보고서 및 EDA 계획의 보존 해시를 확인했다. 기존 B3 평가 예측과 B5/B6 결과를 수정하지 않았다.",
        "- 저장한 네 모델을 문자열로 다시 읽어 L1/600개 트리/특징 순서를 검사했고 네 예측 파일도 다시 읽어 후보별 558,624행의 전체 253개소/시각 격자와 키 유일성, 유한/비음수를 검사했다. 원본 정답으로 같은 표본 W/R/S와 가중 집합 MAE를 다시 계산해 저장 요약과 대조했고 채택 판단도 재현했다.",
        "- 기간 경계와 미래 정답 누출 거부, 결측/실제 0 구분, 월 한도 5% 경계/동률 및 비교 구성 변경 거부를 합성 예제로 확인했다.",
        "- 새 기초 모델 적합은 4회, 이전까지 포함한 누적 이력은 23회다. B4 보정계수 2개는 별도다. V2/제공 평가 구간의 새 학습/예측/채점과 공식 제출은 없다. 기본 EDA 완료/선택/보류 목록을 유지했다.","",
        "## 관련 산출물","",
        "- [저장 결과 검증과 보고서 생성](../scripts/report_b5_rolling.py)",
        "- [가용성](../data/validation/2026-10-20_B3_B5_rolling/availability.json), [사전 코드/입력 검사](../data/validation/2026-10-20_B3_B5_rolling/precheck.json)",
        "- [전체 점수/진단/판단](../data/validation/2026-10-20_B3_B5_rolling/summary.json), [사전 보고서 원문](../data/validation/2026-10-20_B3_B5_rolling/preregistered_report.md)",
        "- [성능 개선 계획](성능개선_진행계획.md), [일별 로그](logs/2026-10-20.md), [기존 EDA 종료와 보류](EDA_진행계획.md)","",
        "원본/전처리 행, 식별자나 개별 예측은 공개 보고서에 복사하지 않았다. 모델/예측과 집계는 Git 제외 data/에 보관하고 공개 가능한 코드/보고서/로그만 로컬 커밋한다.",""]
    rendered="\n".join(lines);require("\u00b7" not in rendered,"Forbidden punctuation")
    REPORT.write_text(rendered,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(at=now(),report_sha256=digest(REPORT),summary_sha256=digest(OUT/"summary.json"),generator_sha256=digest(Path(__file__)),verified_artifacts=artifacts))
    print(f"Stored replay verified and report generated: {REPORT}",flush=True)


if __name__=="__main__":main()
