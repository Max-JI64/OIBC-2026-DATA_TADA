"""Render P-005 verified results; preserve the original preregistration separately."""
from __future__ import annotations
from pathlib import Path
import sys
import sysconfig
import pandas as pd
from h1_cloud_interaction import OUT, REPORT, read
from s0_validation_check import digest, now, write_json
from validation_core import require


def table(head,rows):
    return "\n".join(["| "+" | ".join(head)+" |","|"+"---|"*len(head)]+["| "+" | ".join(map(str,r))+" |" for r in rows])


def f(x,n=2):
    return "정의 안 됨" if x is None else f"{float(x):,.{n}f}"


def main():
    s=read(OUT/"summary.json");v=read(OUT/"verification.json");cv=read(OUT/"condition_verification.json")
    require(s["status"]==v["status"]==cv["status"]=="passed","Verified modeling results required")
    for file,h in cv["outputs_sha256"].items():require(digest(OUT/file)==h,"Condition output drift")
    chosen=s["selection"];decision="H1 채택" if chosen=="H1" else "H1 미채택, B3 유지"
    failures=[p["window"]["name"] for p in s["periods"] if not p["comparison"]["improved"]]
    lines=["## 실제 실행과 검증 결과","",f"완료 확인: {now()}. 판단: **{decision}**.","",
           f"정규 CPython {s['runtime']['python']}, LightGBM {s['runtime']['lightgbm']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']} 실행했다. 사전 점검/학습/저장 검증/조건 진단의 프로세스 종료 코드 0을 확인했다. 새 적합은 정확히 4회, 적합 시도 4회, 적합 오류와 재시도 0회, B3 재적합 0회다. 특징 통계/보정계수 적합도 0회다. 완료 모델의 재적합 없이 네 모델과 예측을 다시 읽어 점수와 판단을 재현했다.","",
           "합성 계약 테스트 7개를 통과했다. 사전 점검에서 기존 B3의 학습 경계, 원시 39개 순서/float32/낮 행/NaN 행/관측 행, 모델 L1/600트리와 실행 환경이 일치해 저장 B3를 재사용했다. 코드/원본/EDA 출력/이전 모델과 평가 예측의 해시는 실행 전후 보존됐다. V2와 제공 평가 구간의 새 학습/추론/채점은 없다.",""]
    train=[];overall=[];gates=[];periodrows=[];groups=[];commons=[];sub=[]
    for p in s["periods"]:
        name=p["window"]["name"];fit=p["fit"];t=fit["training"]
        train.append([name,p["sites"],f(t["train_rows"],0),f(t["daylight_rows"],0),f(t["nan_rows"],0),f(t["new_nan_rows"],0),f(fit["fit_seconds"])])
        for r in p["comparison"]["rows"]:
            row=[name,r["sample"],r["period"],r["times"],f(r["B3_W"]),f(r["H1_W"]),f(r["change_pct"],4)+"%"]
            if r["sample"]=="fixed_cohort":
                if r["period"]=="all":overall.append(row)
                if r["period"].startswith(("Q","M")):periodrows.append(row)
            elif r["period"]=="all":commons.append(row)
        gate=p["comparison"]
        gates.append([name,"감소" if gate["improved"] else "미감소",", ".join(gate["unstable"]) or "없음","통과" if gate["passes"] else "미통과"])
        a={(r["cohort"],r["group"],r["k"]):r for r in p["diagnostics"]["B3"]}
        for b in p["diagnostics"]["H1"]:
            old=a[(b["cohort"],b["group"],b["k"])]
            row=[name,b["cohort"],b["group"],b["k"],b["sites"],f(old["W"]),f(b["W"]),f((b["W"]/old["W"]-1)*100,4)+"%",f(b["over_W"]),f(b["under_W"])]
            if b["cohort"]=="complete":groups.append(row)
            elif name in ("R3","R4") and b["k"]=="all":sub.append(row)
    lines += ["### 실제 학습 행과 특징 결측","",table(["구간","채점 발전소","관측 학습 행","낮 학습 행","기존 특징 NaN 행","추가 3열 NaN 행","학습 초"],train),"",
              "추가 특징 NaN 행 수는 기존 NaN 행 수와 정의가 다르다. 원시 39개와 정답/학습 행은 그대로 유지했고, 추가 운량 교차항 3개만 지정한 결측 마스크를 적용했다.","",
              "### 구간별 전체 집합 W","",table(["구간","표본","기간","시각 수","B3 W","H1 W","변화율"],overall),"",
              "### 사전 채택 기준","",table(["구간","전체 W 엄격 감소","안정성 한도 초과 구간","구간 판단"],gates),"",
              f"최종 판단은 **{decision}**다. 전체 W 감소를 충족하지 못한 구간은 {', '.join(failures) or '없음'}이다. 네 구간의 사전 기준을 모두 통과할 때만 H1을 채택한다. 일부 기간이나 EDA 조건의 개선으로 다른 기간의 악화를 상쇄해 채택하지 않는다. 이 판단은 개발 선택 규칙이며 통계적 유의성 검정이 아니다.","",
              "### 월별/분기별 전체 W","",table(["구간","표본","기간","시각 수","B3 W","H1 W","변화율"],periodrows),"",
              "### 집합과 k 및 방향 비용","",table(["구간","표본","집합","k","발전소","B3 W","H1 W","변화율","H1 과대 비용","H1 과소 비용"],groups),"",
              "G1/G2 W 합, k=1/3 분할과 과대/과소 비용 합을 각각 검사했다. 전체 집합 사이 잔차를 먼저 상쇄하지 않았다.","",
              "### 기존/추가 표본의 별도 내부 W","",table(["구간","표본","집합","k","발전소","B3 W","H1 W","변화율","H1 과대 비용","H1 과소 비용"],sub),"",
              "위 표는 각 표본 안에서 잔차를 별도로 합친 W이므로 기존/추가 행을 더하지 않는다. 합계 보존 기여는 아래 별도 표에 둔다.","",
              "### 전체 253개소의 짧은 공통 정답 진단","",table(["구간","표본","기간","시각 수","B3 W","H1 W","변화율"],commons),"",
              "공통 구간이 없는 기간은 표에 없다. 짧은 겨울 178시간의 결과를 전체 253개소 연간 성능이나 독립 시험으로 해석하지 않으며 사전 채택 기준으로 추가하지 않았다.","",
              "## EDA 조건에 대한 실제 변경 효과","",
              "기존 EDA 코드와 사전 고정한 8개 조건/구간을 그대로 사용했다. B3 조건별 표본과 W/방향/비율, 흐림 공유 셀 대비를 기존 EDA 저장 집계와 다시 대조했다. 원래 주 진단 행 정의를 모델 학습 행으로 옮기지 않았다. 다음은 집합-시각 운량 80% 초과 조건이다.",""]
    d=pd.read_csv(OUT/"condition_comparison.csv",encoding="utf-8-sig")
    d=d.loc[d.cohort.eq("complete")&d.family.eq("cloud17")&d.condition.eq("cloudy>80")]
    lines += [table(["구간","집합","시각/날짜","실제 cf","B3 cf","H1 cf","B3 W","H1 W","조건 변화율"],
                    [[r.window,r.group,f"{r.times}/{r.days}",f(r.actual_cf,4),f(r.B3_cf,4),f(r.H1_cf,4),f(r.B3_W),f(r.H1_W),f(r.change_pct,4)+"%"] for r in d.itertuples()]),"",
              "조건별 감소는 전체 성능 채택과 별개다. 강수/운량 갱신/제공사 불일치의 전체 결과와 기존/추가 표본은 condition_comparison.csv에 보관했고 작은 날짜 수의 해석 제한도 유지한다.","",
              "### 같은 월/시간/태양고도의 흐림 대 나머지","",
              "단위는 발전 비율 %p다. 조건의 실제 비율 대비는 B3/H1에 동일하며 학습용 목표나 잔차 보정값으로 사용하지 않았다.",""]
    m=pd.read_csv(OUT/"cloud_contrasts.csv",encoding="utf-8-sig");main=m.loc[m.cohort.eq("complete")]
    lines += [table(["구간","집합","공유 셀","실제 대비 %p","B3 예측 대비 %p","H1 예측 대비 %p","B3 잔차 대비 %p","H1 잔차 대비 %p"],
                    [[r.window,r.group,r.cells,f(r.delta_actual*100),f(r.delta_B3*100),f(r.delta_H1*100),f(r.B3_bias_contrast*100),f(r.H1_bias_contrast*100)] for r in main.itertuples()]),"",
              "상대 잔차 대비가 0에 가까워지더라도 집합 시각별 W가 반드시 줄어드는 것은 아니다. 대비 자체의 블록 불확실성과 지역 구성 민감도는 이번 모델 실험에서 추가 EDA하지 않았으며 EDA-012 후보로 남긴다.","",
              "실제로 R4의 흐림 잔차 대비는 G1 -2.11→-1.66 %p, G2 -2.09→-1.67 %p로 0에 가까워졌지만 전체 W는 0.9358% 증가했다. R3에서는 흐림 조건 W가 G1 4.4377%, G2 4.5407% 증가했고 공유 셀 잔차 대비도 완화되지 않았다. 따라서 이번 교차 표현이 EDA의 상대 잔차 패턴과 전체 집합 목적을 함께 개선했다고 볼 수 없다. 단일 적합 결과로 운량/태양/계절 관계 자체가 쓸모없다고 결론 내리지는 않는다.","",
              "### 합계를 보존하는 기존/추가 기여","",
              "기여는 각 방법의 전체 집합 잔차 부호로 계산한 q×sign(E전체)×부분 잔차 합이다. 독립 부분 W가 아니며 음수가 될 수 있다. 다음은 조건을 모두 합친 기여로 집합 안의 두 표본 합이 전체 W를 보존한다.",""]
    c=pd.read_csv(OUT/"signed_contributions.csv",encoding="utf-8-sig")
    c=c.loc[c.window.isin(["R3","R4"])&c.family.eq("cloud17")].groupby(["window","group","membership"])[["B3_contribution","H1_contribution"]].sum().reset_index()
    lines += [table(["구간","집합","표본","B3 기여","H1 기여","기여 차이"],[[r.window,r.group,r.membership,f(r.B3_contribution),f(r.H1_contribution),f(r.H1_contribution-r.B3_contribution)] for r in c.itertuples()]),"",
              f"조건 진단에서는 집합별 조건 분할 {cv['group_partitions']}개와 전체 W를 보존하는 발전소 조건/기존-추가 부호 기여를 검증했다. 총 {cv['rows']}개 조건 집계와 {cv['cloud_contrasts']}개 흐림 공유 셀 대비를 저장했다. 모델/예측 재독해 검증과 별도 저장 집계 비교를 모두 마쳤다.","",
              "## 한계와 종료 판단","",
              "T0/V1/R3/R4는 이미 관찰한 개발 자료이며 V1과 R3/R4는 정답 기간이 겹친다. 완전 관측 표본 선택, 짧은 전체 253개소 공통 구간과 단일 적합, 같은 날/지역의 의존성과 공식 채점 구현 동치 미확인 한계를 유지한다. 특징 3개는 새 관측 정보가 아니라 기존 입력의 표현 변경이며 일반화나 원인을 증명하지 않는다.","",
              f"P-005는 예산 4회와 저장 검증/조건 비교/판단 기록으로 종료한다. 현재 우선 참조는 {chosen}다. 추가 가설/설정/난수 탐색, 잔차 보정, 집합 재배정과 평가 예측 갱신으로 이어가지 않았다. 기존 B3 모델과 평가 예측은 보존했다.","",
              "## 완료 근거와 코드","",
              "- [모델 학습/검증 코드](../scripts/h1_cloud_interaction.py), [조건 진단 코드](../scripts/h1_condition_diagnostics.py), [보고서 생성](../scripts/report_h1_cloud_interaction.py), [7개 계약 테스트](../tests/test_h1_cloud_interaction.py)",
              "- [사전 입력/코드/학습 계약](../data/validation/2026-10-21_H1_cloud_interaction/precheck.json), [모델 적합 전 조건 진단 코드 고정](../data/validation/2026-10-21_H1_cloud_interaction/condition_contract.json)",
              "- [완료 요약과 모든 점수/판단](../data/validation/2026-10-21_H1_cloud_interaction/summary.json), [저장 검증](../data/validation/2026-10-21_H1_cloud_interaction/verification.json), [조건 검증](../data/validation/2026-10-21_H1_cloud_interaction/condition_verification.json)",
              "- [전체 조건 비교](../data/validation/2026-10-21_H1_cloud_interaction/condition_comparison.csv), [흐림 공유 셀 대비](../data/validation/2026-10-21_H1_cloud_interaction/cloud_contrasts.csv), [합계 보존 기여](../data/validation/2026-10-21_H1_cloud_interaction/signed_contributions.csv)",
              "- [성능 계획](성능개선_진행계획.md), [EDA 인계](EDA_개선가설_모델링_인계.md), [일별 로그](logs/2026-10-21.md)","",
              "모든 데이터/모델/개별 예측과 CSV는 Git 제외 로컬에 보관하며 공개 코드/집계 보고서와 필요한 문서만 로컬 커밋한다.",""]
    text=(OUT/"preregistered_report.md").read_text(encoding="utf-8")
    text=text.replace("상태: P-005/H1 실행 전 계약. 실제 적합과 비교 결과는 실행 후 별도로 추가한다.",f"상태: P-005/H1의 4회 적합과 저장/조건 검증 완료. 판단은 {decision}다. 아래 실행 전 계약과 해시는 보존하고 실제 결과를 뒤에 추가했다.")
    text+="\n"+"\n".join(lines);require("\u00b7" not in text,"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(at=now(),report_sha256=digest(REPORT),generator_sha256=digest(Path(__file__)),summary_sha256=digest(OUT/"summary.json")))
    print(f"P-005 report written: {decision}")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    main()
