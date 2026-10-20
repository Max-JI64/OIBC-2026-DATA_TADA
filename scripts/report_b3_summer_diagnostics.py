"""Render completed D3 aggregates and a static chart, without new predictions."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from b3_summer_diagnostics import OUT, REPORT
from s0_validation_check import ROOT, digest, now, table, write_json
from validation_core import require


def f(value, digits=2):
    return f"{value:,.{digits}f}" if value is not None else "정의하지 않음"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    require(s["status"]=="passed" and s["model_fit_count"]==s["model_prediction_count"]==s["weight_search_count"]==0,
            "Completed scoped diagnosis required")
    require(digest(OUT/"preregistered_report.md")==s["prereg_report_sha256"],"Scope archive changed")
    for rel,h in s["code_sha256"].items():
        require(digest(OUT/"executed_code"/Path(rel).name)==h,"Executed code archive changed")
    for rel,h in s["protected_sha256"].items(): require(digest(ROOT/rel)==h,"Protected result changed")
    def row(group,period,k="all"):
        return next(r for r in s["periods"] if r["group"]==group and r["period"]==period and r["k"]==str(k))
    a=row("TOTAL","M08")
    group_rows=[]
    for g in ("G1","G2","TOTAL"):
        r=row(g,"M08")
        group_rows.append([g,f(r["B2_W"]),f(r["B3_W"]),f(r["change_pct"],4)+"%",f(r["delta_W"]),f(r["delta_over_W"]),f(r["delta_under_W"])])
    month_rows=[]
    for period in ("M07","M08","M09","Q3"):
        r=row("TOTAL",period)
        month_rows.append([period,f(r["B2_W"]),f(r["B3_W"]),f(r["change_pct"],4)+"%",f(r["delta_over_W"]),f(r["delta_under_W"])])
    k_rows=[]
    for g in ("G1","G2","TOTAL"):
        for k in (1,3):
            r=row(g,"M08",k)
            k_rows.append([g,k,f(r["delta_W"]),f(r["delta_over_W"]),f(r["delta_under_W"])])
    hours=[r for r in s["hours"] if r["group"]=="TOTAL" and r["month"]==8 and 11<=r["end_hour"]<=15]
    hours.sort(key=lambda r:r["end_hour"])
    hshare=sum(r["delta_W"] for r in hours if r["end_hour"] in (14,15))/a["delta_W"]
    bias_rows=[]
    for g in ("G1","G2"):
        r=row(g,"M08");high=row(g,"M08",3)
        bias_rows.append([g,f(r["B2_weighted_bias"]),f(r["B3_weighted_bias"]),f(r["weighted_prediction_shift"]),
                          f"{high['higher_prediction_times']}/{high['positive_weight_times']}"])
    conc_rows=[]
    for r in s["concentration"]:
        conc_rows.append([r["group"],r["worse_days"],r["better_days"],f(r["gross_increase"]),f(r["gross_improvement"]),
                          f(r["net_delta"]),f(r["top_5_positive_days_share"]*100)+"%"])
    daily=sorted([r for r in s["days"] if r["group"]=="TOTAL"],key=lambda r:r["delta_W"],reverse=True)
    top=[[r["target_day"],f(r["delta_W"]),f(r["delta_over_W"]),f(r["delta_under_W"])] for r in daily[:5]]
    sign_rows=[]
    for g in ("G1","G2"):
        for label,condition in (("both_over",lambda r:r["B2_sign"]==1 and r["B3_sign"]==1),
                                ("both_under",lambda r:r["B2_sign"]==-1 and r["B3_sign"]==-1),
                                ("opposite",lambda r:r["B2_sign"]*r["B3_sign"]==-1)):
            rr=[r for r in s["sign_pairs"] if r["group"]==g and r["month"]==8 and condition(r)]
            names={"both_over":"두 방법 모두 과대","both_under":"두 방법 모두 과소","opposite":"오차 방향 반대"}
            sign_rows.append([g,names[label],sum(r["times"] for r in rr),f(sum(r["delta_W"] for r in rr))])

    fig,axes=plt.subplots(2,1,figsize=(12,8),layout="constrained")
    x=np.arange(3)
    over=[row("TOTAL",f"M{m:02}")["delta_over_W"]/1e6 for m in (7,8,9)]
    under=[row("TOTAL",f"M{m:02}")["delta_under_W"]/1e6 for m in (7,8,9)]
    axes[0].bar(x-.17,over,width=.34,label="Change in overprediction cost",color="#d65f45")
    axes[0].bar(x+.17,under,width=.34,label="Change in underprediction cost",color="#4b86b4")
    axes[0].plot(x,np.array(over)+under,"ko",label="Net change")
    axes[0].set_xticks(x,["July","August","September"])
    axes[0].set_title("Fixed 148-site cohort: B3 minus B2, 2024")
    axes[0].set_ylim(min(0,*under,*over)-2,max(0,*under,*over)*1.4+1)
    axes[0].legend(frameon=False,ncol=3)
    for i,g in enumerate(("G1","G2")):
        rr=sorted([r for r in s["days"] if r["group"]==g],key=lambda r:r["target_day"])
        axes[1].bar(np.arange(1,32)+(i-.5)*.38,[r["delta_W"]/1e6 for r in rr],width=.38,label=g,color=("#d65f45","#4b86b4")[i])
    axes[1].set_title("August: daily change in group W")
    axes[1].set_xticks(np.arange(1,32));axes[1].set_xlabel("Target day in August (ts minus one hour)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axhline(0,color="#333333",linewidth=.8);ax.set_ylabel("Delta W / 1e6")
        ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"summer_delta.png",dpi=160)
    plt.close(fig)

    lines=["# B3 여름철 오차 악화 진단","",f"작성: {now()}","",
        "상태: 저장된 V1 B2/B3 예측을 이용한 진단을 완료했다. 모델 적합, 새 모델 추론과 가중치 탐색은 모두 0회다. V2와 최종 평가 자료는 새로 진단하지 않았으며 EDA 종료와 선택 후보 3개/보류 5개를 유지했다.","",
        "## 질문과 목적","",
        "[B3 비교](2026-10-20_05_B3_L1손실_비교.md)에서 확인한 V1 3분기, 특히 8월의 악화가 어느 집합/시간대/방향/날짜에 집중됐는지 확인했다. 발견일은 2026-10-20이고 부모 작업은 B3다. 사용자가 요청한 한 차례 후속 진단으로, 최종 학습 전에 약점과 수정 근거를 점검했다.","",
        "핵심 결과는 **8월 G1의 가중치 3 시간대에서 예측이 높아졌고, 과대예측 비용 증가가 과소예측 비용 감소를 넘어섰다**는 것이다. G2는 개선됐다. 이 결과는 오차의 위치와 방향을 설명하며 L1의 일반적 효과나 기상 원인의 확정은 아니다.","",
        "## 사용 범위와 수행 방법","",
        "고정 148개소의 V1 정답과 SMP, 저장된 B2/B3 예측을 사용했다. 148개소는 G1 126개소/G2 22개소다. 전체 253개소의 G1 134개소/G2 119개소 구성과 다르며 제출 용량 조건을 충족한 표본으로 취급하지 않는다. 여름 전체 253개소 공통 정답 시각은 0개여서 나머지 105개소의 실적을 만들지 않았다.","",
        "분해 전에 보고서에 범위와 종료 기준을 기록하고 코드 사본 및 원문 해시를 저장했다. 연간/분기/월의 기존 고정 표본 점수를 먼저 재현한 뒤 상세 분해를 7~9월 2,208시각에 제한했다. 8월 날짜 분류는 ts-1시간의 KST 날짜이며 31일 모두 유지했다. 기상 변수, 개별 발전소 원인, 실측 기상과 새 특징 분석으로 범위를 넓히지 않았다.","",
        "W=Σ집합,시각 SMP×k×|집합 예측-집합 정답|이다. 각 집합 안에서 먼저 잔차를 합치고 절댓값을 취하며 G1/G2 잔차를 서로 상쇄하지 않았다. ΔW=B3 W-B2 W로 양수는 악화다. 과대 비용은 Σq×max(잔차,0), 과소 비용은 Σq×max(-잔차,0), q=SMP×k다. 두 비용의 합과 증감 합이 W 및 ΔW와 일치하는지 검사했다.","",
        "## 결과","",
        "### 월과 분기","",table(["기간","B2 W","B3 W","W 변화","과대 비용 증감","과소 비용 증감"],month_rows),"",
        "7월은 개선됐고 8~9월은 악화됐다. 특히 8월은 과소 비용이 줄었지만 과대 비용이 더 크게 늘었다. 3분기 가중 편향이 0에 가까워졌더라도 시각별 집합 절대오차가 증가할 수 있으므로 평균 편향만으로 성능을 판단하지 않는다.","",
        "![여름 월별 방향 비용과 8월 일별 집합 W 변화](../data/validation/2026-10-20_D3_B3_summer/summer_delta.png)","",
        "### 8월의 집합과 시간대","",table(["집합","B2 W","B3 W","W 변화","ΔW","과대 비용 증감","과소 비용 증감"],group_rows),"",
        f"G1의 ΔW는 전체 순증가의 {f(row('G1','M08')['delta_W']/a['delta_W']*100)}%이고 G2의 개선이 일부 상쇄했다. 집합별 W 증감을 합산한 값이며 서로 다른 집합의 잔차를 먼저 합쳐 계산하지 않았다.","",
        table(["집합","k","ΔW","과대 비용 증감","과소 비용 증감"],k_rows),"",
        f"전체 8월 순증가의 {f(row('TOTAL','M08',3)['delta_W']/a['delta_W']*100)}%가 k=3에서 발생했다. 아래 종료 14/15시 두 시간대의 순증가 합은 월간 순증가의 {f(hshare*100)}%다. 다른 시간대의 개선이 있어 개별 시간대의 순증가 비중 합이 100%를 넘을 수 있다.","",
        table(["KST 종료 시각","ΔW","과대 비용 증감","과소 비용 증감"],[[r["end_hour"],f(r["delta_W"]),f(r["delta_over_W"]),f(r["delta_under_W"])] for r in hours]),"",
        "### 예측 이동과 오차 방향","",table(["집합","B2 가중 편향 kWh","B3 가중 편향 kWh","B3-B2 가중 예측 이동 kWh","k=3에서 예측 상승/양의 가중 시각"],bias_rows),"",
        "편향은 예측-정답의 q 가중 평균이다. G1은 8월에 과소/과대의 순균형에서 과대 쪽으로 이동했고 G2는 과소 편향이 줄었다. 두 집합에 같은 하향 보정값을 적용할 근거로 사용할 수는 없다.","",
        table(["집합","두 잔차의 방향","시각 수","해당 시각의 ΔW"],sign_rows),"",
        "두 방법 모두 과대인 시각에서 악화가 컸다. 반대 방향인 시각의 개선만 보고 혼합이 유리하다고 단정하지 않았다. 이 진단에서는 평균 혼합이나 보정 예측을 만들지 않았다. 합성 검증의 방향 분해 예제는 실제 새 모델 예측이 아니다.","",
        "### 8월 날짜 집중","",table(["집합","악화일","개선일","악화일 증가 합","개선일 감소 합","월간 순증가","상위 5일/증가 합"],conc_rows),"",
        "상위 날짜 비중의 분모는 양의 일별 ΔW만 합친 값이다. 개선일을 뺀 월간 순증가를 분모로 써 집중도를 부풀리지 않았다. 전체 31일 중 18일이 악화했고 G1은 21일 악화해 한두 날짜만의 현상으로 설명하기 어렵다. 다만 상위 5일이 양의 증가 합의 절반 이상이어서 날짜별 영향의 차이도 있다.","",
        table(["대상 날짜","전체 ΔW","과대 비용 증감","과소 비용 증감"],top),"",
        "위 순위는 날짜별 영향을 설명하기 위한 것이다. 이후 점수에서 제외하거나 특정 날짜 보정을 적합하지 않았다. 상위 악화일 중에도 과대 비용 증가가 주된 날과 과소 비용 증가가 주된 날이 있어 모든 악화를 한 방향으로 설명할 수 없다.","",
        "## 판단과 다음 단계","",
        "현재 판단은 **B3 개발 우선 후보 유지**다. 사전 개발 기준을 통과한 판단을 바꾸거나 8월 개선을 새로운 필수 통과 조건으로 추가하지 않았다. D3는 오차 위치와 방향을 확인하는 한 차례 진단으로 종료한다.","",
        "후속 가설 D3-H1을 선택/후보로 등록한다. 발견일은 2026-10-20이고 출처는 이번 G1/k=3 과대 비용 증가다. 학습 구간에서 만든 별도 검증 예측만으로 G1의 k=3 과대 편향을 추정한 보정이 도움이 될 수 있다는 가설이다. 다만 이번에는 그 학습 구간의 검증 잔차, 계절별 반복성과 보정 안정성을 확인하지 않았다. V1 8월 잔차로 바로 보정값을 정하면 확인한 정답에 맞추는 문제가 생긴다. 최종 학습의 필수 대기로 올리지 않으며 보정값 계산과 비교 실험은 미실행이다. 재개한다면 보정 자료의 시간 분리, 적용 조건과 종료/채택 기준을 먼저 고정하고 한 가지 안만 비교한다.","",
        "이번 근거만으로 8월에 B2로 전환하거나 G1을 일괄 낮추지 않는다. 두 선택 모두 이미 본 여름 결과에 맞춘 추가 규칙이며 G2, 겨울과 다른 날의 개선을 보장하지 못한다. 원인 규명이 미확정이라는 이유로 실험을 연이어 추가하지 않는다.","",
        "다음 실행은 기존 [최종 학습/평가 설계](2026-10-20_06_B3_최종학습과_평가절차_설계.md)의 B3 1회 학습과 고정 참조 구성의 평가 예측 로컬 생성/검증이다. 이번에는 해당 실행과 공식 제출을 하지 않았다. 새로운 독립 미래 정답과 전체 253개소의 여름 성능은 여전히 확보하지 못했다.","",
        "## 검증과 한계","",
        f"정규 CPython {s['runtime']['python']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']}에 실행했고 종료 코드 0을 확인했다. 합성 테스트 {s['synthetic_tests']}개를 통과했다.","",
        "- 기존 고정 148개소의 연간/네 분기/12개월 B2/B3 W/R/S/가격 가중 합/시각 수를 재현했다. 새 표본으로 바꾸지 않았다.",
        "- 저장된 각 V1 전체 격자 2,222,352행의 키/시간축/253개소/유한/비음수를 확인했다. 상세 진단의 정답은 148개소에만 한정했다.",
        "- 집합별 월/k/종료 시각/날짜/두 모델 잔차 방향 분할의 W와 방향 비용 증감을 대조했다. 집합 합과 전체 W도 일치했다.",
        "- 원본 10개 데이터와 배포 README/baseline.py, 이전 모델/예측/진단과 P1 설계 산출물 및 EDA 계획의 해시가 유지됐다. V2는 내용을 해석하지 않고 파일 해시로 보존만 확인했다.",
        "- LightGBM이 import되지 않았으며 새 학습/추론/보정/혼합/가중치 탐색은 없었다. 합성 테스트만 생성된 예제 잔차를 사용했다.","",
        "TOTAL 집계의 times는 두 집합-시각을 세어 실제 시각 수의 두 배이고 편향은 두 집합을 같은 비중으로 평균한 값이다. W는 두 집합의 절대오차 비용 합이다. CSV의 집합별 행을 사용할 때 이 차이를 유지해야 한다.","",
        "관측된 편향과 예측 이동은 기상 조건, 자료 수집이나 학습 손실의 인과 효과를 밝힌 것이 아니다. 이미 본 V1에서 약점을 찾은 사후 진단이며 독립 시험이나 통계적 유의성 검증도 아니다. 공식 채점기 동치와 실제 평가 구간 성능은 미확인이다.","",
        "## 관련 코드와 로컬 산출물","",
        "- [진단 코드](../scripts/b3_summer_diagnostics.py), [합성 테스트](../tests/test_summer_diagnostics.py), [보고서/그림 생성](../scripts/report_b3_summer_diagnostics.py)",
        "- [실행 요약과 보존 해시](../data/validation/2026-10-20_D3_B3_summer/summary.json), [사전 범위 원문](../data/validation/2026-10-20_D3_B3_summer/preregistered_report.md)",
        "- [월/집합/k](../data/validation/2026-10-20_D3_B3_summer/periods.csv), [종료 시각](../data/validation/2026-10-20_D3_B3_summer/hours.csv), [날짜](../data/validation/2026-10-20_D3_B3_summer/days.csv), [방향 조합](../data/validation/2026-10-20_D3_B3_summer/sign_pairs.csv)",
        "- [일별 로그](logs/2026-10-20.md), [EDA 종료와 보류 목록](EDA_진행계획.md)","",
        "원본과 전처리 행, 개별 예측 및 식별자는 공개 보고서에 복사하지 않았다. 집계와 그림은 Git 제외 data/에 저장한다. 보고서 재생성은 저장 집계만 읽고 새 예측을 만들지 않는다.",""]
    text="\n".join(lines)
    require("\u00b7" not in text,"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(generated_at=now(),report_sha256=digest(REPORT),
        summary_sha256=digest(OUT/"summary.json"),generator_sha256=digest(Path(__file__)),figure_sha256=digest(OUT/"summer_delta.png")))
    print(f"D3 report generated: {REPORT}",flush=True)


if __name__=="__main__": main()
