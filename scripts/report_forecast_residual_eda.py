"""Render verified aggregate EDA evidence into the public report, never raw rows."""
from __future__ import annotations
import sys
import sysconfig
from pathlib import Path
import numpy as np
import pandas as pd

from forecast_residual_eda import OUT, REPORT, read
from s0_validation_check import digest, now, write_json
from validation_core import require


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers)]+
                     ["| "+" | ".join(map(str,r))+" |" for r in rows])


def f(x, n=4):
    return f"{float(x):,.{n}f}"


def main():
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    summary=read(OUT/"summary.json");v=read(OUT/"verification.json");tv=read(OUT/"table_verification.json")
    require(v["status"]==tv["status"]==summary["status"]=="passed","Verified tables required")
    for name,h in v["output_sha256"].items():
        require(digest(OUT/name)==h,"Verified output changed")
    g=pd.read_csv(OUT/"group_conditions.csv",encoding="utf-8-sig")
    s=pd.read_csv(OUT/"site_conditions.csv",encoding="utf-8-sig")
    m=pd.read_csv(OUT/"matched_contrasts.csv",encoding="utf-8-sig")
    c=pd.read_csv(OUT/"signed_contributions.csv",encoding="utf-8-sig")
    b=pd.read_csv(OUT/"daily_blocks.csv",encoding="utf-8-sig")
    complete=g.loc[g.cohort.eq("complete")]
    lines=["## 실제 실행과 표 검증", "", f"완료 확인: {now()}。".replace("。","."), "",
           "정규 CPython 3.13.1에서 사전 점검과 본 계산, 별도 검증 프로세스가 각각 종료 코드 0으로 끝났다. 새 모델 적합/모델 로드 추론/보정 적합은 모두 0회다. 처음 사전 점검에서 새 코드의 괄호 오류와 갱신된 EDA 계획을 과거 고정 해시로 검사한 오류를 수정했다. 모두 본 계산 전이며 자료나 결과에 따라 조건을 바꾼 일이 아니다. 현재 계획은 과거 고정 사본과 구분해 보존 검사에서 제외하고 이번 실행 계약 사본을 별도로 저장했다.", "",
           "8개 예측 파일에서 전체 253개소 격자, 키 유일성, 정확한 시간축과 유한/비음수를 검사했다. 분석 고정 표본과 행 수는 다음과 같다. 표본이 겹치므로 네 행을 합쳐 독립 표본 수로 부르지 않는다.", "",
           table(["구간","고정 발전소","전체 행","주 진단 행","B3 W","B5 W"],
                 [[w["window"],w["sites"],f(w["rows"],0),f(w["main_rows"],0),f(w["B3_W"],2),f(w["B5_W"],2)] for w in summary["windows"]]), "",
           "B3/B5의 네 구간 전체 W가 이전 저장 결과와 rtol=1e-12에서 일치했다. 원본 및 기존 산출물 220개 해시는 실행 전후와 최종 검증 시점에 보존됐다. V2/평가 파일의 새 분석은 없다. 5개 합성 계약 테스트와 집합 조건 W 64개/부호 기여 64개 분할, 별도 표 검사 136개를 통과했다. 표 검사는 월/시간/k 합계, 실제-예측-잔차 관계, 작은 표본 표시, 셀 대비와 기존/추가 기여 합계까지 포함한다.", "",
           "R3/R4/V1의 네 예보 조인에는 누락이나 대상 변수의 내부 결측이 없다. T0에서는 A17 조인 누락 64행, B10 1,840행이 있으며 해당 행의 운량/강수도 결측이다. 둘 다 매칭된 행의 대상 변수 내부 결측은 0이다. 진단 범위의 운량/강수 범위 밖 값은 0건이다. 예보 정시 필터를 통과한 원본 지점 행과 발전소 조인 행의 수는 audits.csv에 따로 보관했다.", "",
           "## 확인한 조건과 반복 범위", "", "### 집합 전체가 흐린 조건과 절대 잔차", "",
           "아래는 집합-시각 용량 가중 운량 평균이 80%를 넘는 조건이다. 비율은 무차원이며 잔차는 예측-실제다. 시간/날짜 수는 기상 지점별 독립 관측 수가 아니다.", ""]
    cloud=complete.loc[complete.family.eq("cloud17") & complete.condition.eq("cloudy>80")]
    lines += [table(["구간","집합","시각/날짜","실제 비율","B3 비율","B3 잔차","B5 잔차","조건 W 변화"],
                    [[r.window,r.group,f"{r.times}/{r.days}",f(r.actual_cf),f(r.B3_cf),f(r.B3_bias),f(r.B5_bias),f(r.B5_change_pct,2)+"%"] for r in cloud.itertuples()]), "",
              "R3 흐림에서는 실제 비율이 약 0.32~0.34이고 B3는 과소였다. R4는 실제가 약 0.20~0.23으로 더 낮지만 B3 집합 평균은 과대였다. 같은 흐림 조건의 B5 W는 R3 두 집합에서 감소했으나 R4에서는 각각 5.79%, 3.55% 증가했다. 흐리면 무조건 과대 또는 과소라는 단일 방향 규칙을 확인하지 못했다. T0/V1도 절대 잔차가 같은 방향으로 반복되지 않는다.", "",
              "### 같은 월/시간/태양고도에서 흐림 대 나머지", "",
              "아래 값은 발전소-시각 조건의 공유 셀 대비다. 집합-시각 조건 표와 표본 정의가 다르다. 단위는 발전 비율의 %p이며 각 셀을 같은 비중으로 평균했다. `잔차 대비=예측 대비-실제 대비`다.", ""]
    contrast=m.loc[m.cohort.eq("complete") & m.family.eq("cloud17") & m.condition.eq("cloudy>80")]
    lines += [table(["구간","집합","공유 셀","실제 대비 %p","예측 대비 %p","잔차 대비 %p"],
                    [[r.window,r.group,r.cells,f(r.delta_actual*100,2),f(r.delta_pred*100,2),f(r.delta_bias*100,2)] for r in contrast.itertuples()]), "",
              "흐림의 실제 비율은 나머지 조건보다 낮았고, B3 예측 비율의 차이는 그보다 더 음수였다. 잔차 대비는 네 구간 양 집합에서 모두 음수다. 이 결과는 흐림에 따른 상대 하락폭을 현재 표현이 더 크게 나타낼 가능성을 시사하지만 인과 효과, 독립 추가 정보나 성능 개선의 증거는 아니다. 절대 잔차가 R4에서 양수일 수 있다는 사실과 모순되지 않는다.", "",
              "R3/R4의 기존/추가 표본에서도 같은 대비 방향인지와 지지 범위를 함께 확인했다.", ""]
    sub=m.loc[m.window.isin(["R3","R4"]) & m.cohort.ne("complete") & m.family.eq("cloud17") & m.condition.eq("cloudy>80")]
    support=s.loc[s.family.eq("cloud17") & s.condition.eq("cloudy>80")]
    rows=[]
    for r in sub.itertuples():
        q=support.loc[support.window.eq(r.window)&support.cohort.eq(r.cohort)&support.group.eq(r.group)].iloc[0]
        rows.append([r.window,r.cohort,r.group,r.cells,q["days"],f"{q.stationsA}/{q.stationsB}",f(r.delta_bias*100,2)])
    lines += [table(["구간","표본","집합","공유 셀","날짜","A/B 지점","잔차 대비 %p"],rows), "",
              "추가 표본까지 대비 방향은 음수였다. R3 추가 G1은 A/B 각각 5개 지점으로 사전 하한에 걸쳐 있어 넓은 공간 일반화 근거로 삼지 않는다. T0의 80개소는 기존 148개소의 부분집합이며 T0/V1의 complete와 fixed_148 행은 같은 자료이므로 반복 근거를 두 번 세지 않았다.", "",
              "### 날짜 반복과 불확실성", "",
              "아래 구간은 집합 흐림 조건의 달력일 평균 잔차 기여에 대한 7일 순환 블록 진단이다. 비활성 날짜를 0으로 둔 통계량으로 조건부 평균/공유 셀 대비의 신뢰구간이 아니다. 공간 구성은 날짜 블록과 함께 이동했으며 지점 반복을 독립으로 재표집하지 않았다.", ""]
    block=b.loc[b.cohort.eq("complete")&b.family.eq("cloud17")&b.condition.eq("cloudy>80")]
    lines += [table(["구간","집합","활성 날짜","과대 날짜 비율","달력일 평균 %p","블록 하한~상한 %p"],
                    [[r.window,r.group,r.active_days,f(r.positive_active_share,2),f(r.calendar_bias_mean*100,2),f(r.block_low*100,2)+" ~ "+f(r.block_high*100,2)] for r in block.itertuples()]), "",
              "R3에서는 과소 날짜가 많고 블록 구간도 음수였지만 R4/T0/V1은 이 통계량의 구간이 0을 포함한다. 공유 셀 대비에 대한 별도의 블록 불확실성 추정과 지역 구성을 더 맞춘 민감도는 이번에 실행하지 않았다. 이를 새 후보 EDA-012로만 남기며 H1은 제한된 모델 비교를 해볼 근거로 해석한다.", "",
              "### 강수와 예보 불일치에서 확인한 한계", "",
              "강우량이 커지면 실제 발전 비율이 낮아지는 관계와 잔차 방향은 다르다. 집합 A17 강우량 1 mm 초과에서 R3의 실제 비율은 G1/G2 0.1663/0.1758, B3는 0.1589/0.1685였다. R4는 실제 0.0881/0.1052, 예측 0.0804/0.0903이지만 조건 날짜가 3일/2일로 작은 표본이다. B17 1 mm 초과의 R4 G2는 평균 잔차가 양수이고 G1은 음수여서 일정 방향 보정을 뒷받침하지 않는다.", "",
              "A 운량이 20 %p 넘게 증가한 집합 조건은 R3 14/17일, R4 15/15일이다. R3에서는 양 집합 과소, R4에서는 과대로 바뀐다. B 운량 큰 갱신의 집합 조건은 R3 2~5일, R4 감소 5일이고 R4 증가 조건은 없었다. 큰 갱신을 일반적인 오류 조건으로 확정하기 어렵다.", "",
              "A/B 운량 차이도 조건별 절대 잔차와 월/시간/태양고도 대비가 기간별로 달라, 한 제공사의 우열이나 고정 혼합비를 정하지 않았다. B만 강수를 예보한 발전소 조건의 공유 셀 잔차 대비는 네 구간 양 집합 모두 음수였지만 집합 조건은 R3 12/17일, R4 6/8일로 작았다. 발전소별 조건의 지지와 전체 집합 조건의 반복을 같은 근거로 세지 않는다. 강수 정의의 한계까지 고려해 별도 모델 가설로 올리지 않았다.", "",
              "단순 Pearson에서 A 운량과 실제 비율은 R3 -0.4507, R4 -0.4969였고 잔차와는 -0.1927/-0.0366이었다. A 강수와 실제 비율은 -0.3130/-0.1433이나 잔차와는 0.0050/-0.0325였다. 이는 발전량과 날씨의 상관이 곧 반복되는 예측 오류는 아니라는 예다. 지점 쌍-시각의 행 수도 시간적으로 의존하므로 검정 표본 수로 쓰지 않았다.", "",
              "## 집합 W와 기존/추가 기여의 합계 보존", "",
              "집합-시각 운량 조건의 W 분할이다. 백만 W 단위로 표시한다. 과대/과소 비용은 각 조건 W의 방향 분할이며 합이 W다.", ""]
    part=complete.loc[complete.window.isin(["R3","R4"])&complete.family.eq("cloud17")]
    lines += [table(["구간","집합","조건","B3 W 백만","과대 비용 백만","과소 비용 백만","B5 W 백만"],
                    [[r.window,r.group,r.condition,f(r.B3_W/1e6,3),f(r.B3_over_W/1e6,3),f(r.B3_under_W/1e6,3),f(r.B5_W/1e6,3)] for r in part.itertuples()]), "",
              "다음은 기존/추가 표본의 내부 W가 아니라 전체 집합 부호를 사용한 기여다. 각 집합의 두 행을 더하면 그 집합의 전체 W가 된다. B3/B5 부호는 각각 따로 계산했으며 합계는 두 방법 전체 W 차이도 보존한다.", ""]
    contrib=c.loc[c.window.isin(["R3","R4"])&c.family.eq("cloud17")].groupby(["window","group","membership"])[["B3_contribution","B5_contribution"]].sum().reset_index()
    lines += [table(["구간","집합","표본","B3 기여 백만","B5 기여 백만","기여 차이 백만"],
                    [[r.window,r.group,r.membership,f(r.B3_contribution/1e6,3),f(r.B5_contribution/1e6,3),f((r.B5_contribution-r.B3_contribution)/1e6,3)] for r in contrib.itertuples()]), "",
              "R4 추가 G2의 부호 기여는 약 2.695백만 증가했고 같은 G2 기존 표본은 약 1.225백만 감소했다. 이는 전체 W 차이의 회계적 위치이며 추가 발전소가 악화의 인과 원인이라는 뜻은 아니다. 조건별 독립 부분 W를 더해 전체 W를 설명하지 않았다.", "",
              "## 종료 판단과 다음 단계", "",
              "EDA-011은 정한 범위에서 완료했다. 흐림의 상대 잔차 대비는 반복됐지만 고정 방향의 절대 보정, 강수/불일치 조건의 충분한 집합 날짜 반복, 새 특징의 추가 설명력과 성능 개선은 확인하지 않았다. H1 하나를 제한된 모델 비교로 넘긴다. 상세 식/결측/학습 내 적합/실패 조건과 제안 계약은 [모델링 인계](EDA_개선가설_모델링_인계.md)에 둔다. 최종 인계 상태는 모델링 가능이며 모델 채택이나 성능 개선 완료를 뜻하지 않는다.", "",
              "공식 scoring.py 동치, 전체 253개소의 연간 성능, 지역/운영 원인과 새로운 미래 기간 일반화는 미확인이다. V1과 R3/R4의 기간 중복, 관측 완전성에 따른 표본 선택과 고정 태양고도 셀의 잔여 교란도 남는다. 새 학습/튜닝/추론/보정/집합 재배정, V2 재분석과 제출은 실행하지 않았다.", "",
              "## 완료 산출물", "",
              "- [보고서 생성 코드](../scripts/report_forecast_residual_eda.py), [별도 표 검증 코드](../scripts/verify_forecast_residual_eda.py), [계약 테스트](../tests/test_forecast_residual_eda.py)",
              "- [조건별 실제/예측/잔차와 지지](../data/eda/2026-10-21_forecast_residual/site_conditions.csv), [집합-시각 조건](../data/eda/2026-10-21_forecast_residual/group_conditions.csv)",
              "- [공유 셀 대비](../data/eda/2026-10-21_forecast_residual/matched_contrasts.csv), [날짜 블록 진단](../data/eda/2026-10-21_forecast_residual/daily_blocks.csv)",
              "- [합계 보존 기여](../data/eda/2026-10-21_forecast_residual/signed_contributions.csv), [조인/결측/시간별 집계](../data/eda/2026-10-21_forecast_residual/audits.csv), [단순/월시간 편차 상관](../data/eda/2026-10-21_forecast_residual/correlations.csv)",
              "- [독립 표 검증 기록](../data/eda/2026-10-21_forecast_residual/table_verification.json), [EDA 계획](EDA_진행계획.md), [성능 계획](성능개선_진행계획.md)", ""]
    text=(OUT/"preregistered_report.md").read_text(encoding="utf-8")
    text=text.replace("상태: EDA-011 실행 전 계약. 아래 조건과 종료 기준을 먼저 고정했다. 결과는 실행 후 별도 절에 추가한다.",
                      "상태: EDA-011 실행과 저장 결과 검증 완료. 아래 계산 전 계약을 보존하고 실제 결과를 뒤에 추가했다. 새 학습과 추론은 수행하지 않았다.")
    text += "\n"+"\n".join(lines)
    require("\u00b7" not in text,"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(at=now(),report_sha256=digest(REPORT),generator_sha256=digest(Path(__file__)),
                                             summary_sha256=digest(OUT/"summary.json")))
    print("Verified aggregate report written; no source or prediction changes")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
