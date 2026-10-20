"""Report D1 and nominate one unexecuted future model hypothesis."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from report_b0_v1 import f
from s0_validation_check import SOURCE, digest, now, table, write_json
from v1_residual_diagnostics import OUT, REPORT, ROOT
from validation_core import require


def pct(value):
    return f"{100*value:.2f}%" if value is not None else "정의하지 않음"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    require(s["status"] == "passed" and s["model_fit_count"] == s["model_prediction_count"] == s["weight_search_count"] == 0,
            "D1 scope mismatch")
    for rel, expected in s["protected_artifact_sha256"].items():
        require(digest(ROOT / rel) == expected, "Baseline artifact changed")
    for rel, expected in s["code_sha256"].items():
        require(digest(OUT / "executed_code" / Path(rel).name) == expected, "Executed code archive changed")
    require(digest(SOURCE / "sites.csv") == s["source_sha256"]["sites.csv"], "Sites changed")
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig").set_index("site_id")
    b0 = json.loads((ROOT / "data/validation/2026-10-19_B0_V1/summary.json").read_text(encoding="utf-8"))
    fixed_ids = b0["fixed_148_sites"]
    rows, rounds, coherent = s["groups"], s["rounds"], s["coherence"]
    def group(group, period="all", sample="fixed_148"):
        return next(r for r in rows if r["sample"] == sample and r["group"] == group and r["period"] == period)
    annual = sum(group(g)["W"] for g in ("G1", "G2"))
    high = sum(group(g, "k=3")["W"] for g in ("G1", "G2"))
    pool_round = {}
    for period in ("all", "Q1", "Q2", "Q3", "Q4", "k=1", "k=3"):
        part = [r for r in rounds if r["sample"] == "fixed_148" and r["period"] == period]
        pool_round[period] = {key: sum(r[key] for r in part) for key in ("W10", "W17", "Wmean")}
    pooled = pool_round["all"]
    delta17 = (pooled["Wmean"]/pooled["W17"]-1)*100
    meta = []
    for label, ids in (("고정 148", fixed_ids), ("전체 253", sites.index.tolist())):
        for raw, name in (("A", "G1"), ("B", "G2")):
            subset = sites.loc[ids]
            subset = subset.loc[subset.current_group.eq(raw)]
            meta.append([label, name, len(subset), f(subset.capacity_kw.sum(), 3)])
    overall_rows = []
    for sample in ("fixed_148", "common_253"):
        for g in ("G1", "G2"):
            r = group(g, sample=sample)
            overall_rows.append([sample, g, r["sites"], f(r["times"], 0), f(r["W"]),
                                 f(r["weighted_mae_kwh"]), f(r["weighted_bias_kwh"]), pct(r["over_W_share"]),
                                 pct(r["cancellation_ratio"])])
    month_rows = []
    for m in range(1, 13):
        a, b = group("G1", f"M{m:02}"), group("G2", f"M{m:02}")
        month_rows.append([m, f(a["weighted_mae_kwh"]), f(a["weighted_bias_kwh"]), pct(a["over_W_share"]),
                           f(b["weighted_mae_kwh"]), f(b["weighted_bias_kwh"]), pct(b["over_W_share"])])
    hour_rows = []
    for h in range(8, 19):
        a, b = group("G1", f"H{h:02}"), group("G2", f"H{h:02}")
        hour_rows.append([h, 3 if 11 <= h <= 15 else 1, pct((a["W"]+b["W"])/annual),
                          f(a["weighted_mae_kwh"]), f(a["weighted_bias_kwh"]),
                          f(b["weighted_mae_kwh"]), f(b["weighted_bias_kwh"])])
    top_rows = [[g, pct(group(g)["top_1pct_W_share"]), group(g)["top_1pct_times"],
                 pct(group(g)["top_5pct_W_share"]), group(g)["top_5pct_times"],
                 pct(group(g, "k=3")["top_5pct_W_share"]), group(g, "k=3")["top_5pct_times"]]
                for g in ("G1", "G2")]
    coherence_rows = [[r["group"], r["sites"], r["times"], r["finite_pairs"],
                       f(r["pair_cf_error_corr_mean"], 4), f(r["pair_cf_error_corr_median"], 4),
                       pct(r["weighted_net_direction_site_fraction"]), pct(group(r["group"], "k=3")["cancellation_ratio"])]
                      for r in coherent if r["sample"] == "fixed_148"]
    roundrows = [[p, f(r["W10"]), f(r["W17"]), f(r["Wmean"]), f"{(r['Wmean']/r['W17']-1)*100:+.4f}%"]
                 for p, r in pool_round.items()]
    round_coherence = [[r["group"], r["times"], f(r["error_correlation"], 4), pct(r["opposite_sign_fraction"]),
                        f(r["averaging_gain_pct"], 3)+"%"]
                       for r in rounds if r["sample"] == "fixed_148" and r["period"] == "k=3"]
    lines = ["# V1 집합 잔차와 라운드 진단", "", f"작성: {now()}", "",
             "상태: D1을 완료했다. 저장된 B0 V1 예측의 집합/개별 잔차와 라운드 보완 관계를 진단했으며 새 모델 학습, 새 예측 생성, 보정값 적합, 라운드 가중 탐색과 집합 재배정은 하지 않았다. V2는 파일 보존 해시만 검사했고 내용을 진단에 사용하지 않았다. 기본 EDA 종료와 선택 후보 3개/보류 5개는 유지했다.", "",
             "## 질문과 목적", "",
             "G1/G2에서 큰 오차가 반복되는 시간과 과대/과소 방향, 발전소 잔차 상쇄 정도 및 10시/17시 예측의 보완 관계를 확인한다. 한정된 잔차 진단으로 가장 먼저 줄일 오차 유형과 다음 개선 후보 하나를 제안하고 종료한다. [기존 V1](2026-10-19_20_B0_V1_기준선_학습과_평가.md), [기압 비교](2026-10-19_21_V1_기압_포함과_제외_비교.md)와 [V2 확인](2026-10-20_01_B0_V2_후반확인.md)은 완료된 기준 결과로 보존한다.", "",
             f"고정 148개소의 연간 W 중 종료 시각 11~15시(k=3)가 **{pct(high/annual)}**를 차지했다. 이 시간대의 10시/17시 집합 잔차 상관은 두 집합 모두 약 **0.944**이고 반대 부호 시각은 약 10~12%다. 집합 잔차 방향과 일치하는 비영 잔차 발전소 비율은 가격 가중 기준 약 70~73%다. 우선 줄일 대상은 **발전량이 큰 시간대에 여러 발전소와 두 라운드에서 함께 나타나는 오차**로 정했다. 그 생성 원인이 특정 기상 변수라고 판정한 것은 아니다.", "",
             "다음 후보는 **10시와 17시 기상예보를 한 모델에 공동 입력하는 B2**로 제안한다. 아래의 근거와 검증 조건을 기록했으며 B2를 구현하거나 성능을 확인하지 않았다.", "",
             "## 사용 범위와 수행 방법", "",
             "원본 메타, 학습 발전량과 학습 SMP, 기존 B0 V1의 저장된 라운드별/평균 예측을 사용했다. 새 기상 특징이나 V2 잔차는 읽지 않았다. V1 종료 ts는 2024-01-01 01:00~2025-01-01 00:00 KST다. 월/분기는 구간 시작 ts-1시간, 시각과 k는 KST 종료 시각으로 구분했다. 가격 가중 q=SMP×k를 유지했다. e_i=예측-정답이며 G1/G2 각각 E_g=Σe_i로 집계한다. 서로 다른 집합의 오차를 먼저 합치지 않는다.", "",
             "주 표본은 사후 진단 집단인 148개소의 연간 8,784시간이다. 전체 253개소는 모두 정답이 있는 2024-12-24 15시~2025-01-01 00시의 178시간에만 별도 집계했다. 정답 결손을 채우지 않고, 모든 예측 격자와 SMP=0 시각을 보존했다. 148개소 중 G1/G2 구성은 매우 불균형하므로 W 절댓값으로 집합의 예측 난이도를 비교하지 않는다.", "",
             table(["표본", "집합", "발전소", "용량 (kW)"], meta), "",
             "상쇄율은 1-Σq|Σe_i|/ΣqΣ|e_i|로 정의했다. 0%면 이 가중 집계에서 상쇄가 없고 100%면 집합 오차가 모두 상쇄된 경우다. 편향은 잔차의 평균이며 가격 가중 편향은 ΣqE/Σq다. 과대예측 W 비중은 Σq max(E,0)/W다. 분모가 0이면 정의하지 않는다.", "",
             "## 결과", "",
             "### 집합 오차와 방향", "",
             table(["표본", "집합", "발전소", "시각", "W", "가중 MAE (kWh)", "가중 편향 (kWh)", "과대예측 W 비중", "상쇄율"], overall_rows), "",
             "연간 고정 표본에서 G1은 과대, G2는 과소 방향의 가중 편향이 있으나 두 집합 모두 양방향 오차가 상당하다. 단순한 상수나 배율 보정으로 큰 오차를 해결할 수 있다는 근거는 아니다. 전체 253개소의 짧은 연말 표본에서는 두 집합 모두 과소 방향이 커져 연간의 한 방향을 전체 기간에 적용하기 어렵다.", "",
             "### 월별 방향, 고정 148개소", "",
             table(["월", "G1 가중 MAE", "G1 가중 편향", "G1 과대 W 비중", "G2 가중 MAE", "G2 가중 편향", "G2 과대 W 비중"], month_rows), "",
             "월별 편향과 오차 크기는 기술통계다. 각 월의 평균 잔차로 보정값을 적합하거나 해당 검증 월의 실적을 모델 입력으로 넣지 않았다. MAE와 편향 단위는 집합별 kWh다.", "",
             "### 종료 시각별 오차, 고정 148개소", "",
             table(["종료 시", "k", "연간 W 기여", "G1 가중 MAE", "G1 가중 편향", "G2 가중 MAE", "G2 가중 편향"], hour_rows), "",
             f"표의 W 기여는 두 집합 W를 더한 뒤 연간 W로 나눈 값이다. k=3인 1,830시간은 전체의 {pct(1830/8784)}이며 W의 {pct(high/annual)}다. 발전량 규모와 k=3 가중치 때문에 기여가 커질 수 있으므로 이것만으로 해당 시간대의 상대 정확도가 나쁘다고 단정하지 않는다. 전체 24시각의 집계는 로컬 파일에 저장했다.", "",
             "### 큰 오차 시각의 집중", "",
             table(["집합", "전체 상위 1% W 비중", "시각 수", "전체 상위 5% W 비중", "시각 수", "k=3 내부 상위 5% W 비중", "시각 수"], top_rows), "",
             "순위는 각 집합/범위 안에서 q|E|로 정했고 개수는 올림했다. 연간 전체에는 야간 저오차 시각이 포함되므로 전체 상위 비중과 k=3 내부 비중을 구분한다. 특정 몇 시각을 제거하거나 오차가 큰 날짜를 새 학습 표본으로 재사용하지 않았다.", "",
             "### 발전소 잔차의 상쇄와 동시 방향, 고정 148개소 k=3", "",
             table(["집합", "발전소", "시각", "유효 발전소 쌍", "cf 잔차 상관 평균", "상관 중앙값", "집합과 같은 방향 비율", "가중 상쇄율"], coherence_rows), "",
             "쌍별 상관은 각 발전소의 e_i/capacity_kw에 대한 Pearson 상관이며 k=3 시각만 사용했다. 계절/시각 평균을 제거하거나 특정 기상의 효과를 통제하지 않았으므로 원인이나 독립성 검정이 아니다. 집합과 같은 방향 비율은 시각마다 |e_i|>1e-8인 발전소 중 집합 잔차와 같은 부호인 비율을 계산해 q로 가중 평균한 값이다. |E|>1e-8인 시각만 정의한다. 원시 발전소 수를 단순 합한 비율과 다르다.", "",
             "상쇄는 실제로 존재하지만 그 뒤에도 여러 발전소의 같은 방향 오차가 남는다. G1/G2별 발전소 수와 정적 용량 차이가 커서 상쇄율 차이를 특정 구성의 우월성으로 단정하지 않는다. k=3의 발전소별 방향 기여 Σq e_i sign(E_g)는 로컬에 보존했고 그 합이 집합 W와 일치함을 검사했다. 음의 기여는 해당 시각들의 상쇄에 기여했다는 기술적 의미이며 발전소 제외의 효과가 아니다.", "",
             "### 두 라운드와 기존 평균, 고정 148개소", "",
             table(["기간", "10시 W", "17시 W", "기존 평균 W", "평균의 17시 대비 변화"], roundrows), "",
             f"연간 기존 평균은 17시 단독 W보다 {abs(delta17):.4f}% 낮았다. 분기별 방향은 표에서 별도로 확인하며, 이 진단으로 라운드를 선택하거나 평균 비율을 바꾸지 않았다. 라운드별 결과는 저장된 예측의 사후 진단이며 신규 학습 실험이 아니다.", "",
             table(["집합", "k=3 시각", "라운드 집합 잔차 상관", "반대 부호 시각 비율", "라운드 W 산술평균 대비 상쇄 이득"], round_coherence), "",
             "상쇄 이득은 ((W10+W17)/2-W평균)/((W10+W17)/2)다. 평균 모델이 더 나은 단일 라운드보다 이 비율만큼 좋다는 뜻은 아니다. 삼각부등식에 따른 서로 반대 방향의 라운드 오차 상쇄분을 측정한다. 잔차 상관과 반대 부호 비율은 가격 가중 없이 시각별로 계산했다. 반대 부호의 분모에서는 두 오차가 모두 사실상 0인 시각을 제외했고 k=3에서는 모든 1,830시간이 유효했다.", "",
             "## 다음 개선 후보 하나", "",
             "B2: 10시/17시 기상예보를 같은 발전소-시각 행의 별도 열로 함께 입력하는 공동 모델을 제안한다. 기상 15개씩 총 30개와 기존 태양/달력/메타 9개로 39개 특징을 쓰고, 기압은 두 라운드 모두 유지한다. 각 라운드 모델을 따로 만든 뒤 평균하는 현재 B0와 비교한다. 두 예보가 같은 대상일 전일 17시까지 이용 가능하다는 기존 배포 정의를 따른다.", "",
             "근거는 두 라운드가 발전량이 큰 시간에 함께 틀리는 경우가 많고, 현재 예측 평균이 17시 단독 대비 제공하는 추가 상쇄가 작다는 점이다. 공동 입력은 두 예보의 조합과 갱신 정보를 모델이 직접 활용할 수 있는 한 가지 후보다. 다만 같은 자료를 함께 넣어도 공통 오차가 남을 수 있으며 이번 수치가 B2의 개선을 증명하지는 않는다. 기상 변수의 생성 원인을 밝힌 것도 아니다.", "",
             "후속 비교를 진행한다면 B0 구성/학습 기간/목표 처리/LightGBM 설정을 유지하고 결손이 있어도 같은 학습 행과 예측 격자를 사용해야 한다. 공동 모델은 출력 하나를 만들므로 라운드별 두 번 대신 한 번 적합한다. 라운드 가중 탐색, 편향 보정, 공동 입력 외의 특징 추가와 구성 최적화는 동시에 하지 않는 안이다. 이 문서에서는 B2를 후보로만 제안했다.", "",
             "검증 설계는 실행 전에 별도로 정해야 한다. V1은 이미 B0/B1 선택과 이번 잔차 진단에 사용했으므로 새 B2의 V1 결과는 개발 비교로 표시한다. V2도 이미 점수를 확인했으므로 후보를 반복 고르는 구간으로 재사용하지 않는다. V2를 고정한 상태에서 새 방법의 일반화 확인에 사용할 추가 시간 분할과 학습 가용성을 점수를 보기 전에 정하고, 필요한 B0 재학습 예산도 사전 기록한다. 추가 시간 분할도 이미 전체 자료의 EDA를 수행한 과거 구간이므로 완전히 미관찰된 독립 시험이라고 부르지 않는다. 이번에는 분할 추가와 모델 실행을 하지 않았다.", "",
             "## 검증과 한계", "",
             f"정규 CPython {s['runtime']['python']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']}에 실행했고 종료 코드 0을 확인했다. 합성 테스트 {s['synthetic_tests']}개로 집합 내부 상쇄, 방향 합, 같은/반대 방향 라운드 평균, 0 가중 분모, 비정상 값과 삼각부등식 계약을 검사했다.", "",
             "원본 B0의 고정 148개소와 전체 253개소 공통 시각의 W/R/S/가격 가중 합/집합 MAE를 재현했다. 월/분기/24개 시각/k의 시간 수와 W 분할 합계를 검사했고, 저장된 라운드별 예측과 평균이 일치했다. 원본 10개 데이터와 README/baseline.py, EDA 계획 및 기존 B0/B1/V2 결과/모델/예측 18개 파일의 해시를 보존했다. 실행 코드 사본과 입력 해시는 로컬 산출물에 기록했다. 모델 라이브러리 학습/예측 호출은 없고 새 모델 적합/예측/가중 탐색 횟수는 모두 0이다.", "",
             "완료는 이번 D1의 세 질문과 후보 하나의 제안까지다. 원자료 EDA 목록은 재개하지 않았다. 상관과 방향 일치로 날씨, 설비 고장이나 공급사 처리 오류를 판정하지 않는다. 고정 148개소와 연말 178시간의 편향, 가격/발전량 가중, 정적 용량과 그룹, 공식 채점기 미확인의 한계는 유지한다. V1 결과로 관측이 없는 발전소나 최종 평가 연간 성능을 보장하지 않는다.", "",
             "## 관련 코드와 로컬 산출물", "",
             "- [D1 진단 코드](../scripts/v1_residual_diagnostics.py), [보고서 생성](../scripts/report_v1_residual_diagnostics.py), [합성 테스트](../tests/test_residual_diagnostics.py)",
             "- [입력/보존 해시와 집계 요약](../data/validation/2026-10-20_D1_V1_residuals/summary.json)",
             "- [집합별 월/시각/방향 진단](../data/validation/2026-10-20_D1_V1_residuals/group_diagnostics.csv)",
             "- [라운드 진단](../data/validation/2026-10-20_D1_V1_residuals/round_diagnostics.csv), [동시 방향/상관](../data/validation/2026-10-20_D1_V1_residuals/coherence.csv)",
             "- [발전소별 방향 기여](../data/validation/2026-10-20_D1_V1_residuals/site_contributions.csv)",
             "- [오늘 작업 로그](logs/2026-10-20.md), [EDA 종료와 보류 목록](EDA_진행계획.md)", "",
             "데이터가 담긴 개별 출력과 집계, 실행 코드 사본은 Git 제외 경로 data/에만 저장했다. 공개 보고서에는 원본/전처리 행, 개별 예측/잔차와 발전소 식별자를 복사하지 않았다. 보고서 재생성은 저장 집계를 사용하며 모델을 다시 실행하지 않는다.", ""]
    text = "\n".join(lines)
    require(not any(c in text for c in ("\u00b7", "\u2014", "\u2013")), "Forbidden punctuation")
    REPORT.write_text(text, encoding="utf-8")
    write_json(OUT / "report_manifest.json", dict(generated_at=now(), generator_sha256=digest(Path(__file__)),
               report_sha256=digest(REPORT), summary_sha256=digest(OUT / "summary.json"),
               proposed_candidate="B2_joint_round_inputs", candidate_executed=False))
    print(f"Report generated: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
