"""Report D2's arithmetic attribution and one unexecuted next hypothesis."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from b2_yearend_decomposition import OUT, REPORT, ROOT
from report_b0_v1 import f
from s0_validation_check import digest, now, table, write_json
from validation_core import require


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    require(s["status"]=="passed" and s["model_fit_count"]==s["model_prediction_count"]==0
            and not s["V2_reevaluated"],"Completed D2 required")
    for rel,h in s["protected_sha256"].items():
        require(digest(ROOT/rel)==h,"Protected result/EDA changed")
    for rel,h in s["code_sha256"].items():
        require(digest(OUT/"executed_code"/Path(rel).name)==h,"Executed code archive changed")
    rows=s["decomposition"]
    def get(model,g="both",period="all"):
        return next(r for r in rows if r["model"]==model and r["group"]==g and r["period"]==period)
    def diff(g="both",period="all"):
        return next(r for r in s["deltas"] if r["group"]==g and r["period"]==period)
    d=diff()
    total=d["delta_W_all"]
    b0,b2=get("B0"),get("B2")
    scores={}
    for r in s["cohort_scores"]:
        scores[(r["model"],r["cohort"])]=r
    cohort_table=[]
    for name in ("fixed_148","rest_105","all_253"):
        a,b=scores[("B0",name)],scores[("B2",name)]
        cohort_table.append([name,a["sites"],f(a["W_local"]),f(b["W_local"]),
                             f"{(b['W_local']/a['W_local']-1)*100:+.4f}%",
                             f(a["cf_mae"],6),f(b["cf_mae"],6),f(a["kwh_mae"],4),f(b["kwh_mae"],4)])
    pct=lambda v:f"{v/total*100:.2f}%"
    contrib=[["148개소 집합 W 증가",d["delta_W_fixed"],"+ΔW148"],
             ["105개소 집합 W 증가",d["delta_W_rest"],"+ΔW105"],
             ["집단 사이 상쇄 감소",-d["delta_cross_cancellation"],"-ΔC사이"]]
    fine=[["148개소 개별 가중 절대오차 증가",d["delta_abs_individual_fixed"]],
          ["105개소 개별 가중 절대오차 증가",d["delta_abs_individual_rest"]],
          ["148개소 내부 상쇄 변화",-d["delta_within_fixed"]],
          ["105개소 내부 상쇄 변화",-d["delta_within_rest"]],
          ["두 집단 사이 상쇄 변화",-d["delta_cross_cancellation"]]]
    group_table=[]
    bias=[]
    for g in ("G1","G2"):
        a,b=get("B0",g),get("B2",g)
        dg=diff(g)
        group_table.append([g,f(a["W_all"]),f(b["W_all"]),f"{dg['W_change_pct']:+.4f}%",
                            f(dg["delta_W_fixed"]),f(dg["delta_W_rest"]),f(-dg["delta_cross_cancellation"])])
        for cohort in ("fixed","rest","all"):
            bias.append([g,cohort,f(a["weighted_bias_"+cohort]),f(b["weighted_bias_"+cohort]),
                         f(a["over_W_"+cohort]),f(b["over_W_"+cohort]),
                         f(a["under_W_"+cohort]),f(b["under_W_"+cohort])])
    dayrows=[]
    for r in s["deltas"]:
        if r["group"]=="both" and r["period"].startswith("2024-"):
            dayrows.append([r["period"],get("B0",period=r["period"])["times"],
                            f(r["delta_W_fixed"]),f(r["delta_W_rest"]),f(-r["delta_cross_cancellation"]),
                            f(r["delta_W_all"]),f"{r['W_change_pct']:+.4f}%"])
    high=diff(period="k=3")
    lines=["# B2 연말 전체 집합 악화 분해","",f"작성: {now()}","",
        "상태: D2를 완료했다. 같은 V1 공통 178시간에서 B0/B2의 집단 오차와 상쇄 변화를 분해했다. 저장된 예측만 사용했으며 새 학습/예측, V2 재평가, 편향 보정과 집합 재배정은 하지 않았다. EDA 종료와 선택 후보 3개/보류 5개 및 기존 실험 결과를 유지했다.","",
        "## 질문과 목적","",
        "[B2 비교](2026-10-20_03_B2_공동입력_검증설계와_비교.md)에서 연간 고정 148개소는 개선됐으나 전체 253개소의 짧은 연말 표본에서는 W가 9.0231% 악화됐다. 동일 시간으로 맞춘 두 발전소 집단의 오차와 집단 사이 상쇄 변화로 이 증가분을 설명한다. 산술 분해 후 다음 실험 하나를 제안하고 종료하는 범위로 시작 전에 기록했다.","",
        f"전체 W 증가분 **{f(total)}**는 148개소 집합 W 증가 **{f(d['delta_W_fixed'])}**, 105개소 증가 **{f(d['delta_W_rest'])}**, 두 집단 사이 상쇄 감소 **{f(-d['delta_cross_cancellation'])}**로 분해된다. 같은 178시간에서는 학습 기록이 있는 148개소에서도 B2가 악화됐다. 따라서 학습 기록 없는 발전소의 문제나 상쇄 감소만으로 설명할 수 없다.","",
        "## 사용 범위와 수행 방법","",
        f"기존 공통 시각 마스크의 {s['first_ts']}~{s['last_ts']} 178시간을 B0/B2, 148개소/105개소에 동일 적용했다. 선택한 시각에는 모든 253개소의 원본 정답이 있다. ΣSMP×k는 {f(s['price_k_sum'])}다. G1/G2 참조 배정과 원본 정답을 유지했고 두 집합 사이에서는 잔차를 상쇄하지 않았다.","",
        table(["집단","집합","발전소","용량 (kW)","V1 학습 기록 있음","없음"],
              [[r["cohort"],r["group"],r["sites"],f(r["capacity_kw"],3),r["has_train_sites"],r["no_train_sites"]]
               for r in s["cohort_metadata"]]),"",
        "학습 기록은 V1의 s >= 2022-01-01, ts <= 2023-12-31 00:00 KST 기준으로 다시 확인했다. 148개소는 모두 해당 마감 전에 기록이 있었고 나머지 105개소에는 없었다. 이는 제공 기록의 상태이며 실제 신규 설비나 운영 개시 이력을 판정한 것은 아니다. 과거 두 집단의 용량/기간/그룹 구성도 다르다.","",
        "잔차 e=예측-정답, q=SMP×k다. 각 집합에서 E148=Σ148 e_i, E105=Σ105 e_i를 구하고 C사이=Σq(|E148|+|E105|-|E148+E105|)로 정의했다. 다음 항등식이 모델별, 집합별, k별과 날짜별로 맞는지 검사했다.","",
        "```text\nW전체 = W148 + W105 - C사이\nΔW전체 = ΔW148 + ΔW105 - ΔC사이\nW전체 = A148 + A105 - C내부148 - C내부105 - C사이\nA집단 = ΣqΣ집단|e_i|,  C내부집단 = A집단 - W집단\n```","",
        "Δ는 B2-B0다. 기여는 이 항등식에서의 계산상 기여이며 특정 원인이나 반사실적 모델 변경 효과를 뜻하지 않는다. 날짜는 구간 시작 ts-1시간의 날짜, k는 종료 시각 기준이다. 개별 cf MAE는 원본 capacity_kw로 나눈 잔차를 관측 행마다 같은 비중으로 집계한다.","",
        "## 결과","",
        "### 같은 178시간의 집단별 성능","",
        table(["집단","발전소","B0 W","B2 W","W 변화","B0 cf MAE","B2 cf MAE","B0 kWh MAE","B2 kWh MAE"],cohort_table),"",
        "연간 고정 148개소의 2.0986% 개선과 달리 동일한 연말 178시간에서는 그 148개소도 악화됐다. 따라서 연간/연말 결과의 차이에는 시간 조건이 함께 얽혀 있다. 나머지 105개소도 악화됐지만 이 집단만의 일반화 실패로 결론 내릴 수 없다. 두 집단의 개별 cf/kWh MAE도 이 구간에서 증가했다. 일부 표본의 MAE가 좋아졌는데 상쇄만 나빠진 상황은 이번 전체 집단 결과와 다르다.","",
        "### 전체 W 증가분의 산술 분해","",
        table(["항목","W 증가에 대한 부호","증가분 기여","전체 증가분 대비"],
              [[label,sign,f(value),pct(value)] for label,value,sign in contrib]),"",
        "여기서 두 집단의 W를 단순히 더하면 같은 G1/G2 안의 집단 사이 상쇄를 놓친다. 그 상쇄 감소분을 더해야 실제 전체 증가분이 된다. 반대로 G1/G2끼리의 상쇄는 허용하지 않는다.","",
        table(["더 세분한 항목","W 증가분 기여","전체 증가분 대비"],[[label,f(value),pct(value)] for label,value in fine]),"",
        "세분 표는 바로 위 표를 다시 풀어 쓴 것으로 두 표의 기여를 중복 합산하면 안 된다. 음의 기여는 전체 악화를 일부 줄인 항이다. 개별 절대오차 합은 대회 W가 아니며 내부 상쇄와 집단 사이 상쇄를 차감한 후에만 W와 일치한다.","",
        "### G1/G2별 변화","",
        table(["집합","B0 W","B2 W","W 변화율","148개소 W 변화","105개소 W 변화","집단 사이 상쇄 감소 기여"],group_table),"",
        "G1의 나머지 8개소 집합 W는 감소했지만 기존 126개소 악화와 집단 사이 상쇄 감소로 G1 전체는 악화됐다. G2는 두 집단 W가 모두 증가하고 집단 사이 상쇄도 줄었다. 집단별 발전소 수와 용량이 달라 원값 크기로 발전소의 상대 난이도를 비교하지 않는다.","",
        "### 잔차 방향","",
        table(["집합","집단","B0 가중 편향","B2 가중 편향","B0 과대 W","B2 과대 W","B0 과소 W","B2 과소 W"],bias),"",
        "편향은 ΣqE/Σq로 kWh 단위다. 전체 G1/G2의 음의 편향이 B2에서 더 커졌다. 일부 집단은 양의 편향을 보이므로 전체 실적에 하나의 배율을 곱하거나 이 178시간의 정답으로 보정값을 적합하지 않았다. 원자료의 기상/설비 원인을 특정한 결과도 아니다.","",
        "### 시간대와 날짜별 증가분","",
        table(["범위","전체 W 변화","148개소 W 변화","105개소 W 변화","집단 사이 상쇄 감소 기여"],
              [[p,f(diff(period=p)["delta_W_all"]),f(diff(period=p)["delta_W_fixed"]),
                f(diff(period=p)["delta_W_rest"]),f(-diff(period=p)["delta_cross_cancellation"])] for p in ("k=1","k=3")]),"",
        f"k=3의 36시각에서 전체 증가분의 {high['delta_W_all']/total*100:.2f}%가 발생했다. k=1은 142시각이다. 가격과 k 가중을 포함한 결과이며 실제 발전량 대비 상대 정확도의 비교는 아니다.","",
        table(["대상 날짜","시각 수","148개소 W 변화","105개소 W 변화","상쇄 감소 기여","전체 W 변화","그날 W 변화율"],dayrows),"",
        "12월 24일은 공통 정답이 시작한 날의 일부 시각만 포함한다. 모든 날짜가 악화된 것은 아니며 12월 30~31일의 큰 악화와 다른 날의 개선이 함께 합쳐진 결과다. 날짜별 시각 수와 각 분해 항의 합계가 전체와 일치함을 확인했다. 이 날짜를 보고 학습 표본을 골라내거나 규칙을 맞추지 않았다.","",
        "## 다음 실험 하나","",
        "B3 후보로 B2의 공동 입력 구조와 특징, 학습 표본 및 설정을 유지하고 학습 손실만 절대오차 손실 L1으로 바꾸는 비교를 제안한다. 이번에는 양 집단의 개별 절대오차도 증가했고, 여러 집단/두 집합의 오차와 상쇄 변화가 함께 작용했으므로 상쇄량만을 최적화하거나 학습 기록 없는 집단에만 임의 보정을 적용할 근거가 부족하다. 대회의 절대오차 목적에 가까운 학습 후보를 하나씩 비교하는 안이다.","",
        "이는 L1이 과소예측을 줄인다는 주장이나 개선 예측이 아니다. L1은 평균과 다른 중심을 학습할 수 있어 과소예측을 더 키울 가능성도 남는다. 발전소별 cf L1 역시 SMP×k로 가중한 집합 W와 일치하지 않는다. 따라서 후보를 시험할 때 최종 평가는 집합 W로 해야 한다. 손실 변경과 동시에 가격/용량 가중, 편향 보정, 라운드 비율과 구성 변경을 넣지 않는 안이다.","",
        "B3 실행 예산과 비교/판단 규칙은 후속 설계에서 먼저 정해야 한다. B2를 고정 비교 대상으로 삼고 B0 결과도 참조해 V1 개발 표본과 T0 과거 재현의 연간/분기 W를 확인하며, 이번 178시간에서는 G1/G2와 두 집단의 악화 여부를 별도 기록하는 방향이다. 이미 본 178시간만 개선되도록 선택하거나 V2를 반복 튜닝에 사용하지 않는다. 이 문서에서는 B3를 제안만 했고 실제 코드를 구현하거나 학습하지 않았다.","",
        "## 검증과 한계","",
        f"정규 CPython {s['runtime']['python']}, pandas {s['runtime']['pandas']}, NumPy {s['runtime']['numpy']}에서 {s['started_at']}~{s['finished_at']}에 실행했고 종료 코드 0을 확인했다. 합성 테스트 {s['synthetic_tests']}개로 두 집단 모두 좋아져도 상쇄 감소로 전체가 악화될 수 있는 예제, 내부/집단 사이 상쇄의 구분, 서로 다른 집합 오차의 비상쇄, 0 가격과 비정상 입력 처리를 확인했다.","",
        "B0/B2의 전체 253개소 W/R/S/가격 가중 합/시각 수가 기존 저장 결과와 일치했다. 같은 공통 시간의 두 집단 정답을 유지했고 분해 항등식과 k/날짜 분할 합계를 검사했다. 원본 10개 데이터 파일과 배포 README/baseline.py, 기존 B0/B1/V2와 B2 관련 보존 파일, D1 요약과 EDA 계획의 해시가 유지됐다. 저장 예측의 전체 2,222,352행과 키 유일성/유한/비음수도 확인했다.","",
        "집계 실행은 처음에 성공했다. 이후 보고서 생성 코드의 목록 괄호 문법 오류 1회를 수정하고 저장된 집계로 보고서 생성을 통과했다. 이 수정으로 진단 집계를 다시 실행하거나 모델을 학습하지 않았다.","",
        "새 적합과 모델 예측은 0회이며 기존 모델의 출력만 읽었다. 이 결과는 178시간의 산술 분해로 연간 전체 발전소 성능이나 악화의 기상/설비 원인을 확정하지 못한다. V1은 개발에 노출된 구간이고 148개소 선정과 그룹/용량 구성의 한계도 남는다. 기존 B2 유망 후보 판단을 최종 방법 확정으로 바꾸지 않았으며 EDA 종료/후보/보류 목록과 V2를 유지했다.","",
        "## 관련 코드와 로컬 산출물","",
        "- [D2 분해 코드](../scripts/b2_yearend_decomposition.py), [보고서 생성](../scripts/report_b2_yearend_decomposition.py), [합성 테스트](../tests/test_yearend_decomposition.py)",
        "- [해시, 표본과 집계 요약](../data/validation/2026-10-20_D2_B2_yearend/summary.json)",
        "- [집단별 점수](../data/validation/2026-10-20_D2_B2_yearend/cohort_scores.csv), [집단 메타 집계](../data/validation/2026-10-20_D2_B2_yearend/cohort_metadata.csv)",
        "- [집합/시간/날짜별 분해](../data/validation/2026-10-20_D2_B2_yearend/decomposition.csv), [B2-B0 변화](../data/validation/2026-10-20_D2_B2_yearend/delta.csv)",
        "- [오늘 로그](logs/2026-10-20.md), [EDA 종료와 보류 목록](EDA_진행계획.md)","",
        "데이터가 담긴 출력과 실행 코드 사본은 Git 제외 경로 data/에만 보존했다. 공개 보고서에는 원본/전처리 행, 개별 예측과 발전소 식별자를 복사하지 않았다. 보고서 재생성은 저장 집계로만 수행한다.",""]
    text="\n".join(lines)
    require(not any(c in text for c in ("\u00b7","\u2014","\u2013")),"Forbidden punctuation")
    REPORT.write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",dict(generated_at=now(),generator_sha256=digest(Path(__file__)),
        report_sha256=digest(REPORT),summary_sha256=digest(OUT/"summary.json"),next_candidate="B3_B2_L1_loss_only",candidate_executed=False))
    print(f"Report generated: {REPORT}",flush=True)


if __name__=="__main__":
    main()
