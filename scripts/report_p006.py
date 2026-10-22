"""Render verified P-006 aggregates only; never load row data or a model."""
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/2026-10-22_P006_site_category"
REPORT = ROOT / "reports/2026-10-22_03_P006_발전소ID_범주입력_비교.md"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows] + [""]


def number(x):
    return "해당 없음" if x is None else f"{x:,.2f}"


def main():
    pre, summary, verification = [read(n) for n in ("precheck.json", "summary.json", "verification.json")]
    assert summary["status"] == verification["status"] == "passed"
    assert summary["model_fit_count"] == summary["fit_attempts"] == 4
    assert summary["selection"] == verification["selection"]
    assert not summary["V2_executed"] and not summary["evaluation_executed"]
    lines = ["## 실제 입력 점검과 수행 결과", "",
             f"사전 점검: {pre['at']}. 실행: {summary['started_at']}~{summary['finished_at']}. 저장 검증: {verification['at']}.", "",
             "새 코드의 합성 계약 테스트 8개를 통과하고 실제 입력 계약을 고정한 뒤 후보를 구간별 한 번씩 총 4회 적합했다. 참조 재학습은 0회다. 실행과 저장 검증은 별도 프로세스에서 수행했다.", "",
             "### 가용성과 동일성", ""]
    lines += table(["구간", "관측 학습 행", "낮 학습 행", "학습 ID", "격자 미관측 ID", "주 표본 미관측 ID", "주 표본", "253 공통 시간"],
                   [[p["window"]["name"], f"{p['training']['train_rows']:,}", f"{p['training']['daylight_rows']:,}",
                     p["known_train_sites"], p["unseen_grid_sites"], p["unseen_cohort_sites"], p["cohort_sites"], p["common253_times"]]
                    for p in pre["periods"]])
    lines += ["원본 6개와 저장 B3의 모델/예측/학습 요약/기간 요약, 기존 특징 코드를 과거 동결 해시와 대조했다. 각 구간의 기존 39열, 학습 행과 경계, 자료형/NaN 행 수, 순서가 포함된 목표값 해시가 일치했다. 새 학습 ID 사전과 기존 특징 값/행 순서 해시도 실행 전에 고정했다.", "",
              "범주 사전은 실제 낮 학습 관측에서만 만들었다. 아래 범위는 ID별 낮 행 수와 관측 월 수의 최소/중앙/최대이며 발전량 성능 지표가 아니다.", ""]
    lines += table(["구간", "낮 행 수 최소/중앙/최대", "관측 월 수 최소/중앙/최대"],
                   [[p["window"]["name"],
                     "/".join(f"{v:,.0f}" for v in (min(r["daylight_rows"] for r in p["training"]["per_site"]), median(r["daylight_rows"] for r in p["training"]["per_site"]), max(r["daylight_rows"] for r in p["training"]["per_site"]))),
                     "/".join(str(v) for v in (min(r["months"] for r in p["training"]["per_site"]), median(r["months"] for r in p["training"]["per_site"]), max(r["months"] for r in p["training"]["per_site"])))] for p in pre["periods"]])
    lines += ["### 고정 주 기준의 결과", "", "변화율은 (P-006 W / B3 W - 1)×100이다. 음수가 개선이며 서로 다른 구간의 W 절댓값을 직접 비교하지 않는다.", ""]
    rows = []
    for p in summary["periods"]:
        c = p["comparison"]
        r = next(r for r in c["rows"] if r["sample"] == "fixed_cohort" and r["period"] == "all")
        rows.append([p["window"]["name"], number(r["B3_W"]), number(r["P006_W"]), f"{r['change_pct']:+.4f}%",
                     "감소" if c["improved"] else "미감소", ", ".join(c["unstable"]) or "없음", "통과" if c["passes"] else "미통과"])
    lines += table(["구간", "B3 W", "P-006 W", "변화", "전체 개선", "5% 초과 악화", "기간 기준"], rows)
    lines += [f"저장된 전체 선택은 **{summary['selection']}**다. 정답/가격/시각/구성이 같은지 확인했으며 저장 B3 점수도 기존 결과와 재현됐다.", "", "### 분기/월 안정성 세부", ""]
    for p in summary["periods"]:
        name = p["window"]["name"]
        prefix = "Q" if name in ("T0", "V1") else "M"
        lines += [f"{name}", ""]
        lines += table(["기간", "B3 W", "P-006 W", "변화"], [[r["period"], number(r["B3_W"]), number(r["P006_W"]), f"{r['change_pct']:+.4f}%"]
            for r in p["comparison"]["rows"] if r["sample"] == "fixed_cohort" and r["period"].startswith(prefix)])
    lines += ["### 보조 진단", "", "다음은 사전에 정한 진단이며 새 채택 기준이 아니다. 소집합 내부 W는 서로 합산하지 않는다.", ""]
    rows = []
    for p in summary["periods"]:
        a = {(r["cohort"], r["group"], r["k"]): r for r in p["diagnostics"]["B3"]["group_rows"]}
        for r in p["diagnostics"]["P006"]["group_rows"]:
            if r["cohort"] not in ("complete", "fixed_148", "added_complete"):
                continue
            if r["cohort"] != "complete" and r["k"] != "all":
                continue
            b = a[(r["cohort"], r["group"], r["k"])]
            change = (r["W"]/b["W"]-1)*100 if b["W"] else None
            rows.append([p["window"]["name"], r["cohort"], r["group"], r["k"], number(b["W"]), number(r["W"]),
                         f"{change:+.4f}%" if change is not None else "해당 없음", number(r["over_W"]), number(r["under_W"])])
    lines += table(["구간", "표본", "집합", "k", "B3 W", "P-006 W", "변화", "P-006 과대 비용", "P-006 과소 비용"], rows)
    lines += ["학습 ID 관측/미관측의 아래 값은 전체 격자 중 실제 정답이 있는 개별 행의 MAE다. 서로 다른 관측 행 구성이므로 주 W나 연간 전체 집합 성능으로 대체하지 않는다. 결측 정답은 0으로 채우지 않았다.", ""]
    rows = []
    for p in summary["periods"]:
        old = {r["category"]: r for r in p["diagnostics"]["B3"]["observed_id_errors"]}
        for r in p["diagnostics"]["P006"]["observed_id_errors"]:
            rows.append([p["window"]["name"], r["category"], r["sites"], f"{r['observed_site_hours']:,}",
                         number(old[r["category"]]["individual_mae_kwh"]), number(r["individual_mae_kwh"])])
    lines += table(["구간", "학습 ID", "발전소", "관측 행", "B3 MAE kWh", "P-006 MAE kWh"], rows)
    lines += ["전체 253개소의 공통 정답 시각에 한정한 보조 비교는 다음과 같다. 표에 없는 구간은 공통 시각이 없어 W를 정의하지 않았다.", ""]
    lines += table(["구간", "공통 시간", "B3 W", "P-006 W", "변화"],
                   [[p["window"]["name"], r["times"], number(r["B3_W"]), number(r["P006_W"]), f"{r['change_pct']:+.4f}%"]
                    for p in summary["periods"] for r in p["comparison"]["rows"] if r["sample"] == "common_253" and r["period"] == "all"])
    lines += ["### 적합과 저장 검증", ""]
    lines += table(["구간", "적합 횟수", "학습 초", "ID 분기 횟수", "저장 전후 점검 행"],
                   [[p["window"]["name"], p["fit"]["model_fit_count"], f"{p['fit']['fit_seconds']:.2f}",
                     p["fit"]["id_split_count"], p["fit"]["reload_probe_rows"]] for p in summary["periods"]])
    lines += ["ID 분기 횟수는 입력이 사용됐는지를 보여 줄 뿐, 중요도의 인과 해석이나 유효성/과적합 판정 지표가 아니다.", "",
              "각 모델의 600트리/40열/명목 범주 인덱스 39/범주 사전/L1/범주 기본값을 저장 문자열과 재로드 모델에서 확인했다. 253개소의 개발 시각 4개로 만든 1,012행에서 저장 전후, 역순과 배치 분할 예측을 대조했다. 별도 검증 프로세스에서 입력을 재구성하고 저장 예측과 다시 비교했다. 상대 허용오차 1e-12, 절대 허용오차 0을 사용했다.", "",
              "전체 개발 예측 격자의 키, 유한/비음수와 해시를 검증했다. 저장 점수/채택 판단과 그룹/시간 가중/과대/과소 진단을 재현했으며 월/분기/k 분할 합계도 점검했다. 저장 검증에서 새 적합은 0회다. V2와 제공 평가 구간의 새 실행은 없었다.", "",
              "실행 환경: " + ", ".join(f"{k}={v}" for k, v in summary["runtime"].items()) + ".", ""]
    base = (OUT / "preregistered_report.md").read_text(encoding="utf-8")
    base = base.replace("상태: 실제 입력 점검과 모델 비교 진행.", "상태: 실제 입력 점검, 네 구간 적합과 저장 검증 완료.")
    REPORT.write_text(base + "\n" + "\n".join(lines), encoding="utf-8")
    print("P006 verified aggregate report written")


if __name__ == "__main__":
    main()
