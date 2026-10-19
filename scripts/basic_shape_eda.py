"""Univariate, read-only EDA. No relationships, prediction, or model training.

Run from regular CPython 3.13. Data-bearing outputs stay under ignored data/.
Only aggregate descriptions are written to the Markdown analysis report.
"""
from __future__ import annotations

import gc
import hashlib
import html
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "OIBC_2026_DATA" / "01_participant_dataset"
OUT = ROOT / "data" / "eda" / "2026-10-19_basic_shape"
FIG = OUT / "figures"
REPORT = ROOT / "reports" / "2026-10-19_02_데이터_형태_EDA.md"
FILES = ["sites.csv", "smp.csv", "train/smp.csv", "train/generation.parquet",
         "train/forecast_wa.parquet", "train/forecast_wb.parquet",
         "train/observed_wa.parquet", "train/observed_wb.parquet",
         "test/forecast_wa.parquet", "test/forecast_wb.parquet"]
KEYS = {
    "sites.csv": ["site_id"], "smp.csv": ["ts"], "train/smp.csv": ["ts"],
    "train/generation.parquet": ["site_id", "ts"],
}
QUANTILES = [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]
NUMERIC = []
SCHEMAS = []
GALLERY = []
SUMMARY = {"scope": "file structure and univariate distributions only",
           "excluded": ["relationships", "correlations", "prediction", "training",
                        "group optimization", "submission"], "files": {}}
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
COLORS = ["#26718c", "#db8653", "#6a9170", "#8a76a4"]


def log(message):
    print(f"[{datetime.now(ZoneInfo('Asia/Seoul')).strftime('%H:%M:%S')}] {message}", flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tag(rel):
    return rel.replace("/", "_").replace(".parquet", "").replace(".csv", "")


def fmt(value):
    if value is None or pd.isna(value):
        return "없음"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{float(value):,.6g}"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in rows])


def save(fig, name, title, note, file):
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    GALLERY.append({"path": f"figures/{name}.png", "title": title,
                    "note": note, "file": file})


def plot_numeric(d, columns, rel, profiles):
    """Every panel describes one numeric column, never a pair of variables."""
    for start in range(0, len(columns), 4):
        cols = columns[start:start + 4]
        page = start // 4 + 1
        for kind in ["hist", "box", "ecdf"]:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
            for ax, col in zip(axes.flat, cols):
                a = d[col].to_numpy(dtype="float64", na_value=np.nan)
                a = a[np.isfinite(a)]
                p = profiles[col]
                color = COLORS[cols.index(col)]
                if not len(a):
                    ax.text(.5, .5, "유한한 값 없음", ha="center")
                    continue
                lo, hi = p["q001"], p["q999"]
                if kind == "hist":
                    ax.hist(a, bins=60, color=color, edgecolor="white", linewidth=.25)
                    ax.set_yscale("log")
                    ax.set_ylabel("행 수 (로그 축)")
                    ax.set_xlabel(col)
                    ax.set_title(f"{col}\n전체 범위, 유한값 {len(a):,}행", fontsize=10)
                    if lo < hi and (p["min"] < lo or p["max"] > hi):
                        inset = ax.inset_axes([.48, .44, .49, .42])
                        inside = a[(a >= lo) & (a <= hi)]
                        inset.hist(inside, bins=40, color=color, alpha=.85)
                        inset.set_yscale("log")
                        inset.set_title("0.1~99.9 백분위 확대", fontsize=7)
                        inset.tick_params(labelsize=6)
                        inset.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
                elif kind == "box":
                    q1, q3 = p["q25"], p["q75"]
                    iqr = q3 - q1
                    low = a[a >= q1 - 1.5 * iqr].min()
                    high = a[a <= q3 + 1.5 * iqr].max()
                    ax.bxp([{"med": p["q50"], "q1": q1, "q3": q3,
                             "whislo": float(low), "whishi": float(high), "fliers": []}],
                           vert=False, showfliers=False, patch_artist=True,
                           boxprops={"facecolor": color, "alpha": .65})
                    ax.set_yticks([])
                    ax.set_xlabel(col)
                    ax.set_title(f"{col}\nIQR 밖 {p['iqr_outside']:,}행, 표식 생략", fontsize=10)
                    ax.text(.02, .90, f"전체 최소 {fmt(p['min'])}, 최대 {fmt(p['max'])}",
                            transform=ax.transAxes, fontsize=8)
                else:
                    ax.plot(p["ecdf_x"], p["ecdf_y"], color=color, lw=1.7)
                    ax.set_ylim(0, 1.02)
                    ax.set_ylabel("누적 비율")
                    ax.set_xlabel(col)
                    ax.set_title(f"{col}\n501개 분위수로 표시한 누적분포", fontsize=10)
                ax.grid(axis="y", alpha=.18)
            for ax in list(axes.flat)[len(cols):]:
                ax.set_visible(False)
            labels = {"hist": "히스토그램", "box": "상자그림", "ecdf": "누적분포"}
            title = f"{rel} / {labels[kind]} / {page}"
            fig.suptitle(title, fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, .95])
            notes = {
                "hist": "전체 유한값을 집계. 세로축 로그. 작은 삽입 그림만 꼬리를 제외해 확대하며 원자료는 바꾸지 않음.",
                "box": "전체 유한값의 사분위수와 1.5 IQR 수염. 수염 밖 행의 표식을 생략했으며 제거하지 않음. IQR 밖이라는 이유만으로 오류로 판정하지 않음.",
                "ecdf": "전체 유한값에서 계산한 501개 분위수를 연결한 근사 표시. 원자료를 표본 추출한 결과가 아님.",
            }
            save(fig, f"{tag(rel)}_{kind}_{page:02d}", title, notes[kind], rel)


def structural_plot(values, title, name, label):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(values, bins=min(40, max(5, len(np.unique(values)))), color=COLORS[0], edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel("발전소 수")
    fig.tight_layout()
    save(fig, name, title, "발전량과 다른 변수의 관계가 아닌 기록의 가용성만 요약함.", "기록 가용성")


def inspect_file(rel):
    path = SOURCE / rel
    d = pd.read_csv(path, encoding="utf-8-sig") if path.suffix == ".csv" else pd.read_parquet(path)
    log(f"읽기 완료: {rel}, {len(d):,}행 x {len(d.columns)}열")
    if "ts" in d:
        d["ts"] = pd.to_datetime(d.ts)
    keys = KEYS.get(rel)
    if keys is None:
        station = "wa_station" if "wa_station" in d else "wb_station"
        keys = [station] + (["round"] if "round" in d else []) + ["ts"]
    info = {"rows": len(d), "columns": len(d.columns), "bytes": path.stat().st_size,
            "keys": keys, "duplicate_keys": int(d.duplicated(keys).sum()),
            "key_null_rows": int(d[keys].isna().any(axis=1).sum()),
            "columns_with_null": int(d.isna().any().sum()),
            "null_cells": int(d.isna().sum().sum()), "numeric": {}}
    number_cols = list(d.select_dtypes(include="number").columns)
    real_cols = [c for c in number_cols if c != "round"]
    for col in d:
        SCHEMAS.append({"file": rel, "column": col, "dtype": str(d[col].dtype),
                        "null_count": int(d[col].isna().sum()),
                        "unique_nonnull": int(d[col].nunique())})
    for col in real_cols:
        a = d[col].to_numpy(dtype="float64", na_value=np.nan)
        finite = a[np.isfinite(a)]
        qs = np.quantile(finite, QUANTILES) if len(finite) else np.full(len(QUANTILES), np.nan)
        names = ["min", "q001", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q999", "max"]
        p = {name: float(value) for name, value in zip(names, qs)}
        iqr = p["q75"] - p["q25"]
        p.update(count=int(len(finite)), nulls=int(d[col].isna().sum()),
                 infinite=int(np.isinf(a).sum()), zeros=int((finite == 0).sum()),
                 negative=int((finite < 0).sum()), mean=float(finite.mean()),
                 std=float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                 unique=int(d[col].nunique()),
                 iqr_outside=int(((finite < p["q25"] - 1.5 * iqr) |
                                  (finite > p["q75"] + 1.5 * iqr)).sum()),
                 ecdf_x=np.quantile(finite, np.linspace(0, 1, 501)).tolist(),
                 ecdf_y=np.linspace(0, 1, 501).tolist())
        info["numeric"][col] = p
        NUMERIC.append({"file": rel, "column": col,
                        **{k: v for k, v in p.items() if not k.startswith("ecdf_")}})
    categories = [c for c in d if c != "ts" and c not in real_cols]
    info["categories"] = {}
    for col in categories:
        counts = d[col].value_counts(dropna=False)
        info["categories"][col] = {"levels": int(d[col].nunique()),
                                    "min_frequency": int(counts.min()),
                                    "max_frequency": int(counts.max())}
        counts.rename_axis(col).rename("rows").to_csv(OUT / f"{tag(rel)}_{col}_counts.csv", encoding="utf-8-sig")
        if col in ("round", "current_group"):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(counts.index.astype(str), counts.values, color=COLORS[0])
            ax.set_title(f"{rel} / {col} / 행 수")
            ax.set_ylabel("행 수")
            for i, value in enumerate(counts.values):
                ax.text(i, value, f"{value:,}", ha="center", va="bottom")
            ax.margins(y=.15)
            fig.tight_layout()
            save(fig, f"{tag(rel)}_{col}_counts", f"{rel} / {col} 빈도", "범주형 키 또는 참조 배정의 빈도만 집계.", rel)
    if "ts" in d:
        times = pd.DatetimeIndex(d.ts.dropna().unique()).sort_values()
        expected = pd.date_range(times.min(), times.max(), freq="h")
        info["time"] = {"start": str(times.min()), "end": str(times.max()),
                        "timezone": str(d.ts.dt.tz), "unique": len(times),
                        "global_missing_hours": len(expected.difference(times)),
                        "off_hour": int(((d.ts.dt.minute != 0) | (d.ts.dt.second != 0) |
                                         (d.ts.dt.microsecond != 0) | (d.ts.dt.nanosecond != 0)).sum())}
        group_keys = [k for k in keys if k != "ts"]
        if group_keys:
            coverage = d.groupby(group_keys, observed=True).agg(rows=("ts", "size"),
                       first_ts=("ts", "min"), last_ts=("ts", "max"), unique_ts=("ts", "nunique"))
            coverage["span_hours"] = ((coverage.last_ts - coverage.first_ts).dt.total_seconds() / 3600 + 1).astype(int)
            coverage["within_span_absent"] = coverage.span_hours - coverage.unique_ts
            coverage.to_csv(OUT / f"{tag(rel)}_coverage.csv", encoding="utf-8-sig")
            info["coverage"] = {"groups": len(coverage), "rows_min": int(coverage.rows.min()),
                                "rows_median": float(coverage.rows.median()), "rows_max": int(coverage.rows.max()),
                                "within_span_absent": int(coverage.within_span_absent.sum())}
        if rel == "train/generation.parquet":
            train_times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(SOURCE / "train/smp.csv", encoding="utf-8-sig").ts))
            before = ((coverage.first_ts - train_times.min()).dt.total_seconds() / 3600).clip(lower=0).sum()
            after = ((train_times.max() - coverage.last_ts).dt.total_seconds() / 3600).clip(lower=0).sum()
            present = d.groupby("ts").site_id.nunique().reindex(train_times, fill_value=0)
            info["generation_availability"] = {
                "expected_full_grid": len(train_times) * len(coverage),
                "absent_full_grid": len(train_times) * len(coverage) - len(d),
                "before_first_record": int(before), "after_last_record": int(after),
                "within_record_span": int(coverage.within_span_absent.sum()),
                "latest_first_record": str(coverage.first_ts.max()),
                "all_sites_observed_hours": int((present == len(coverage)).sum())}
            assert info["generation_availability"]["absent_full_grid"] == int(before + after + coverage.within_span_absent.sum())
            structural_plot(coverage.rows.to_numpy(), "발전소별 실적 기록 수의 분포", "generation_record_count", "발전소당 기록 수 (행)")
            structural_plot(coverage.within_span_absent.to_numpy(), "최초~마지막 실적 기록 사이 빈 시간 수의 분포", "generation_internal_absence", "발전소당 빈 시간 수")
    if rel.startswith("test/forecast"):
        station = keys[0]
        score_times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(SOURCE / "smp.csv", encoding="utf-8-sig").ts))
        expected_index = pd.MultiIndex.from_product([sorted(d[station].unique()), [10, 17], score_times], names=keys)
        actual = pd.MultiIndex.from_frame(d[keys])
        missing = expected_index.difference(actual)
        info["evaluation_grid"] = {"expected": len(expected_index), "absent": len(missing),
                                    "unexpected": len(actual.difference(expected_index))}
    plot_numeric(d, real_cols, rel, info["numeric"])
    SUMMARY["files"][rel] = info
    del d
    gc.collect()
    log(f"단변량 분포 완료: {rel}")


def build_gallery():
    cards = []
    for g in GALLERY:
        cards.append(f'<section data-file="{html.escape(g["file"])}"><h2>{html.escape(g["title"])}</h2>'
                     f'<p>{html.escape(g["note"])}</p><a href="{g["path"]}"><img loading="lazy" '
                     f'src="{g["path"]}" alt="{html.escape(g["title"])}"></a></section>')
    options = "".join(f'<option>{html.escape(rel)}</option>' for rel in [*FILES, "기록 가용성"])
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OIBC 데이터 형태 EDA</title><style>body{font:16px/1.65 'Malgun Gothic',sans-serif;background:#f4f5f5;color:#24323b;margin:0}
header,main{max-width:1150px;margin:auto;padding:24px}header{position:sticky;top:0;background:#f4f5f5ed;z-index:2;border-bottom:1px solid #ddd}
h1{font-size:26px;margin:0}h2{font-size:18px}section{background:white;padding:22px;margin:24px 0;border:1px solid #ddd;border-radius:8px}
img{max-width:100%;height:auto}select{font:inherit;padding:8px;max-width:100%}p{margin:8px 0;color:#55616b}a{color:#206b85}</style>
<header><h1>OIBC 데이터 형태 EDA</h1><p>구조, 키, 결측, 개별 변수의 분포만 확인. 변수 간 관계, 상관분석, 머신러닝 없음.</p>
<label>파일 선택 <select id="filter"><option value="all">전체 그림</option>''' + options + '''</select></label>
<p>각 그림을 누르면 원본 크기로 열립니다. 그래프와 상세 집계는 로컬 전용입니다.</p></header><main>''' + "".join(cards) + '''</main>
<script>document.getElementById('filter').addEventListener('change',function(){document.querySelectorAll('section').forEach(s=>s.hidden=this.value!=='all'&&s.dataset.file!==this.value)})</script></html>'''
    (OUT / "index.html").write_text(page, encoding="utf-8")


def build_report():
    f = SUMMARY["files"]
    energy = f["train/generation.parquet"]["numeric"]["energy"]
    availability = f["train/generation.parquet"]["generation_availability"]
    chunks = ["# 기본 EDA: 데이터 형태와 개별 변수의 분포", "",
              f"작성: {SUMMARY['started_at']}. 상태: 실제 데이터 읽기와 단변량 EDA 실행 완료.", "",
              "## 질문과 목적", "",
              "어떤 파일과 열이 있고, 자료형과 키는 무엇이며, 기록이 얼마나 비어 있고 개별 수치가 어떻게 분포하는지 확인한다. 사용자의 범위 정정에 따라 변수 간 관계, 상관분석, 발전량과 날씨 연결, 예측 및 머신러닝 학습은 수행하지 않았다.", "",
              "## 사용 범위와 수행 방법", "",
              "배포 폴더의 CSV 3개와 Parquet 7개를 모두 읽었다. 원본은 수정하지 않았고, 실행 전후 10개 입력 파일의 SHA-256이 같은지 확인했다. 원본 행을 보고서에 복사하지 않았다.", "",
              "- 구조: 행과 열 수, 실제 자료형, 비결측 고유값 수, 정의된 키의 중복과 키 결측.",
              "- 가용성: 파일별 시작과 종료, 시간대, 정시 여부, 고유 시각과 빈 시간, 발전소 또는 지점별 기록 범위. 최초 기록 이전을 결측 원인이 확인된 장애로 해석하지 않는다.",
              "- 단변량 요약: 유한값 수, 결측, 무한대, 0과 음수, 평균과 표준편차, 최소와 최대, 0.1/1/5/25/50/75/95/99/99.9 백분위.",
              "- 그림: 모든 비키 수치 열에 전체 범위 히스토그램, 상자그림, 누적분포를 작성했다. round와 참조 그룹은 빈도 막대그림으로 다뤘다. 두 수치 변수의 산점도, 상관행렬, 변수별 결합 분석은 만들지 않았다.",
              "- 분포 계산에는 전체 유한값을 사용했다. 히스토그램은 로그 세로축을 쓰며, 꼬리가 있는 경우 0.1~99.9 백분위 구간을 삽입 그림으로 확대했다. 상자그림은 수염 밖 표식만 생략했다. 누적분포는 전체 값의 501개 분위수로 표시한 근사 곡선이다.",
              "- 서로 다른 단위의 열은 별도 축에 그렸다. IQR 밖 행은 분포 설명용이며 오류 판정이나 제거 기준으로 사용하지 않았다.", "",
              f"실행 환경: CPython {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, pyarrow {pyarrow.__version__}, matplotlib {matplotlib.__version__}. 정규 CPython 3.13의 Start-Process 경로로 실행했다.", "",
              "## 파일 구조", "",
              md_table(["파일", "행", "열", "디스크 MiB", "키", "추가 중복행", "값 결측 셀"],
                       [[rel, fmt(v["rows"]), v["columns"], f'{v["bytes"]/1024**2:.2f}', ", ".join(v["keys"]), fmt(v["duplicate_keys"]), fmt(v["null_cells"])] for rel, v in f.items()]), "",
              "CSV는 텍스트 형식, Parquet는 열 단위 저장 형식이다. 같은 시각이 여러 번 나오는 발전량과 기상 파일은 여러 발전소 또는 지점을 담은 긴 형식이며, ts 단독 중복은 오류 기준이 아니다. 위 표의 복합 키로 판정했다.", "",
              "## 시간 구조와 기록 가용성", "",
              "시각은 모두 KST이며 우측닫힘을 의도한 데이터다. 예를 들어 ts=01:00은 앞선 00:00~01:00 구간을 뜻한다. 다만 학습 예보 B의 비정시 기록은 아래 표와 별도 검사에서 구분한다. round=10과 17은 대상일 전날 발행 시각을 나타내며 연속 수치 변수로 해석하지 않는다.", "",
              md_table(["파일", "시작", "종료", "고유 시각", "전체 범위 내 시각 누락", "비정시 행"],
                       [[rel, v["time"]["start"], v["time"]["end"], fmt(v["time"]["unique"]), fmt(v["time"]["global_missing_hours"]), fmt(v["time"]["off_hour"])] for rel, v in f.items() if "time" in v]), "",
              "전체 범위 내 시각 누락은 그 시각에 어떤 발전소나 지점의 행도 없을 때만 센다. 0이어도 특정 발전소나 지점의 기록이 비어 있을 수 있다.", "",
              "학습 forecast_wb에서 정시가 아닌 기록 92행을 확인했다. 기존 검사 JSON에도 92행으로 기록되어 있었으나 기존 확인 보고서의 '분이나 초가 0이 아닌 행은 없다'는 설명이 잘못되어 함께 정정했다. 나머지 파일의 시간 열에는 비정시 행이 없다. 원본 시각은 반올림하거나 바꾸지 않았다.", "",
              f"발전 실적의 전체 학습 시간축 격자 {fmt(availability['expected_full_grid'])}개 중 없는 조합은 {fmt(availability['absent_full_grid'])}개다. 각 발전소의 최초 기록 이전 {fmt(availability['before_first_record'])}개, 마지막 기록 이후 {fmt(availability['after_last_record'])}개, 최초~마지막 기록 사이 {fmt(availability['within_record_span'])}개로 구분했다. 이것은 기록 위치의 분류이며 원인의 분류가 아니다.", "",
              f"253개 발전소의 기록이 동시에 존재하는 시각은 {fmt(availability['all_sites_observed_hours'])}개다. 가장 늦은 발전소의 최초 기록은 {availability['latest_first_record']}다. 이 사실만으로 검증 구간이나 학습 구간을 확정하지 않았다.", "",
              "## 개별 변수의 분포에서 확인한 사실", "",
              f"발전량 energy의 중앙값은 {fmt(energy['q50'])} kWh, 95백분위는 {fmt(energy['q95'])} kWh다. 0인 행은 {fmt(energy['zeros'])}개({energy['zeros']/energy['count']:.2%}), 음수는 {fmt(energy['negative'])}개다. 최소 {fmt(energy['min'])}, 최대 {fmt(energy['max'])} kWh이며, 원인을 추정하거나 값을 보정하지 않았다.", "",
              f"설비용량 capacity_kw는 최소 {fmt(f['sites.csv']['numeric']['capacity_kw']['min'])}, 중앙값 {fmt(f['sites.csv']['numeric']['capacity_kw']['q50'])}, 최대 {fmt(f['sites.csv']['numeric']['capacity_kw']['max'])} kW다. 발전량을 용량으로 나누거나 두 변수의 관계를 분석하지 않았다.", "",
              f"학습 A 예보의 강우량 wa_rain은 유한값 중 {f['train/forecast_wa.parquet']['numeric']['wa_rain']['zeros']/f['train/forecast_wa.parquet']['numeric']['wa_rain']['count']:.2%}, 강설량 wa_snow는 {f['train/forecast_wa.parquet']['numeric']['wa_snow']['zeros']/f['train/forecast_wa.parquet']['numeric']['wa_snow']['count']:.2%}가 0이다. 이처럼 0에 몰린 열은 상자그림이 선으로 보일 수 있어 히스토그램과 누적분포를 함께 확인한다.", "",
              f"학습 A 실측 wa_wind_speed의 최대는 {fmt(f['train/observed_wa.parquet']['numeric']['wa_wind_speed']['max'])} m/s, 학습 B 실측 wb_cloud의 최대는 {fmt(f['train/observed_wb.parquet']['numeric']['wb_cloud']['max'])}%다. 중심부에서 멀리 떨어진 값도 그대로 포함해 집계했다. 발생 원인을 확정하거나 보정하지 않았다.", "",
              "값이 0인 행은 실제로 존재하는 기록이다. 값 NaN, 키 조합 자체가 없는 경우, 중복 키는 별개로 집계했다. 기온 등 음수가 정상일 수 있는 열까지 음수 건수를 집계했으므로 음수라는 이유만으로 이상값으로 해석하지 않는다.", "",
              "### 값 결측과 키 문제", "",
              "- 학습 observed_wa의 동일 지점과 시각 키에 추가 중복행 6개가 있다. 현재 EDA에서는 원본 행을 유지했으므로 해당 분포에 중복 기록이 포함된다.",
              "- 학습 forecast_wb의 강수 관련 8개 열에 값 결측이 있다. 열마다 개별 결측을 집계했으며 행 전체를 제거하지 않았다.",
              f"- 평가 forecast_wa는 정의된 전체 격자에서 {fmt(f['test/forecast_wa.parquet']['evaluation_grid']['absent'])}개 조합이 없다. 평가 forecast_wb는 {fmt(f['test/forecast_wb.parquet']['evaluation_grid']['absent'])}개다. 존재하는 행의 값 결측 여부와 구분한다.", "",
              "### 수치 요약 전체", "",
              "아래는 원본 행이 아닌 열별 집계다. round는 아래 연속형 요약에서 제외하고 범주 빈도로 기록했다. 결측 수는 NaN/빈 셀, 0 비율은 유한값 기준이다.", ""]
    for rel in FILES:
        rows = [p for p in NUMERIC if p["file"] == rel]
        chunks += [f"#### {rel}", "", md_table(["열", "최소", "25%", "중앙값", "75%", "99%", "최대", "0 비율", "값 결측"],
                   [[p["column"], fmt(p["min"]), fmt(p["q25"]), fmt(p["q50"]), fmt(p["q75"]), fmt(p["q99"]), fmt(p["max"]), f'{p["zeros"]/p["count"]:.2%}', fmt(p["nulls"])] for p in rows]), ""]
    chunks += ["## 자료형과 열별 가용성", "", md_table(["파일", "열", "자료형", "비결측 고유값", "값 결측"],
               [[p["file"], p["column"], p["dtype"], fmt(p["unique_nonnull"]), fmt(p["null_count"])] for p in SCHEMAS]), "",
               "## 그래프와 상세 집계", "",
               f"총 {len(GALLERY)}개 그림을 로컬 갤러리에 저장했다. 각 그림 안의 패널은 개별 변수만 설명한다.", "",
               "- [그래프 갤러리](../data/eda/2026-10-19_basic_shape/index.html)",
               "- [상세 요약 JSON](../data/eda/2026-10-19_basic_shape/summary.json)",
               "- [열별 수치 통계 CSV](../data/eda/2026-10-19_basic_shape/numeric_summary.csv)",
               "- [자료형과 결측 CSV](../data/eda/2026-10-19_basic_shape/schema.csv)", "",
               "그래프, HTML, JSON, CSV는 데이터가 담긴 출력으로 보아 Git에서 제외한 data/ 아래에만 둔다. 위 로컬 링크는 GitHub에 출력 파일이 없으므로 원격에서는 열리지 않는다. 코드를 로컬에서 다시 실행하면 생성된다.", "",
               "## 한계와 다음 단계", "",
               "전체 행을 그대로 집계한 기상 분포에는 여러 지점, 두 예보 라운드, 서로 다른 기간이 섞여 있다. 분포만으로 제공사 차이, 예보 정확도, 계절 효과 또는 발전량에 대한 영향을 판단하지 않는다. 발전 실적 분포도 관측 기간과 발전소 크기가 서로 다른 기록의 혼합이다.", "",
               "본 세션은 데이터 형태 확인만 수행했다. 변수 간 관계, 상관분석, 시계열 패턴 해석, 전처리 기준 확정, 머신러닝 학습, 예측, 집합 구성 최적화, 제출 파일 생성과 공식 제출은 하지 않았다. 다음 작업의 범위는 사용자가 정한다.", "",
               "## 관련 코드와 실행", "",
               "- [데이터 형태 EDA 코드](../scripts/basic_shape_eda.py)",
               "- [시각 정밀도 대조 코드](../scripts/inspect_time_precision.py)",
               "- [산출물 점검 코드](../scripts/verify_shape_eda.py)",
               "- [기존 제공 데이터 검사 결과](../데이터_확인_결과.md)",
               "- [오늘 작업 로그](logs/2026-10-19.md)", "",
               "Windows Codex에서는 정규 Python 3.13을 기본 샌드박스에서 먼저 실행하지 않는다. 첫 실행부터 require_escalated 도구 호출에서 다음 PowerShell 명령을 사용한다.", "",
               "```powershell", "$p = Start-Process -FilePath 'C:/Program Files/Python313/python.exe' -ArgumentList @('-u', 'scripts/basic_shape_eda.py') -WorkingDirectory 'D:/대회/OIBC-2026' -Wait -PassThru -NoNewWindow", "exit $p.ExitCode", "```", "",
               "보조 검사는 같은 실행 경로에서 inspect_time_precision.py와 verify_shape_eda.py를 실행한다. --report-only 옵션은 저장된 집계로 보고서만 다시 생성하며, 원본 재분석이나 그래프 재생성은 하지 않는다.", ""]
    precision_path = OUT / "time_precision.json"
    if precision_path.is_file():
        precision = json.loads(precision_path.read_text(encoding="utf-8"))
        pos = chunks.index("## 개별 변수의 분포에서 확인한 사실")
        chunks[pos:pos] = [
            f"시각 정밀도 추가 대조: 비정시 92행은 18개 시각과 24개 지점에 걸쳐 있으며 round=17이다. 가장 가까운 정시와의 거리는 {precision['distance_to_nearest_hour_ns_min']/1e9:g}~{precision['distance_to_nearest_hour_ns_max']/1e9:g}초다. pandas의 시간 성분과 Arrow 원시 정수 연산 모두 비정시 92행으로 일치했다. 반올림은 진단용 비교에만 사용했고, 조인이나 원본에는 적용하지 않았다.", ""]
    verification_path = OUT / "qa" / "verification.json"
    if verification_path.is_file():
        check = json.loads(verification_path.read_text(encoding="utf-8"))
        pos = chunks.index("## 한계와 다음 단계")
        chunks[pos:pos] = [f"산출물 검사: {check['numeric_profiles_checked']}개 파일별 수치 열의 건수 합계, 분위수 순서와 누적분포 단조성을 확인했고, {check['figures_checked']}개 PNG의 읽기 및 갤러리의 {check['html_links_checked']}개 로컬 링크 검사를 통과했다. 그림 검토용 모음은 원본의 50% 크기, 한 장당 최대 4개로 만들었다.", ""]
    REPORT.write_text("\n".join(chunks), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    assert sys.version_info[:2] == (3, 13), "This project requires regular CPython 3.13."
    assert not sysconfig_free_threaded(), "Free-threaded runtime is not permitted."
    if "--report-only" in sys.argv:
        SUMMARY.update(json.loads((OUT / "summary.json").read_text(encoding="utf-8")))
        GALLERY.extend(json.loads((OUT / "gallery.json").read_text(encoding="utf-8")))
        NUMERIC.extend(pd.read_csv(OUT / "numeric_summary.csv", encoding="utf-8-sig").to_dict("records"))
        SCHEMAS.extend(pd.read_csv(OUT / "schema.csv", encoding="utf-8-sig").to_dict("records"))
        build_report()
        log("Report rebuilt from saved aggregate outputs; source data and figures not reread.")
        return
    SUMMARY["started_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes")
    SUMMARY["runtime"] = {"executable": sys.executable, "python": sys.version,
                          "pandas": pd.__version__, "numpy": np.__version__,
                          "pyarrow": pyarrow.__version__, "matplotlib": matplotlib.__version__}
    log(f"Runtime started: {sys.executable}, {platform.python_version()}")
    before = {rel: digest(SOURCE / rel) for rel in FILES}
    for rel in FILES:
        inspect_file(rel)
    after = {rel: digest(SOURCE / rel) for rel in FILES}
    assert before == after, "Source files changed during the read-only EDA."
    SUMMARY["source_sha256"] = before
    SUMMARY["source_unchanged"] = True
    SUMMARY["figure_count"] = len(GALLERY)
    SUMMARY["finished_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes")
    pd.DataFrame(NUMERIC).to_csv(OUT / "numeric_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(SCHEMAS).to_csv(OUT / "schema.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "gallery.json").write_text(json.dumps(GALLERY, ensure_ascii=False, indent=2), encoding="utf-8")
    build_gallery()
    build_report()
    assert len(list(FIG.glob("*.png"))) == len(GALLERY)
    assert not any("\u00b7" in path.read_text(encoding="utf-8") for path in [REPORT, OUT / "index.html", Path(__file__)])
    log(f"Completed: {len(FILES)} input files, {len(SCHEMAS)} columns, {len(NUMERIC)} numeric profiles, {len(GALLERY)} figures")
    log(f"Report: {REPORT}")


def sysconfig_free_threaded():
    import sysconfig
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


if __name__ == "__main__":
    main()
