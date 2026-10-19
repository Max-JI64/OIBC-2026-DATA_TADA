"""Descriptive relationships of generation, rated capacity, hour, and month.

No fitted predictive model, forecast generation, or official scoring.
Only local ignored data/ receives data-bearing tables and plots.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from basic_shape_eda import ROOT, SOURCE, digest, fmt, md_table, log

OUT = ROOT / "data/eda/2026-10-19_generation_relationships"
FIG = OUT / "figures"
REPORT = ROOT / "reports/2026-10-19_05_발전량과_용량_시간_월의_관계_EDA.md"
INPUTS = ["sites.csv", "train/generation.parquet", "train/smp.csv"]
GALLERY = []
TABLES = {}
INFO = {}
COLORS = ["#26718c", "#d7804f", "#6a8c68", "#86739f"]
CAP_BINS = [0, 50, 100, 300, 1000, np.inf]
CAP_LABELS = ["<50", "50~<100", "100~<300", "300~<1000", ">=1000"]


def save(fig, name, title, note):
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    GALLERY.append({"file": f"figures/{name}.png", "title": title, "note": note})


def table(name, frame):
    TABLES[name] = frame
    frame.to_csv(OUT / f"{name}.csv", index=True, encoding="utf-8-sig")


def site_summary(d):
    s = d.groupby("site_id", observed=True).agg(capacity_kw=("capacity_kw", "first"),
         n=("energy", "size"), mean_energy=("energy", "mean"), mean_cf=("cf", "mean"),
         median_cf=("cf", "median"), q95_energy=("energy", lambda x: x.quantile(.95)),
         zero_share=("is_zero", "mean"), first=("ts", "min"), last=("ts", "max"))
    s["capacity_bin"] = pd.cut(s.capacity_kw, CAP_BINS, labels=CAP_LABELS, right=False)
    return s


def hourly(d):
    h = d.groupby("hour", observed=True).agg(mean_cf=("cf", "mean"),
         median_cf=("cf", "median"), q25_cf=("cf", lambda x: x.quantile(.25)),
         q75_cf=("cf", lambda x: x.quantile(.75)), sum_energy=("energy", "sum"),
         sum_capacity=("capacity_kw", "sum"), n=("energy", "size"),
         sites=("site_id", "nunique"), zero_share=("is_zero", "mean"))
    h["capacity_weighted_cf"] = h.sum_energy / h.sum_capacity
    h["equal_site_cf"] = d.groupby(["site_id", "hour"], observed=True).cf.mean().groupby("hour").mean()
    return h.reindex(range(24))


def monthly(d, key="month"):
    m = d.groupby(key, observed=True).agg(mean_cf=("cf", "mean"), sum_energy=("energy", "sum"),
         sum_capacity=("capacity_kw", "sum"), n=("energy", "size"), sites=("site_id", "nunique"))
    m["capacity_weighted_cf"] = m.sum_energy / m.sum_capacity
    m["equal_site_cf"] = d.groupby(["site_id", key], observed=True).cf.mean().groupby(key).mean()
    return m


def correlation(s, name):
    pairs = [("capacity_kw", "mean_energy"), ("capacity_kw", "q95_energy"),
             ("capacity_kw", "mean_cf")]
    result = []
    for x, y in pairs:
        a = s[[x, y]].dropna()
        result.append({"window": name, "x": x, "y": y, "sites": len(a),
                       "pearson": float(a[x].corr(a[y], method="pearson")),
                       "spearman": float(a[x].corr(a[y], method="spearman"))})
    return result


def capacity_bin_summary(s):
    return s.groupby("capacity_bin", observed=True).agg(sites=("mean_cf", "size"),
        median_site_cf=("mean_cf", "median"), mean_site_cf=("mean_cf", "mean")).reindex(CAP_LABELS)


def plot_capacity(site_sets, full):
    for variable, ylabel, name in [("mean_energy", "발전소별 관측 시간당 평균 발전량 (kWh)", "01_capacity_mean_energy"),
                                   ("mean_cf", "발전소별 관측 시간당 평균 용량 대비 발전 비율", "02_capacity_mean_cf")]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, (label, s) in zip(axes, site_sets.items()):
            ax.scatter(s.capacity_kw, s[variable], s=24, alpha=.65, color=COLORS[0], edgecolor="none")
            ax.set_xlabel("설비용량 (kW)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label}\n발전소 {len(s):,}개")
            ax.grid(alpha=.15)
        save(fig, name, "설비용량과 발전소별 평균의 관계", "점 하나는 발전소 하나. 전체 기간은 발전소마다 관측 기간이 다름. 회귀선을 적합하지 않음.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, ylabel in zip(axes, ["energy", "cf"], ["시간별 발전량 (kWh)", "용량 대비 발전 비율"]):
        hb = ax.hexbin(full.capacity_kw, full[col], gridsize=55, mincnt=1, bins="log", cmap="viridis")
        fig.colorbar(hb, ax=ax, label="행 수 (로그 색상)")
        ax.set_xlabel("설비용량 (kW)")
        ax.set_ylabel(ylabel)
    save(fig, "03_capacity_hourly_hexbin", "설비용량과 개별 시간 기록의 관계", "음수 발전량을 제외한 전체 관측 행을 집계. 같은 발전소의 반복 기록은 독립 표본이 아님.")
    s = site_sets["2024 고정 집단"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    values, labels = [], []
    for label in CAP_LABELS:
        v = s.loc[s.capacity_bin == label, "mean_cf"].to_numpy()
        if len(v):
            values.append(v)
            labels.append(f"{label}\n발전소 {len(v)}개")
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_xlabel("설비용량 구간 (kW)")
    ax.set_ylabel("발전소별 연평균 용량 대비 발전 비율")
    save(fig, "04_capacity_bins", "2024 고정 집단의 용량 구간별 분포", "분석 단위는 발전소. 구간별 표본 수가 다르며, 차이를 용량의 인과 효과로 해석하지 않음.")


def plot_time(data_sets, hours, months2024, daily2024, daily_month, yearmonth, availability, heat_cf, heat_zero):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (label, h) in zip(axes, hours.items()):
        for col, title, color in [("mean_cf", "관측 행 동일 가중", COLORS[0]),
                                  ("equal_site_cf", "발전소 동일 가중", COLORS[1]),
                                  ("capacity_weighted_cf", "용량 가중", COLORS[2])]:
            ax.plot(h.index, h[col], label=title, color=color)
        ax.set_title(label)
        ax.set_xlabel("구간 시작 시각 (KST)")
        ax.set_ylabel("평균 용량 대비 발전 비율")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    save(fig, "05_hourly_profiles", "시간대별 발전 패턴과 집계 방식", "0시는 00:00~01:00 구간. 없는 기록은 0으로 채우지 않음. 같은 시각의 기상 차이를 통제하지 않은 기술통계.")
    h = hours["2024 고정 집단"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(h.index, h.mean_cf, color=COLORS[0], label="평균")
    ax.plot(h.index, h.median_cf, color=COLORS[1], label="중앙값")
    ax.fill_between(h.index, h.q25_cf, h.q75_cf, alpha=.20, color=COLORS[0], label="25~75백분위")
    ax.set_xticks(range(24))
    ax.set_xlabel("구간 시작 시각 (KST)")
    ax.set_ylabel("용량 대비 발전 비율")
    ax.legend()
    save(fig, "06_hourly_spread", "2024 고정 집단의 시간대별 중심과 산포", "평균과 중앙값의 차이 및 관측 행 분포. 신뢰구간이 아님.")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for color, (label, h) in zip(COLORS, hours.items()):
        ax.plot(h.index, h.zero_share*100, color=color, label=label)
    ax.set_xticks(range(24))
    ax.set_xlabel("구간 시작 시각 (KST)")
    ax.set_ylabel("정확한 0인 행 (%)")
    ax.legend()
    save(fig, "07_hourly_zeros", "시간대별 정확한 0 비율", "0의 물리적 원인을 확정하지 않음. 근사 0은 정확한 0과 구분.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for col, label, color in [("mean_cf", "관측 행 동일 가중", COLORS[0]),
                              ("equal_site_cf", "발전소 동일 가중", COLORS[1]),
                              ("capacity_weighted_cf", "용량 가중", COLORS[2])]:
        axes[0].plot(months2024.index, months2024[col], marker="o", label=label, color=color)
    axes[0].set_xlabel("2024년 월")
    axes[0].set_ylabel("월평균 용량 대비 발전 비율")
    axes[0].set_xticks(range(1, 13))
    axes[0].legend(fontsize=8)
    axes[1].plot(daily_month.index, daily_month["median"], marker="o", color=COLORS[0])
    axes[1].fill_between(daily_month.index, daily_month.q25, daily_month.q75, color=COLORS[0], alpha=.2)
    axes[1].set_xlabel("2024년 월")
    axes[1].set_ylabel("완전한 발전소 하루의 발전량 / 용량 (kWh/kW)")
    axes[1].set_xticks(range(1, 13))
    save(fig, "08_monthly_fixed_cohort", "2024년 고정 집단의 월별 패턴", "음수 기록을 제외. 오른쪽은 24개 유효 시간 기록이 있는 발전소 하루만 사용한 중앙값과 IQR.")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    x = np.arange(len(yearmonth))
    axes[0].plot(x, yearmonth.sum_energy/1e6, color=COLORS[0])
    axes[0].set_ylabel("관측된 발전량 합계 (GWh)")
    axes[1].plot(x, yearmonth.n/1e6, color=COLORS[1])
    axes[1].set_ylabel("기록 수 (백만 행)")
    tick = np.arange(0, len(yearmonth), 3)
    axes[1].set_xticks(tick, yearmonth.index[tick], rotation=45, ha="right")
    save(fig, "09_observed_monthly_totals", "전체 기간의 관측 합계와 기록량", "발전소 구성과 기록량이 변하므로 위 합계를 계절 효과나 동일 설비의 성장으로 해석하지 않음.")
    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(availability, aspect="auto", vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    tick = np.arange(0, availability.shape[1], 3)
    ax.set_xticks(tick, availability.columns[tick], rotation=45, ha="right")
    ax.set_ylabel("발전소 순서 (ID 정렬)")
    fig.colorbar(image, ax=ax, label="해당 달의 기록 가용 비율")
    save(fig, "10_observation_coverage", "발전소별 월간 기록 가용성", "0은 그 달에 기록이 없다는 의미이며 발전량 0이 아님. 설비 편입 원인을 확정하지 않음.")
    for name, matrix, title, label in [("11_month_hour_cf", heat_cf, "2024 고정 집단의 월과 시간대별 평균", "평균 용량 대비 발전 비율"),
                                        ("12_month_hour_zero", heat_zero*100, "2024 고정 집단의 월과 시간대별 0 비율", "정확한 0 (%)")]:
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(12), range(1, 13))
        ax.set_xticks(range(24), range(24))
        ax.set_xlabel("구간 시작 시각 (KST)")
        ax.set_ylabel("2024년 월")
        fig.colorbar(im, ax=ax, label=label)
        save(fig, name, title, "월과 시간대의 관측 패턴. 날씨나 설비 특성의 인과 효과를 분리한 결과가 아님.")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(daily2024.index, daily2024["mean"], color=COLORS[0], lw=1, label="발전소별 일일 값의 평균")
    ax.fill_between(daily2024.index, daily2024.q25, daily2024.q75, color=COLORS[0], alpha=.18, label="발전소 간 IQR")
    ax.set_ylabel("일일 발전량 / 용량 (kWh/kW)")
    ax.legend(fontsize=8)
    save(fig, "13_daily_variation", "2024년 고정 집단의 일별 변동", "완전한 발전소 하루만 사용. 선의 변화 원인을 날씨로 확정하지 않음.")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    raw = TABLES["negative_sensitivity"].set_index("window")
    x = np.arange(len(raw))
    ax.bar(x-.18, raw.raw_mean_cf, width=.36, color=COLORS[0], label="음수 포함")
    ax.bar(x+.18, raw.nonnegative_mean_cf, width=.36, color=COLORS[1], label="음수만 제외")
    ax.set_xticks(x, raw.index)
    ax.set_ylabel("관측 행 평균 용량 대비 발전 비율")
    ax.legend()
    save(fig, "14_negative_sensitivity", "음수 발전량 표시 조건에 대한 민감도", "전체 평균의 차이가 작아도 개별 행의 정확성을 보장하지 않음. 원본은 변경하지 않음.")


def write_gallery():
    sections = "".join(f'<section><h2>{g["title"]}</h2><p>{g["note"]}</p><a href="{g["file"]}"><img src="{g["file"]}" loading="lazy" alt="{g["title"]}"></a></section>' for g in GALLERY)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>발전량과 용량, 시간, 월의 관계 EDA</title><style>body{font:16px/1.65 'Malgun Gothic',sans-serif;background:#f4f5f5;color:#26333c;margin:0}header,main{max-width:1150px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:19px}p{color:#53616b}section{background:white;padding:22px;margin:22px 0;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}</style>
<header><h1>발전량과 용량, 시간, 월의 관계 EDA</h1><p>기술통계와 관계 그림. 모델 학습과 예측, 인과 효과 추정은 수행하지 않았습니다.</p></header><main>''' + sections + "</main></html>"
    (OUT / "index.html").write_text(page, encoding="utf-8")
    qa = OUT / "qa"
    qa.mkdir(exist_ok=True)
    for start in range(0, len(GALLERY), 4):
        group = GALLERY[start:start+4]
        pictures = []
        for g in group:
            with Image.open(OUT/g["file"]) as im:
                im.load()
                pictures.append(im.convert("RGB").resize((im.width//2, im.height//2), Image.Resampling.LANCZOS))
        w, h = max(p.width for p in pictures), max(p.height for p in pictures)+25
        sheet = Image.new("RGB", (2*w, 2*h), "#ededed")
        pen = ImageDraw.Draw(sheet)
        for i, (im, g) in enumerate(zip(pictures, group)):
            x, y = (i%2)*w, (i//2)*h
            sheet.paste(im, (x, y+25))
            pen.text((x+5, y+4), Path(g["file"]).stem, fill="black")
        sheet.save(qa/f"contact_{start//4+1:02d}.png")


def report():
    stats = INFO
    corr = TABLES["site_correlations"]
    h = TABLES["hour_2024"]
    m = TABLES["month_2024"]
    fixed_corr = corr.loc[corr.window == "2024 고정 집단"].set_index("y")
    p = ["# 발전량과 설비용량, 시간대, 월의 관계 EDA", "",
         f"작성: {stats['finished_at']}. 상태: 실제 관측 자료를 이용한 관계 EDA 실행 완료.", "",
         "## 질문과 목적", "",
         "발전량의 규모 차이가 설비용량과 어떻게 연결되는지, 용량으로 나눈 뒤에도 차이가 남는지, 시간대와 월에 따라 어떤 관측 패턴이 나타나는지 확인한다. 첫 관계 EDA의 범위를 발전량, 용량과 달력 시간으로 한정했다. 기상 변수와 발전량, 예보와 실측의 관계는 다음 단계이며 이번에는 분석하지 않았다.", "",
         "## 사용 범위와 방법", "",
         "sites.csv, train/generation.parquet와 학습 시간축 확인용 train/smp.csv를 사용했다. 발전소 메타 키의 유일성과 many_to_one 조인, 조인 전후 행 수, 원본 해시 일치를 검사했다. 기상 파일을 조인하지 않아 이전에 발견한 실측 중복과 비정시 예보를 이번 분석에 섞지 않았다.", "",
         "- 시간별 발전량 energy(kWh)를 설비용량(kW)과 1시간으로 나눈 cf를 용량 대비 발전 비율로 정의했다. 1시간 자료이므로 수치상 energy/capacity_kw와 같다. 정적 메타 용량을 과거 전 기간에 적용한다는 가정이 있다.",
         "- 원본 음수 40행을 유지한 민감도 집계를 별도로 만들고, 주된 관계 그림은 음수 기록만 제외했다. 0, 근사 0과 없는 기록은 서로 구분했고 결측을 0으로 채우지 않았다. 양의 큰 값은 임의 상한으로 자르지 않았다.",
         "- ts는 구간 종료 시각이다. ts에서 1시간을 뺀 구간 시작으로 시간대, 날짜와 월을 정했다. 따라서 00:00으로 끝나는 기록은 전날과 전월에 속할 수 있다.",
         "- 최근 완전한 달력 연도인 2024년을 고정 집단의 기준으로 정했다. 그 해의 8,784시간 기록이 모두 존재하는 발전소를 선택했다. 발전량의 크기나 상관계수를 보고 집단을 고르지 않았다.",
         "- 전체 관측 기간, 2024년 고정 집단, 253개소가 모두 관측된 공통 시각을 비교했다. 각 집단의 기간과 발전소 구성이 다르므로 숫자 차이를 동일 집단의 전후 효과로 해석하지 않는다.",
         "- 발전소별 평균을 이용한 Pearson/Spearman 상관을 계산했다. 같은 발전소의 수천 행을 독립 발전소로 세지 않았고 p값이나 예측 성능을 보고하지 않았다.",
         "- 관측 행 동일 가중, 발전소 동일 가중, 용량 가중 평균을 나눴다. 용량 가중 비율의 분모에는 실제 기록이 있는 발전소의 용량만 시간별로 포함한다.", "",
         "## 관측 범위와 품질", "",
         md_table(["항목", "결과"], [["원본 발전량 행", fmt(stats["source_rows"])], ["메타 조인 후 행", fmt(stats["joined_rows"])],
                  ["음수 제외 행", fmt(stats["nonnegative_rows"])], ["2024 고정 집단 발전소", fmt(stats["fixed_sites"])],
                  ["2024 원본 기록", fmt(stats["fixed_raw_rows"])], ["2024 주된 집계 기록", fmt(stats["fixed_valid_rows"])],
                  ["253개소 공통 시각", fmt(stats["common_times"])], ["공통 시각 범위", stats["common_start"]+" ~ "+stats["common_end"]],
                  ["cf>1 기록", fmt(stats["cf_gt1"])], ["cf>1.2 기록", fmt(stats["cf_gt1_2"])] ]), "",
         "cf가 1보다 크다는 사실만으로 오류나 제거 대상으로 확정하지 않았다. 참조값 1.2도 진단용 문턱일 뿐 이번 분석의 제거 조건이 아니다. 원본과 음수 제외의 결과는 로컬 민감도 표에서 확인할 수 있다.", "",
         "## 1. 설비용량과 발전량", "",
         md_table(["비교 구간", "비교 변수", "발전소 수", "Pearson", "Spearman"],
                  [[r.window, r.x+" / "+r.y, r.sites, f"{r.pearson:.4f}", f"{r.spearman:.4f}"] for r in corr.itertuples()]), "",
         "mean_energy는 발전소별 관측 시간 평균 발전량, q95_energy는 관측 발전량의 95백분위, mean_cf는 관측 시간 평균 용량 대비 비율이다. 전체 기간의 각 발전소 평균에는 관측 연도와 계절 차이가 함께 들어 있다. 2024 고정 집단과 공통 시각 비교로 그 영향을 구분해 보았지만 설비 방향, 기상, 운영 상태를 통제한 인과 분석은 아니다.", "",
         f"2024 고정 집단에서 용량과 평균 발전량의 Pearson 상관은 {fixed_corr.loc['mean_energy', 'pearson']:.4f}로 높다. 반면 용량과 평균 cf의 Spearman 상관은 {fixed_corr.loc['mean_cf', 'spearman']:.4f}로, 단조로운 순위 관계는 약하다. 같은 비교의 Pearson 값({fixed_corr.loc['mean_cf', 'pearson']:.4f})과 구분해서 해석해야 한다. 큰 발전소의 원시 발전량이 크다는 사실과 큰 발전소의 용량 대비 성과가 좋다는 주장은 별개의 내용이다.", "",
         md_table(["설비용량 구간 kW", "발전소 수", "발전소별 연평균 cf의 중앙값", "발전소별 연평균 cf의 평균"],
                  [[i, int(r.sites), f"{r.median_site_cf:.4f}", f"{r.mean_site_cf:.4f}"] for i, r in TABLES['capacity_bins_2024'].iterrows()]), "",
         "구간별 발전소 수의 차이가 크다. 특히 1,000 kW 이상 구간은 발전소 1개이므로 대표적인 대규모 발전소의 분포로 볼 수 없다. 구간별 상자그림의 모양만으로 효율 순위를 정하지 않는다.", "",
         "용량으로 나누면 규모 차이를 줄인 비교가 가능하다. 다만 cf에는 용량이 분모로 들어가므로 capacity_kw와 cf의 상관을 용량의 순수한 영향으로 해석할 수 없다. 어느 발전소가 더 효율적이거나 어떤 용량이 우수하다는 결론도 내리지 않았다.", "",
         "## 2. 시간대별 패턴", "",
         f"2024 고정 집단의 관측 행 평균 cf가 가장 높은 구간은 {int(h.mean_cf.idxmax()):02d}:00~{int(h.mean_cf.idxmax())+1:02d}:00 KST이고, 평균 cf는 {h.mean_cf.max():.4f}다. 용량 가중 평균의 최대 구간 시작 시각은 {int(h.capacity_weighted_cf.idxmax())}시다. 이는 시간별 관측 패턴이며 다음 날 예측 결과가 아니다.", "",
         md_table(["구간 시작 시각", "평균 cf", "중앙값 cf", "용량 가중 cf", "정확한 0 비율"],
                  [[i, f"{r.mean_cf:.4f}", f"{r.median_cf:.4f}", f"{r.capacity_weighted_cf:.4f}", f"{r.zero_share:.2%}"] for i, r in h.iterrows()]), "",
         "0 비율은 해당 시각에 기록된 값 중 정확히 0인 비율이다. 계측 이상, 야간, 정지 등의 물리적 원인을 여기서 분류하지 않았다. 평균과 중앙값, IQR은 분포 설명이며 독립 표본을 가정한 신뢰구간이 아니다.", "",
         "## 3. 월별 패턴과 기록량의 영향", "",
         f"2024 고정 집단의 관측 행 평균 cf는 {int(m.mean_cf.idxmax())}월에 가장 높고({m.mean_cf.max():.4f}), {int(m.mean_cf.idxmin())}월에 가장 낮다({m.mean_cf.min():.4f}). 용량 가중 기준 최대 월은 {int(m.capacity_weighted_cf.idxmax())}월이다. 다른 연도의 일반적인 계절 순서나 원인을 확정한 결과는 아니다.", "",
         md_table(["2024년 월", "발전소 수", "유효 기록", "평균 cf", "용량 가중 cf", "발전소 동일 가중 cf"],
                  [[i, int(r.sites), fmt(int(r.n)), f"{r.mean_cf:.4f}", f"{r.capacity_weighted_cf:.4f}", f"{r.equal_site_cf:.4f}"] for i, r in m.iterrows()]), "",
         "전체 기간의 월별 합계에는 관측되는 발전소 수와 기록량이 함께 변한다. 이 합계로 계절 효과를 판단하지 않고, 같은 발전소가 1년간 기록된 고정 집단의 정규화 비율과 완전한 하루의 발전량/용량을 함께 봤다. 월별 일수와 하루 기록 수의 차이도 총량 비교와 구분했다.", "",
         "## 한계와 다음 단계", "",
         "기술통계와 관계 그림만 작성했다. 기록이 완전한 발전소만 고른 2024 집단에도 선택 편향이 있을 수 있으며 253개 전체를 대표한다고 보장하지 않는다. 253개소 공통 시각은 최근의 제한된 계절을 포함하므로 연간 계절성 판단에 사용하지 않는다. 정적 용량, 익명화된 위치, 미확인 운영 상태와 계측 원인을 고려해야 한다.", "",
         "기상 변수를 연결할 때는 원시 발전량과 용량 대비 비율, 전체 시각과 같은 시간대의 관계를 구분해야 한다. 시각과 발전소 구성을 맞추고 기상 실측 중복 및 비정시 예보를 처리하는 것은 다음 EDA의 별도 범위다. 이번에는 머신러닝 학습, 예측, 집합 최적화, 제출 파일 생성과 공식 제출을 하지 않았다.", "",
         "## 관련 코드와 로컬 결과", "",
         "- [관계 EDA 코드](../scripts/generation_relationship_eda.py)",
         "- [그래프 갤러리](../data/eda/2026-10-19_generation_relationships/index.html)",
         "- [집계와 범위 JSON](../data/eda/2026-10-19_generation_relationships/summary.json)",
         "- [발전소 단위 상관계수](../data/eda/2026-10-19_generation_relationships/site_correlations.csv)",
         "- [2024 월별 집계](../data/eda/2026-10-19_generation_relationships/month_2024.csv)",
         "- [시간축과 집계 및 산출물 검증](../data/eda/2026-10-19_generation_relationships/verification.json)",
         "- [앞선 품질 점검](2026-10-19_04_값의_의미와_데이터_품질_점검.md)", "",
         "상세 표, 그림과 HTML은 데이터가 담긴 출력으로 보아 로컬 data/에만 보관한다. Git에는 코드와 요약 보고서, 일별 로그만 포함한다.", ""]
    if (OUT/"visual_review.md").is_file():
        p += ["시각 검토: 그림 14개를 원본의 50% 크기로 한 장당 최대 4개씩 모은 검토본 4장에서 모두 확인했다. 원본 전체 크기의 픽셀 검사를 의미하지 않는다.", ""]
    REPORT.write_text("\n".join(p), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    assert sys.version_info[:2] == (3, 13)
    FIG.mkdir(parents=True, exist_ok=True)
    if "--report-only" in sys.argv:
        INFO.update(json.loads((OUT/"summary.json").read_text(encoding="utf-8")))
        for name in ["site_correlations", "hour_2024", "month_2024"]:
            TABLES[name] = pd.read_csv(OUT/f"{name}.csv", index_col=0, encoding="utf-8-sig")
        s = pd.read_csv(OUT/"site_2024.csv", index_col=0, encoding="utf-8-sig")
        table("capacity_bins_2024", capacity_bin_summary(s))
        report()
        log("Updated relationship interpretation from saved aggregate results.")
        return
    if "--verify-only" in sys.argv:
        result = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT/"gallery.json").read_text(encoding="utf-8"))
        axis = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(SOURCE/"train/smp.csv", usecols=["ts"], encoding="utf-8-sig").ts))
        assert axis.equals(pd.date_range(axis.min(), axis.max(), freq="h"))
        assert ((axis-pd.Timedelta(hours=1)).year == 2024).sum() == 8784
        source_ts = pd.read_parquet(SOURCE/"train/generation.parquet", columns=["ts"]).ts
        assert source_ts.isin(axis).all()
        assert result["source_rows"] == result["joined_rows"] == len(source_ts)
        assert result["fixed_raw_rows"] == result["fixed_sites"]*8784
        assert result["complete_2024_site_days"]*24 == result["fixed_valid_rows"]
        hour = pd.read_csv(OUT/"hour_2024.csv", index_col=0, encoding="utf-8-sig")
        month = pd.read_csv(OUT/"month_2024.csv", index_col=0, encoding="utf-8-sig")
        assert len(hour) == 24 and len(month) == 12
        assert hour.n.sum() == month.n.sum() == result["fixed_valid_rows"]
        assert month.sites.eq(result["fixed_sites"]).all()
        assert np.allclose(hour.mean_cf, hour.equal_site_cf)
        assert np.allclose(month.mean_cf, month.equal_site_cf)
        assert result["source_sha256"] == {r: digest(SOURCE/r) for r in INPUTS}
        for g in manifest:
            with Image.open(OUT/g["file"]) as im:
                im.verify()
        page = (OUT/"index.html").read_text(encoding="utf-8")
        links = re.findall(r'(?:href|src)="([^"]+)"', page)
        assert len(manifest) == 14 and len(links) == 28
        assert all((OUT/p).is_file() for p in links)
        assert "\u00b7" not in REPORT.read_text(encoding="utf-8")
        verification = {"source_rows": len(source_ts), "source_unchanged": True,
                        "source_hourly_axis_checked": True, "fixed_site_count": result["fixed_sites"],
                        "fixed_2024_hours_per_site": 8784, "hour_and_month_accounting": True,
                        "complete_site_day_accounting": True, "figures_checked": 14, "gallery_links_checked": 28}
        (OUT/"verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        print(json.dumps(verification, indent=2), flush=True)
        return
    INFO["started_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes")
    before = {r: digest(SOURCE/r) for r in INPUTS}
    baseline = json.loads((ROOT/"data/eda/2026-10-19_basic_shape/summary.json").read_text(encoding="utf-8"))
    assert all(before[r] == baseline["source_sha256"][r] for r in INPUTS)
    log("Runtime started; reading generation and metadata.")
    meta = pd.read_csv(SOURCE/"sites.csv", encoding="utf-8-sig")
    gen = pd.read_parquet(SOURCE/"train/generation.parquet")
    assert not meta.site_id.duplicated().any() and (meta.capacity_kw > 0).all()
    assert not gen.duplicated(["site_id", "ts"]).any()
    d = gen.merge(meta[["site_id", "capacity_kw"]], on="site_id", how="left", validate="many_to_one")
    assert len(d) == len(gen) and d.capacity_kw.notna().all()
    d["ts"] = pd.to_datetime(d.ts)
    d["cf"] = d.energy/d.capacity_kw
    start = d.ts-pd.Timedelta(hours=1)
    d["hour"] = start.dt.hour.astype("int8")
    d["year"] = start.dt.year.astype("int16")
    d["month"] = start.dt.month.astype("int8")
    d["day"] = start.dt.floor("D")
    d["year_month"] = start.dt.tz_localize(None).dt.to_period("M").astype(str)
    d["is_zero"] = d.energy.eq(0)
    year_counts = d.loc[d.year == 2024].groupby("site_id").size()
    fixed_ids = year_counts[year_counts == 366*24].index
    assert len(fixed_ids) > 1
    all_times = d.groupby("ts").site_id.nunique()
    common_times = all_times[all_times == len(meta)].index
    raw_sets = {"전체 관측 기간": d, "2024 고정 집단": d.loc[(d.year == 2024)&d.site_id.isin(fixed_ids)],
                "253개 공통 시각": d.loc[d.ts.isin(common_times)]}
    sets = {name: z.loc[z.energy >= 0] for name, z in raw_sets.items()}
    table("negative_sensitivity", pd.DataFrame([{"window": name, "raw_rows": len(z), "negative": int((z.energy<0).sum()),
                "raw_mean_cf": float(z.cf.mean()), "nonnegative_mean_cf": float(sets[name].cf.mean())} for name, z in raw_sets.items()]))
    sites = {name: site_summary(z) for name, z in sets.items()}
    for name, key in [("전체 관측 기간", "site_all"), ("2024 고정 집단", "site_2024"), ("253개 공통 시각", "site_common")]:
        table(key, sites[name])
    table("site_correlations", pd.DataFrame([r for name, s in sites.items() for r in correlation(s, name)]))
    table("capacity_bins_2024", capacity_bin_summary(sites["2024 고정 집단"]))
    hours = {name: hourly(z) for name, z in sets.items()}
    for name, key in [("전체 관측 기간", "hour_all"), ("2024 고정 집단", "hour_2024"), ("253개 공통 시각", "hour_common")]:
        table(key, hours[name])
    table("month_2024", monthly(sets["2024 고정 집단"]))
    table("year_month_all", monthly(sets["전체 관측 기간"], "year_month"))
    fixed = sets["2024 고정 집단"]
    site_days = fixed.groupby(["site_id", "day"], observed=True).agg(n=("energy", "size"), total=("energy", "sum"), cap=("capacity_kw", "first"))
    complete = site_days.loc[site_days.n == 24].copy()
    complete["yield"] = complete.total/complete.cap
    complete["month"] = complete.index.get_level_values("day").month
    daily2024 = complete.groupby("day")["yield"].agg(["mean", "median", lambda x: x.quantile(.25), lambda x: x.quantile(.75), "size"])
    daily2024.columns = ["mean", "median", "q25", "q75", "site_days"]
    daily_month = complete.groupby("month")["yield"].agg(["mean", "median", lambda x: x.quantile(.25), lambda x: x.quantile(.75), "size"])
    daily_month.columns = ["mean", "median", "q25", "q75", "site_days"]
    table("complete_site_days_2024", complete)
    table("daily_2024", daily2024)
    table("daily_yield_by_month_2024", daily_month)
    hm = fixed.groupby(["month", "hour"], observed=True).cf.mean().unstack("hour").reindex(index=range(1, 13), columns=range(24))
    hz = fixed.groupby(["month", "hour"], observed=True).is_zero.mean().unstack("hour").reindex(index=range(1, 13), columns=range(24))
    table("month_hour_cf_2024", hm)
    table("month_hour_zero_2024", hz)
    availability = d.groupby(["site_id", "year_month"], observed=True).size().unstack("year_month").reindex(meta.site_id).fillna(0)
    for col in availability:
        availability[col] /= pd.Period(col, freq="M").days_in_month*24
    assert (availability.to_numpy() <= 1).all()
    table("monthly_record_availability", availability)
    INFO.update(source_rows=len(gen), joined_rows=len(d), nonnegative_rows=len(sets["전체 관측 기간"]),
                fixed_sites=len(fixed_ids), fixed_raw_rows=len(raw_sets["2024 고정 집단"]), fixed_valid_rows=len(fixed),
                fixed_expected_hours=8784, common_times=len(common_times), common_start=str(common_times.min()), common_end=str(common_times.max()),
                cf_gt1=int((d.cf>1).sum()), cf_gt1_2=int((d.cf>1.2).sum()),
                complete_2024_site_days=len(complete), incomplete_2024_site_days=int((site_days.n<24).sum()))
    log(f"Aggregates completed: fixed 2024 cohort {len(fixed_ids)} sites; plotting.")
    plot_capacity(sites, sets["전체 관측 기간"])
    plot_time(sets, hours, TABLES["month_2024"], daily2024, daily_month, TABLES["year_month_all"], availability, hm, hz)
    assert before == {r: digest(SOURCE/r) for r in INPUTS}
    assert len(GALLERY) == 14
    INFO.update(source_unchanged=True, source_sha256=before, figures=len(GALLERY),
                finished_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes"))
    (OUT/"summary.json").write_text(json.dumps(INFO, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"gallery.json").write_text(json.dumps(GALLERY, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gallery()
    report()
    assert "\u00b7" not in REPORT.read_text(encoding="utf-8")
    assert INFO["fixed_raw_rows"] == INFO["fixed_sites"]*8784
    assert TABLES["hour_2024"].n.sum() == len(fixed) == TABLES["month_2024"].n.sum()
    log(f"Completed relationship EDA: {len(GALLERY)} figures. Report: {REPORT}")


if __name__ == "__main__":
    main()
