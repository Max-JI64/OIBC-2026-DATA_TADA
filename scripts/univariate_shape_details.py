"""Full-column shape review: atoms, tails, discretization, and peak candidates.

No cross-variable analysis, distribution fitting, model training, or predictions.
Every numeric file-column gets its own diagnostic figure and aggregate summary.
"""
from __future__ import annotations

import gc
import html
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
from scipy.signal import find_peaks

from basic_shape_eda import FILES, SOURCE, ROOT, digest, fmt, md_table, log

OUT = ROOT / "data" / "eda" / "2026-10-19_shape_details"
FIG = OUT / "figures"
REPORT = ROOT / "reports" / "2026-10-19_03_분포_형태_추가점검.md"
RESULTS = []
MANIFEST = []
CAT_RESULTS = []
AUDIT_RULES = {
    "wa_cloud": ("운량 0~100% 밖", lambda a: (a < 0) | (a > 100), "범위 밖 값"),
    "wb_cloud": ("운량 0~100% 밖", lambda a: (a < 0) | (a > 100), "범위 밖 값"),
    "wa_humidity": ("상대습도 0~100% 밖", lambda a: (a < 0) | (a > 100), "범위 밖 값"),
    "wb_rel_hum": ("상대습도 0~100% 밖", lambda a: (a < 0) | (a > 100), "범위 밖 값"),
    "wa_wind_speed": ("풍속 >50 m/s", lambda a: a > 50, "안내문 관찰 기준"),
    "wb_wind_speed": ("풍속 >200 km/h", lambda a: a > 200, "안내문 관찰 기준"),
    "wb_uv_idx": ("UV >15", lambda a: a > 15, "안내문 관찰 기준"),
    "wb_vis": ("시정 >50 km", lambda a: a > 50, "안내문 관찰 기준"),
    "wb_pressure": ("기압 <900 hPa", lambda a: a < 900, "안내문 관찰 기준"),
    "energy": ("발전량 <0 kWh", lambda a: a < 0, "음수 기록"),
}


def code_name(rel, col):
    return rel.replace("/", "_").replace(".parquet", "").replace(".csv", "") + "__" + col


def peaks(a, bins):
    """A visual aid, not a statistical test or proof of mixture distributions."""
    if len(a) < 100 or a.min() == a.max():
        return []
    hist, edges = np.histogram(a, bins=bins)
    smooth = np.convolve(hist.astype(float), [.25, .5, .25], mode="same")
    indices, _ = find_peaks(smooth, prominence=max(1., smooth.max() * .12),
                            distance=max(2, bins // 12))
    centers = (edges[:-1] + edges[1:]) / 2
    return [float(centers[i]) for i in indices]


def summarize(a, rel, col):
    counts = pd.Series(a).value_counts()
    n = len(a)
    q = np.quantile(a, [0, .01, .25, .5, .75, .95, .99, .999, 1])
    lo, q01, q25, median, q75, q95, q99, q999, hi = map(float, q)
    iqr = q75 - q25
    m = float(a.mean())
    sd = float(a.std())
    if sd > 0:
        z = (a - m) / sd
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3)
        del z
    else:
        skew = kurt = 0.0
    nonzero = a[a != 0]
    central = a[(a >= q01) & (a <= q99)]
    peak32, peak64 = peaks(central, 32), peaks(central, 64)
    peak32_nz, peak64_nz = [], []
    if len(nonzero):
        nz01, nz99 = np.quantile(nonzero, [.01, .99])
        nz = nonzero[(nonzero >= nz01) & (nonzero <= nz99)]
        peak32_nz, peak64_nz = peaks(nz, 32), peaks(nz, 64)
    outside = int(((a < q25 - 1.5 * iqr) | (a > q75 + 1.5 * iqr)).sum())
    zeros = int((a == 0).sum())
    top = [{"value": float(v), "count": int(c), "share": float(c / n)} for v, c in counts.head(8).items()]
    unique = int(len(counts))
    steps = {}
    if 1 < unique <= 5000:
        delta = np.diff(np.sort(counts.index.to_numpy(dtype=float)))
        rounded = np.round(delta, 9)
        gap_counts = pd.Series(rounded).value_counts()
        steps = {"most_common_unique_gap": float(gap_counts.index[0]),
                 "gap_share": float(gap_counts.iloc[0] / len(delta))}
    flags = []
    if zeros / n >= .20:
        flags.append("0 집중")
    if top[0]["share"] >= .10:
        flags.append("특정 값 집중")
    if unique <= 30:
        flags.append("소수의 이산 값")
    elif unique <= 300 and n > 10000:
        flags.append("유한한 값 단계")
    if skew > 2:
        flags.append("긴 오른쪽 꼬리")
    elif skew < -2:
        flags.append("긴 왼쪽 꼬리")
    if iqr == 0:
        flags.append("IQR=0")
    if outside / n >= .10:
        flags.append("IQR 밖 10% 이상")
    if len(peak32) >= 2 and len(peak64) >= 2:
        flags.append("복수 봉우리 후보")
    if len(peak32_nz) >= 2 and len(peak64_nz) >= 2:
        flags.append("0 제외 후 복수 봉우리 후보")
    rule = None
    if col in AUDIT_RULES:
        title, check, kind = AUDIT_RULES[col]
        number = int(check(a).sum())
        rule = {"condition": title, "count": number, "share": number / n, "kind": kind}
        if number:
            flags.append(kind)
    r = {"file": rel, "column": col, "n": n, "unique": unique,
         "min": lo, "q01": q01, "q25": q25, "median": median,
         "q75": q75, "q95": q95, "q99": q99, "q999": q999, "max": hi,
         "zero_count": zeros, "zero_share": zeros / n,
         "negative_count": int((a < 0).sum()), "skewness": skew,
         "excess_kurtosis": kurt, "iqr_outside": outside,
         "iqr_outside_share": outside / n, "iqr_zero": bool(iqr == 0),
         "top_values": top, "steps": steps,
         "peaks32": peak32, "peaks64": peak64,
         "nonzero_peaks32": peak32_nz, "nonzero_peaks64": peak64_nz,
         "rule": rule, "flags": flags}
    return r


def draw(a, r):
    fig, axs = plt.subplots(2, 2, figsize=(12.2, 7.8))
    a0, a1, a2, a3 = axs.flat
    a0.hist(a, bins=64, color="#276f86", edgecolor="white", lw=.2)
    a0.set_title("전체 범위 / 선형 세로축")
    a0.set_ylabel("행 수")
    main = a[(a >= r["q01"]) & (a <= r["q99"])]
    for bins, color in [(32, "#276f86"), (64, "#d47e47")]:
        h, edges = np.histogram(main, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        a1.plot(centers, h / max(1, h.sum()) / np.diff(edges), color=color, lw=1.2, label=f"{bins}개 구간")
    a1.set_title("1~99백분위 확대 / 구간 수 민감도")
    a1.set_ylabel("구간 폭당 비율")
    a1.legend(fontsize=8)
    nonzero = a[a != 0]
    if len(nonzero):
        low, high = np.quantile(nonzero, [.01, .99])
        inside = nonzero[(nonzero >= low) & (nonzero <= high)]
        a2.hist(inside, bins=64, color="#6b8d66", edgecolor="white", lw=.2)
        a2.set_title(f"0 제외 후 1~99백분위 / 전체의 {len(nonzero)/len(a):.2%}")
    else:
        a2.text(.5, .5, "모든 값이 0", ha="center", transform=a2.transAxes)
        a2.set_title("0 제외 후 값 없음")
    a2.set_ylabel("행 수")
    top = r["top_values"]
    labels = [fmt(v["value"]) for v in top]
    shares = [v["share"] * 100 for v in top]
    a3.barh(np.arange(len(top)), shares, color="#87749e")
    a3.set_yticks(np.arange(len(top)), labels)
    a3.invert_yaxis()
    a3.set_xlabel("전체 유한값 중 비율 (%)")
    a3.set_title("가장 자주 등장하는 값 8개")
    a3.margins(x=.20)
    for i, v in enumerate(shares):
        a3.text(v, i, f" {v:.3g}%", va="center", fontsize=8)
    for ax in (a0, a1, a2):
        ax.set_xlabel(r["column"])
        ax.grid(axis="y", alpha=.15)
    title = f"{r['file']} / {r['column']}"
    subtitle = f"0: {r['zero_share']:.2%} | 고유값: {r['unique']:,} | IQR 밖: {r['iqr_outside_share']:.2%} | 왜도: {r['skewness']:.3g}"
    fig.suptitle(title + "\n" + subtitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .92])
    name = code_name(r["file"], r["column"]) + ".png"
    fig.savefig(FIG / name, dpi=125, bbox_inches="tight")
    plt.close(fig)
    r["figure"] = "figures/" + name
    MANIFEST.append({"file": r["file"], "column": r["column"], "figure": r["figure"]})


def write_gallery():
    sections = []
    for r in RESULTS:
        t = html.escape(r["file"] + " / " + r["column"])
        flags = html.escape(", ".join(r["flags"]) or "지표상 두드러진 플래그 없음")
        sections.append(f'<section data-file="{html.escape(r["file"])}"><h2>{t}</h2><p>{flags}</p>'
                        f'<a href="{r["figure"]}"><img loading="lazy" src="{r["figure"]}" alt="{t}"></a></section>')
    for r in CAT_RESULTS:
        t = html.escape(r["file"] + " / " + r["column"] + " / 범주 빈도")
        sections.append(f'<section data-file="{html.escape(r["file"])}"><h2>{t}</h2>'
                        '<p>명목형 코드와 round의 빈도만 확인. 다른 변수와의 관계는 분석하지 않음.</p>'
                        f'<a href="{r["figure"]}"><img loading="lazy" src="{r["figure"]}" alt="{t}"></a></section>')
    options = "".join(f'<option>{html.escape(rel)}</option>' for rel in FILES)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>개별 변수 분포 추가 점검</title><style>body{font:16px/1.6 'Malgun Gothic',sans-serif;margin:0;background:#f4f5f5;color:#26333c}
header,main{max-width:1150px;margin:auto;padding:22px}header{position:sticky;top:0;z-index:2;background:#f4f5f5ee;border-bottom:1px solid #ddd}
h1{font-size:25px;margin:0}h2{font-size:18px}p{color:#53616d}section{padding:20px;background:white;margin:22px 0;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}select{font:inherit;padding:6px}</style>
<header><h1>개별 변수 분포 추가 점검</h1><p>수치 열의 전체 범위, 중심부, 0 제외 분포, 특정 값 집중과 범주형 열의 빈도. 복수 봉우리는 혼합분포의 확정 근거가 아닙니다.</p>
<select id="f"><option value="all">모든 수치 열</option>''' + options + '''</select></header><main>''' + "".join(sections) + '''</main>
<script>document.getElementById('f').onchange=function(){document.querySelectorAll('section').forEach(s=>s.hidden=this.value!=='all'&&s.dataset.file!==this.value)}</script></html>'''
    (OUT / "index.html").write_text(page, encoding="utf-8")


def contact_sheets():
    qa = OUT / "qa"
    qa.mkdir(exist_ok=True)
    for start in range(0, len(MANIFEST), 4):
        group = MANIFEST[start:start + 4]
        images = []
        for g in group:
            with Image.open(OUT / g["figure"]) as im:
                im.load()
                images.append(im.convert("RGB").resize((im.width // 2, im.height // 2), Image.Resampling.LANCZOS))
        w, h = max(im.width for im in images), max(im.height for im in images) + 25
        sheet = Image.new("RGB", (w * 2, h * 2), "#ededed")
        pen = ImageDraw.Draw(sheet)
        for i, (im, g) in enumerate(zip(images, group)):
            x, y = (i % 2) * w, (i // 2) * h
            sheet.paste(im, (x, y + 25))
            pen.text((x + 5, y + 4), f"{start+i+1:02d}: {g['file']} {g['column']}", fill="black")
        sheet.save(qa / f"contact_{start // 4 + 1:02d}.png")


def build_report():
    def get(rel, col):
        return next(r for r in RESULTS if r["file"] == rel and r["column"] == col)
    e = get("train/generation.parquet", "energy")
    cap = get("sites.csv", "capacity_kw")
    rain = get("train/forecast_wa.parquet", "wa_rain")
    snow = get("train/forecast_wa.parquet", "wa_snow")
    bvis = get("train/forecast_wb.parquet", "wb_vis")
    bceil = get("train/forecast_wb.parquet", "wb_ceiling")
    clouds = [r for r in RESULTS if r["column"] in ["wa_cloud", "wb_cloud"]]
    cloud_rows = []
    for r in clouds:
        at100 = next(v["share"] for v in r["top_values"] if v["value"] == 100)
        cloud_rows.append([r["file"], r["column"], f"{r['zero_share']:.2%}", f"{at100:.2%}"])
    tiny_values = [[r["file"], r["column"], f"{v['value']:.12g}", fmt(v["count"]), f"{v['share']:.2%}"]
                   for r in RESULTS for v in r["top_values"] if 0 < abs(v["value"]) < 1e-10]
    p = ["# 개별 변수 분포의 형태 추가 점검", "",
         f"작성: {datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')}. 상태: 전체 수치 열의 추가 집계와 그림 생성 완료.", "",
         "## 질문과 목적", "",
         "이전 EDA는 모든 행의 기초통계를 계산했지만 그래프는 대표 일부만 시각적으로 확인했다. 0과 특정 값에 몰림, 여러 봉우리, 긴 꼬리, 이산 단계, 극단값을 충분히 설명하지 못했다. 이번 작업은 그 부분을 개별 변수 단위로 보완한다.", "",
         "## 사용 범위와 수행 방법", "",
         "CSV 3개와 Parquet 7개 전체를 다시 읽고, round를 제외한 파일별 수치 열 94개를 모두 점검했다. 파일별로 같은 이름의 열도 별도 집계했으며 표본 추출, 변수 간 관계, 하위 집단 비교, 혼합모형 적합, 머신러닝은 하지 않았다. 원본 행은 보고서에 복사하지 않는다.", "",
         "- 각 열의 최빈값과 상위 8개 값의 비율, 0 비율, 고유값 수, 왜도와 초과첨도를 계산했다. 고유값이 5,000개 이하인 열은 정렬한 고유값의 간격도 집계했다.",
         "- 전체 범위의 선형 세로축 히스토그램, 1~99백분위 확대, 0 제외 분포, 자주 나오는 값의 빈도를 열마다 한 그림으로 만들었다. 확대 그림 밖 값은 원본이나 전체 집계에서 삭제하지 않았다.",
         "- 중심 구간에서 32개와 64개 구간 히스토그램을 비교했다. 3점 평활 뒤 최대 높이의 12% 이상 두드러지는 내부 봉우리를 후보로 셌다. 양쪽 끝의 경계 봉우리는 이 방법의 탐지 대상이 아니며, 구간 폭과 이산화 때문에 후보 수가 달라질 수 있다.",
         "- IQR 밖 비율과 안내문의 관찰 기준 초과 비율을 분리했다. IQR 밖은 데이터 오류의 수가 아니다. 구간 제한이 명확한 운량과 상대습도도 원인을 확인하지 않고 자동 제거하지 않았다.", "",
         "## 확인한 특징", "",
         f"### 발전량: 0에 큰 덩어리가 있고 오른쪽 꼬리가 길다", "",
         f"0이 {e['zero_share']:.2%}다. 중앙값 {fmt(e['median'])}, 99백분위 {fmt(e['q99'])}, 최대 {fmt(e['max'])} kWh다. 왜도 {e['skewness']:.3f}, IQR 밖 비율 {e['iqr_outside_share']:.2%}다. 0과 양수 기록이 함께 있는 모양을 확인했으며, 이를 확률모형 의미의 영과잉 분포로 확정하지 않았다. 음수 {e['negative_count']:,}행은 별도로 기록했다.", "",
         "### 설비용량: 100 kW 부근 집중과 큰 값의 긴 꼬리", "",
         f"25/50/75백분위는 {fmt(cap['q25'])}/{fmt(cap['median'])}/{fmt(cap['q75'])} kW이고, 최대는 {fmt(cap['max'])} kW다. IQR 밖이 {cap['iqr_outside_share']:.2%}다. 설비용량처럼 규모 자체가 다른 값에서는 이 비율을 고장이나 입력 오류 비율로 읽을 수 없다.", "",
         "### 강수 계열: 상자그림만으로는 모양을 설명하기 어렵다", "",
         f"학습 A 예보 강우량의 0 비율은 {rain['zero_share']:.2%}, 강설량은 {snow['zero_share']:.2%}다. 두 열의 IQR은 0이다. 이때 양수 기록이 IQR 밖으로 분류되므로 '이상치가 많다'고 해석하면 안 된다. 0을 제외한 분포를 별도로 그려 남은 값의 모양을 확인할 수 있게 했다.", "",
         "### 일부 기상 열: 특정 숫자에 큰 비중이 쌓인다", "",
         f"학습 B 예보 wb_vis의 최빈값은 {fmt(bvis['top_values'][0]['value'])} km이고 비중은 {bvis['top_values'][0]['share']:.2%}다. wb_ceiling은 {fmt(bceil['top_values'][0]['value'])} m에 {bceil['top_values'][0]['share']:.2%}가 쌓인다. 이런 날카로운 봉우리는 연속분포의 여러 집단을 뜻할 수도 있지만, 기록 단계, 상한값 또는 코드값의 영향일 수도 있다. 이번에는 숫자에 몰리는 사실만 확인했으며 원인은 미확인이다.", "",
         "### 운량: 0%와 100%의 경계 봉우리", "",
         "중간값보다 양끝에 크게 쌓이는 모양을 확인했다. 내부 봉우리 탐지에서 0/0으로 표시되더라도 단봉분포라는 뜻이 아니다. 이번 봉우리 탐지기는 양끝의 봉우리를 세지 않기 때문에 실제 그림과 끝값 빈도를 함께 확인해야 한다.", "",
         md_table(["파일", "열", "0%에 있는 비율", "100%에 있는 비율"], cloud_rows), "",
         "### 전력가격, 단계형 값, 풍향에서 주의할 모양", "",
         "학습 SMP는 넓은 주된 덩어리 외에 높은 가격 쪽에도 작은 덩어리와 봉우리가 보이며, 평가 SMP에도 여러 봉우리 후보가 있다. 원인을 연도나 계절로 나누는 분석은 하지 않았다. 위도, 경도와 관측 지점 거리의 히스토그램도 들쭉날쭉하지만 발전소 수가 253개이고 거리에는 기록 간격이 있으므로 구간 수에 따른 모양 변화를 함께 봐야 한다.", "",
         "일부 B 예보의 기온, 풍속과 강수량에는 좁고 반복되는 빗살 같은 봉우리가 있다. A 예보의 습도와 기압에도 단계가 보인다. 이런 모양을 연속분포 여러 개의 혼합으로 바로 해석하지 않는다. 정확한 값의 빈도와 고유값의 간격은 로컬 상세 집계에 보관했다.", "",
         "풍향은 0도와 360도가 같은 방향인 원형 변수다. 선형 축에서 양끝이 떨어져 보인다고 서로 완전히 다른 두 집단으로 해석할 수 없다. k는 유효한 값 1과 3만 가지며, 3인 20.83%가 IQR 밖으로 분류되는 것은 오류 판정이 아니다.", "",
         "### 정확한 0과 거의 0인 값", "",
         "최빈 상위 8개 값 안에서 다음과 같은 아주 작은 비영 값도 확인했다. 표시 자릿수를 줄이면 0처럼 보이지만 정확한 0 비율에는 포함되지 않는다. 부동소수점 계산의 흔적일 수 있으나 원인을 확정하거나 반올림하지 않았다. 아래는 상위 8개 값에서 발견한 사례이며 모든 근사 0을 전수 집계한 표는 아니다.", "",
         md_table(["파일", "열", "값", "빈도", "비율"], tiny_values), "",
         "### 봉우리가 여러 개면 여러 분포가 섞였다고 확정할 수 있는가", "",
         "확정할 수 없다. 여러 봉우리는 혼합 가능성을 보여 주는 모양이다. 이산 값의 간격, 반올림, 특정 값의 집중만으로도 비슷한 모양이 나온다. 현재 데이터는 여러 발전소와 지점, 시각을 함께 담고 있지만 어떤 집단 때문에 어떤 봉우리가 생겼는지는 변수 관계 또는 집단별 분석이 필요하므로 이번 범위에서 판단하지 않았다. 봉우리 후보는 아래 표와 그림에 남겼다.", "",
         "## 안내 범위 또는 관찰 기준을 벗어나는 값", "",
         "아래 비율의 분모는 해당 열의 유한값 수다. 관찰 기준은 배포 안내의 확인 조건이며 절대적인 오류 판정이 아니다. 범위 밖 값과 통계적 꼬리를 구분한다.", "",
         md_table(["파일", "열", "조건", "건수", "비율", "구분"],
                  [[r["file"], r["column"], r["rule"]["condition"], fmt(r["rule"]["count"]), f"{r['rule']['share']:.5%}", r["rule"]["kind"]] for r in RESULTS if r["rule"] and r["rule"]["count"]]), "",
         "## 전체 수치 열 점검표", "",
         "표의 '봉우리'는 전체 중심 구간 32/64개 구간에서 찾은 내부 봉우리 수다. 여러 개여도 혼합분포 확정은 아니다. '최빈 비율'은 가장 자주 나타난 정확한 한 값의 비율이다. 왜도는 전체 유한값의 표준화 3차 중심모멘트이며 큰 절댓값은 극단값의 영향을 받을 수 있다.", ""]
    for rel in FILES:
        selected = [r for r in RESULTS if r["file"] == rel]
        p += [f"### {rel}", "", md_table(["열", "0", "고유값", "최빈 비율", "왜도", "IQR 밖", "봉우리 32/64", "형태 플래그"],
                [[r["column"], f"{r['zero_share']:.2%}", fmt(r["unique"]), f"{r['top_values'][0]['share']:.2%}", f"{r['skewness']:.3f}", f"{r['iqr_outside_share']:.2%}", f"{len(r['peaks32'])}/{len(r['peaks64'])}", ", ".join(r["flags"]) or "지표상 두드러진 플래그 없음"] for r in selected]), ""]
    if CAT_RESULTS:
        p += ["## 범주형 열과 시각 열의 확인 범위", "",
              "수치 열 94개 외에 ID, 관측 지점, 참조 그룹과 round의 15개 파일별 범주형 열을 모두 빈도로 확인했다. 이 빈도는 앞선 전체 행 EDA의 집계를 재사용했고, 이번 수치 EDA의 입력 해시가 당시와 같은지 확인했다. 시각 열 9개는 앞선 보고서의 범위, 정시 여부와 가용성 검사에 해당하며 이번에 시간에 따른 다른 변수의 변화를 분석하지 않았다. 따라서 파일별 전체 118개 열은 수치 94개, 범주 15개, 시각 9개로 구분된다.", "",
              md_table(["파일", "범주 열", "범주 수", "최소 빈도", "중앙 빈도", "최대 빈도", "가장 큰 범주 비율"],
                       [[r["file"], r["column"], fmt(r["levels"]), fmt(r["min_frequency"]), fmt(r["median_frequency"]), fmt(r["max_frequency"]), f"{r['largest_share']:.2%}"] for r in CAT_RESULTS]), "",
              "관측 지점과 발전소 ID는 숫자로 바꾸어 연속분포로 해석하지 않았다. 빈도 차이는 기록의 불균형만 보여 주며 그 원인은 판단하지 않았다.", ""]
    p += ["## 그림과 관련 코드", "",
          "- [전체 수치 열과 범주형 열의 분포 갤러리](../data/eda/2026-10-19_shape_details/index.html)",
          "- [상세 수치와 봉우리 후보 위치](../data/eda/2026-10-19_shape_details/details.json)",
          "- [분포 추가 점검 코드](../scripts/univariate_shape_details.py)",
          "- [앞선 데이터 형태 EDA](2026-10-19_02_데이터_형태_EDA.md)", "",
          "그래프와 상세 집계는 로컬 data/에만 보관한다. Git에는 코드와 이 요약 보고서, 일별 로그만 포함한다.", "",
          "## 한계 및 현재 상태", "",
          "기본 EDA와 이번 점검은 전체 행을 읽은 단변량 기술통계다. 원자료에 있는 모든 문제를 발견했다는 보장은 하지 않는다. 여러 봉우리의 원인과 각 봉우리를 만드는 집단, 극단값의 발생 원인, 제거 기준은 확정하지 않았다. 학습과 예측, 변수 간 관계 분석은 하지 않았다.", "",
          "생성한 모든 그림은 파일 단위 읽기 검사를 했고, 수치 열별 집계 건수와 앞선 EDA의 일치, 원본 해시 유지도 검사했다. 개별 그림의 시각 검토 내역은 로컬 visual_review.md에 별도로 기록한다.", ""]
    if (OUT / "visual_review.md").is_file():
        p += ["시각 검토 완료: 수치 열 94개와 범주형 열 15개의 그림 109개를 전부 확인했다. 원본의 50% 크기로 한 장당 최대 4개를 모은 검토본 28장을 사용했다. 개별 픽셀을 원본 크기로 전부 확인했다는 뜻은 아니다.", ""]
    REPORT.write_text("\n".join(p), encoding="utf-8")


def categorical_shapes(base):
    previous = ROOT / "data/eda/2026-10-19_basic_shape"
    for rel, info in base["files"].items():
        for col, meta in info["categories"].items():
            name = rel.replace("/", "_").replace(".csv", "").replace(".parquet", "")
            d = pd.read_csv(previous / f"{name}_{col}_counts.csv", encoding="utf-8-sig")
            counts = d["rows"].to_numpy()
            assert counts.sum() == info["rows"] and len(counts) == meta["levels"]
            r = {"file": rel, "column": col, "levels": len(counts),
                 "min_frequency": int(counts.min()), "median_frequency": float(np.median(counts)),
                 "max_frequency": int(counts.max()), "largest_share": float(counts.max()/counts.sum())}
            fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
            if len(counts) <= 5:
                labels = d[col].astype(str).tolist()
                axes[0].bar(labels, counts, color="#276f86")
                axes[0].set_xlabel(col)
            else:
                axes[0].bar(np.arange(1, len(counts)+1), np.sort(counts)[::-1], color="#276f86")
                axes[0].set_xlabel("범주 빈도 순위 (코드의 크기 순서가 아님)")
            axes[0].set_ylabel("행 수")
            axes[0].set_title("범주별 기록 수")
            axes[1].hist(counts, bins=min(30, max(2, len(np.unique(counts)))), color="#d47e47", edgecolor="white")
            axes[1].set_title("범주별 기록 수의 분포")
            axes[1].set_xlabel("한 범주의 기록 수")
            axes[1].set_ylabel("범주 수")
            fig.suptitle(f"{rel} / {col} / 범주 {len(counts):,}개", fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, .92])
            name = "category__" + code_name(rel, col) + ".png"
            fig.savefig(FIG / name, dpi=125, bbox_inches="tight")
            plt.close(fig)
            r["figure"] = "figures/" + name
            CAT_RESULTS.append(r)
            MANIFEST.append({"file": rel, "column": col, "figure": r["figure"]})
    assert len(CAT_RESULTS) == 15
    (OUT / "categorical_details.json").write_text(json.dumps(CAT_RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    assert sys.version_info[:2] == (3, 13)
    FIG.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "data/eda/2026-10-19_basic_shape/summary.json").read_text(encoding="utf-8"))
    if "--verify-only" in sys.argv:
        details = json.loads((OUT / "details.json").read_text(encoding="utf-8"))
        cats = json.loads((OUT / "categorical_details.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        assert len(details) == 94 and len(cats) == 15 and len(manifest) == 109
        assert len({(g["file"], g["column"]) for g in manifest}) == 109
        for r in details:
            old = base["files"][r["file"]]["numeric"][r["column"]]
            assert r["n"] == old["count"] and r["zero_count"] == old["zeros"]
            assert r["iqr_outside"] == old["iqr_outside"]
            assert sum(v["count"] for v in r["top_values"]) <= r["n"]
        for g in manifest:
            with Image.open(OUT / g["figure"]) as im:
                im.verify()
        page = (OUT / "index.html").read_text(encoding="utf-8")
        links = re.findall(r'(?:href|src)="([^"]+)"', page)
        assert len(links) == 218
        assert all((OUT / name).is_file() for name in links)
        assert len(list((OUT / "qa").glob("contact_*.png"))) == 28
        assert (OUT / "visual_review.md").is_file()
        assert {rel: digest(SOURCE / rel) for rel in FILES} == base["source_sha256"]
        assert not any("\u00b7" in p.read_text(encoding="utf-8") for p in [Path(__file__), REPORT, OUT / "index.html"])
        check = {"numeric_columns": 94, "categorical_columns": 15, "timestamp_columns_previously_checked": 9,
                 "total_file_columns": 118, "figures_read_checked": 109,
                 "html_links_checked": 218, "contact_sheets": 28,
                 "all_counts_match_previous": True, "source_unchanged": True,
                 "visual_review_record_exists": True}
        (OUT / "verification.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
        print(json.dumps(check, indent=2), flush=True)
        return
    if "--report-only" in sys.argv:
        RESULTS.extend(json.loads((OUT / "details.json").read_text(encoding="utf-8")))
        CAT_RESULTS.extend(json.loads((OUT / "categorical_details.json").read_text(encoding="utf-8")))
        build_report()
        log("Report updated from saved aggregate results only.")
        return
    before = {rel: digest(SOURCE / rel) for rel in FILES}
    if "--complete-categories" in sys.argv:
        assert before == base["source_sha256"]
        RESULTS.extend(json.loads((OUT / "details.json").read_text(encoding="utf-8")))
        MANIFEST.extend(json.loads((OUT / "manifest.json").read_text(encoding="utf-8")))
        MANIFEST[:] = MANIFEST[:94]
        categorical_shapes(base)
        (OUT / "manifest.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8")
        write_gallery()
        contact_sheets()
        build_report()
        log(f"Completed categorical review: {len(CAT_RESULTS)} columns; total {len(MANIFEST)} figures.")
        return
    for rel in FILES:
        d = pd.read_csv(SOURCE / rel, encoding="utf-8-sig") if rel.endswith(".csv") else pd.read_parquet(SOURCE / rel)
        cols = [c for c in d.select_dtypes(include="number") if c != "round"]
        for col in cols:
            a = d[col].to_numpy(dtype=float, na_value=np.nan)
            a = a[np.isfinite(a)]
            r = summarize(a, rel, col)
            assert r["n"] == base["files"][rel]["numeric"][col]["count"]
            assert r["zero_count"] == base["files"][rel]["numeric"][col]["zeros"]
            assert r["iqr_outside"] == base["files"][rel]["numeric"][col]["iqr_outside"]
            draw(a, r)
            RESULTS.append(r)
        del d
        gc.collect()
        log(f"Shape details completed: {rel} / {len(cols)} numeric columns")
    assert len(RESULTS) == 94
    after = {rel: digest(SOURCE / rel) for rel in FILES}
    assert before == after == base["source_sha256"]
    (OUT / "details.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    categorical_shapes(base)
    (OUT / "manifest.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gallery()
    contact_sheets()
    build_report()
    assert not any("\u00b7" in p.read_text(encoding="utf-8") for p in [Path(__file__), REPORT, OUT / "index.html"])
    (OUT / "verification.json").write_text(json.dumps({"numeric_columns": len(RESULTS),
                "figures": len(MANIFEST), "source_unchanged": True,
                "all_counts_match_previous": True, "image_read_checks": len(MANIFEST)}, indent=2), encoding="utf-8")
    log(f"Completed: {len(RESULTS)} column-level figures and summaries. Report: {REPORT}")


if __name__ == "__main__":
    main()
