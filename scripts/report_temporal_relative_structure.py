"""Render saved DA-004 statistics; no statistical refitting or new selection."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from temporal_relative_structure import OUT, REPORT, dump


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def figures(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from eda_plot_utils import save_plot, write_gallery, check_gallery

    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    p = np.load(OUT / 'panel.npz', allow_pickle=True)  # Locally generated trusted station strings.
    r = np.load(OUT / 'results.npz')
    dates = pd.to_datetime(p['dates'])
    ids = p['site_ids'][p['eligible']]
    candidates = np.flatnonzero([v['robust_candidate'] for v in s['sites']])
    a = s['cases']['3']
    color = ['#21647f', '#b66537']
    gallery = []

    def date_axis(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(axis='y', alpha=.15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    configs = [(d, b) for d in ['3', '6'] for b in ['7', '14']]
    values = [s['cases'][d]['blocks'][b]['significant_sites'] for d, b in configs]
    axes[0].bar(range(4), values, color=['#437992', '#83a5b3', '#b16f47', '#d7a383'])
    axes[0].set(xticks=range(4), xticklabels=[f'{d}차 / {b}일' for d, b in configs], ylabel='발전소 수', title='각 설정의 전역 경험적 p <= 0.05')
    for i, v in enumerate(values):
        axes[0].text(i, v + .3, str(v), ha='center')
    counts = [70, s['counts']['all_four_significant'], s['counts']['date_and_direction_pass'], s['counts']['robust_candidates']]
    axes[1].bar(range(4), counts, color=['#aebac0', '#83a5b3', '#4a8198', '#b66537'])
    axes[1].set(xticks=range(4), xticklabels=['비교 가능', '네 설정 통과', '날짜/방향 통과', '90일/크기 통과'], ylabel='발전소 수', title='통계 반복과 지속 후보를 구분')
    for i, v in enumerate(counts):
        axes[1].text(i, v + .7, str(v), ha='center')
    save_plot(fig, OUT, gallery, '01_filters', '비교 대상과 사전 필터별 결과', '70개소/17지점. 후보 수는 독립 장애 수나 전체 발전소 비율이 아니다.')

    fig, ax = plt.subplots(figsize=(12, 7))
    order = np.argsort(a['statistics'])[-15:][::-1]
    smooth = pd.DataFrame(r['residual_3']).rolling(30, min_periods=30).mean().to_numpy() * 100
    limit = max(1., float(np.nanquantile(np.abs(smooth[:, order]), .98)))
    im = ax.imshow(smooth[:, order].T, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit,
                   extent=[mdates.date2num(dates[0]), mdates.date2num(dates[-1]), len(order)-.5, -.5])
    ax.set(yticks=range(len(order)), yticklabels=ids[order], xlabel='구간 시작일', title='주 분석 최대 통계 상위 15개소, 표시를 위한 선택')
    ax.xaxis_date()
    date_axis(ax)
    fig.colorbar(im, ax=ax, label='계절 제거 상대 차이의 30일 평균 (CF %p)')
    save_plot(fig, OUT, gallery, '02_relative_heatmap', '상대 차이의 시간 경로', '3차 계절 제거 잔차. 색 범위는 표시값 절댓값의 98백분위로 제한했다. 이 그림에서 재선택하지 않았다.')

    # The completed contract produced two candidates. Panels adapt if rerendered on another saved result.
    shown = candidates if len(candidates) else np.argsort(a['statistics'])[-2:][::-1]
    fig, axes = plt.subplots(len(shown), 1, figsize=(12, 4*len(shown)), squeeze=False)
    for ax, j in zip(axes.ravel(), shown):
        relative = p['relative'][:, j] * 100
        fit = relative - r['residual_3'][:, j]*100
        k = a['cut_indices'][j]
        ax.scatter(dates, relative, s=5, alpha=.15, color=color[0], rasterized=True)
        ax.plot(dates, pd.Series(relative).rolling(30, min_periods=30).mean(), color=color[0], linewidth=1.8, label='상대 차이 30일 평균')
        ax.plot(dates, fit, color=color[1], label='두 해 반복 계절 성분')
        ax.axvline(dates[k], color='#555555', linestyle='--', label='주 분석 탐색일')
        ax.set(title=f"{ids[j]}, {s['sites'][j]['station']}, 동료 {s['sites'][j]['peers']}개소", ylabel='동료 대비 CF 차이 (%p)')
        date_axis(ax)
        ax.legend(fontsize=8, ncol=3)
    save_plot(fig, OUT, gallery, '03_candidates_season', '지속 후보의 상대 수준과 반복 계절', '탐색일은 단일 분할 통계의 최대 날짜다. 실제 설비 변경일이나 발생일 구간을 추정한 것이 아니다.')

    fig, axes = plt.subplots(len(shown), 1, figsize=(12, 4*len(shown)), squeeze=False)
    for ax, j in zip(axes.ravel(), shown):
        for degree, c in [(3, color[0]), (6, color[1])]:
            z = r[f'residual_{degree}'][:, j] * 100
            ax.plot(dates, pd.Series(z).rolling(30, min_periods=30).mean(), color=c, label=f'{degree}차 제거, 30일 평균')
        k = a['cut_indices'][j]
        ax.axvspan(dates[k-90], dates[k-1], color=color[0], alpha=.08)
        ax.axvspan(dates[k], dates[k+89], color=color[1], alpha=.08)
        ax.axvline(dates[k], color='#555555', linestyle='--')
        ax.axhline(0, color='#999999', linewidth=.6)
        ax.set(title=f"{ids[j]}, 주 분석 직전/직후 90일 차이 {a['delta90_pp'][j]:.3f}%p", ylabel='계절 제거 상대 차이 (CF %p)')
        date_axis(ax)
        ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, '04_candidates_residual', '두 계절식과 90일 비교 범위', '음영은 주 분석 탐색일 전후 90일. 평활 곡선은 우측 정렬이며 탐색 통계는 평활 전 일별 잔차로 계산했다.')

    fig, axes = plt.subplots(len(shown), 1, figsize=(12, 4*len(shown)), squeeze=False)
    stations = p['stations']
    for ax, j in zip(axes.ravel(), shown):
        original = int(np.flatnonzero(p['site_ids'] == ids[j])[0])
        peers = (stations == stations[original]) & (p['site_ids'] != ids[j])
        own = p['daily'][:, original] * 100
        peer = p['daily'][:, peers].mean(axis=1) * 100
        ax.plot(dates, pd.Series(own).rolling(30, min_periods=30).mean(), color=color[0], label='본인 일평균 CF의 30일 평균')
        ax.plot(dates, pd.Series(peer).rolling(30, min_periods=30).mean(), color=color[1], label='동료 평균 CF의 30일 평균')
        ax.axvline(dates[a['cut_indices'][j]], color='#555555', linestyle='--')
        ax.set(title=str(ids[j]), ylabel='09~16시 평균 CF (%)')
        date_axis(ax)
        ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, '05_own_and_peers', '본인과 동료의 원래 일별 수준', '비교를 위한 기술적 표시다. 상대 증가만으로 본인의 절대 성능 향상이나 동료의 장애를 판정하지 않는다.')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (degree, block) in zip(axes.ravel(), configs):
        maxima = r[f'null_max_{degree}_{block}']
        ax.hist(maxima, bins=24, color='#9eb1bb', alpha=.85)
        ax.axvline(np.quantile(maxima, .95), color='#777777', linestyle='--', label='전역 참조 95백분위')
        for position, j in enumerate(shown):
            ax.axvline(s['cases'][degree]['statistics'][j], color=color[position % 2], label=str(ids[j]))
        ax.set(title=f'{degree}차 / {block}일 블록', xlabel='전체 발전소/날짜 최대 통계', ylabel='재표집 수')
        ax.legend(fontsize=7)
    save_plot(fig, OUT, gallery, '06_global_reference', '날짜와 발전소 탐색을 반영한 공동 참조', '각 설정 499회. 경험적 p의 최소값과 간격은 0.002이며 점근 CUSUM p값을 쓰지 않았다.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    allsig = np.array([v['all_four_significant'] for v in s['sites']])
    final = np.array([v['robust_candidate'] for v in s['sites']])
    x, y = np.asarray(a['delta90_pp']), np.asarray(s['cases']['6']['delta90_pp'])
    axes[0].scatter(x[~allsig], y[~allsig], color='#b5bdc1', s=18, label='네 설정 미통과')
    axes[0].scatter(x[allsig], y[allsig], color=color[0], s=30, label='네 설정 통과')
    axes[0].scatter(x[final], y[final], facecolors='none', edgecolors=color[1], s=120, label='최종 지속 후보')
    for value in [-2, 2]:
        axes[0].axvline(value, color='#999999', linestyle='--', linewidth=.7)
    axes[0].set(xlabel='3차 식의 자체 탐색일 전후 90일 차이 (%p)', ylabel='6차 식의 자체 탐색일 전후 90일 차이 (%p)')
    axes[0].legend(fontsize=8)
    annotations = {}
    for j, v in enumerate(s['sites']):
        if v['all_four_significant']:
            axes[1].scatter(abs(a['cut_indices'][j]-s['cases']['6']['cut_indices'][j]), abs(x[j]),
                            color=color[1] if v['robust_candidate'] else color[0], s=35)
            key = (abs(a['cut_indices'][j]-s['cases']['6']['cut_indices'][j]), round(abs(x[j]), 3))
            annotations.setdefault(key, []).append(str(ids[j]))
    for point, labels in annotations.items():
        axes[1].annotate('/'.join(labels), point, xytext=(4, 4), textcoords='offset points', fontsize=7)
    axes[1].axvline(60, color='#999999', linestyle='--')
    axes[1].axhline(2, color='#999999', linestyle='--')
    axes[1].set(xlabel='계절식 간 탐색일 차이 (일)', ylabel='주 분석 90일 차이 절댓값 (CF %p)', xlim=(-5, 78))
    save_plot(fig, OUT, gallery, '07_sensitivity', '통계적 반복과 변화 크기의 차이', '6차 식의 자체 90일 차이는 기술적 비교다. 최종 지속/크기 기준은 사전 정의한 주 분석의 90일 차이에 적용했다.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mirror = [int(np.flatnonzero(ids == name)[0]) for name in ['PV059', 'PV246']]
    for c, j in zip(color, mirror):
        axes[0].plot(dates, pd.Series(r['residual_3'][:, j]*100).rolling(30, min_periods=30).mean(), color=c, label=str(ids[j]))
    axes[0].axhline(0, color='#999999', linewidth=.6)
    axes[0].set(ylabel='계절 제거 상대 차이, 30일 평균 (CF %p)', title='WB007, 두 발전소가 서로의 유일한 동료')
    date_axis(axes[0])
    axes[0].legend()
    width = .32
    for n, j in enumerate(shown):
        axes[1].bar(np.arange(3) + (n-.5)*width, a['thirds_pp'][j], width=width, color=color[n % 2], label=str(ids[j]))
    axes[1].axhline(0, color='#999999', linewidth=.7)
    axes[1].set(xticks=range(3), xticklabels=['가까운 30일', '그다음 30일', '먼 30일'], ylabel='전후 같은 거리 구간의 평균 차이 (%p)', title='주 분석 탐색일 전후의 세 구간')
    axes[1].legend()
    save_plot(fig, OUT, gallery, '08_mirror_and_persistence', '동료 의존성과 지속 기준의 의미', '거울 변화는 독립 사건 둘을 뜻하지 않는다. 지속 후보는 세 구간 모두 같은 방향이라는 뜻이며 각 구간 2%p 이상이라는 뜻은 아니다.')
    write_gallery(OUT, gallery, 'DA-004 상대 수준의 시간 구조')
    dump(OUT / 'figure_verification.json', check_gallery(OUT, 8))


def report(s):
    a, b = s['cases']['3'], s['cases']['6']
    rows, detail, raw_rows = [], [], []
    panel = np.load(OUT / 'panel.npz')
    for j, v in enumerate(s['sites']):
        if not v['all_four_significant']:
            continue
        maxp = max(s['cases'][d]['blocks'][bl]['p_adjusted'][j] for d in ['3', '6'] for bl in ['7', '14'])
        reason = '지속 후보' if v['robust_candidate'] else '탐색일 차이 >60일' if not v['dates_within_60d'] else '90일 크기/방향 미통과'
        rows.append([v['site_id'], v['station'], a['dates'][j], b['dates'][j], f"{a['delta90_pp'][j]:+.3f}", f'{maxp:.3f}', reason])
        if v['robust_candidate']:
            detail.append([v['site_id'], v['peers'], f"{a['delta_pp'][j]:+.3f}",
                           '/'.join(f'{x:+.3f}' for x in a['thirds_pp'][j]),
                           '/'.join(f"{s['cases'][d]['blocks'][bl]['p_adjusted'][j]:.3f}" for d in ['3','6'] for bl in ['7','14'])])
            k = a['cut_indices'][j]
            raw_change = (panel['relative'][k:k+90, j].mean() - panel['relative'][k-90:k, j].mean()) * 100
            seasonal_change = raw_change - a['delta90_pp'][j]
            raw_rows.append([v['site_id'], f'{raw_change:+.3f}', f'{seasonal_change:+.3f}', f"{a['delta90_pp'][j]:+.3f}"])
    counts = [[f'{d}차', f'{bl}일', s['cases'][d]['blocks'][bl]['significant_sites']] for d in ['3','6'] for bl in ['7','14']]
    prefix = REPORT.read_text(encoding='utf-8').split('## 결과\n\n')[0]
    text = '''## 결과

### 실행과 표본

2026-10-21 13:33 KST에 계산을 완료했다. 정규 CPython 3.13.1에서 공통 80개소의 731일, 09~16시 467,840행을 원본에서 집계해 이전 패널과 대조했다. 0 기록 6,059행을 유지했으며 CF 1 초과는 없었다. 동료가 없는 10개소를 제외한 70개소/17개 B 지점이 분석 대상이다. 제외 10개소의 안정성은 평가하지 않았다.

반복 계절식 2개와 각 식의 공동 재표집 998회, 합계 1,996회를 실행했다. 새로운 발전량 예측 학습/추론/평가/제출은 수행하지 않았다.

### 통계 기준을 통과하는 변화와 지속 후보는 다르다

설정별 전역 경험적 p<=0.05를 통과한 수는 다음과 같다. 14일 블록에서 줄어드는 수를 함께 보며 짧은 의존성 가정에 기대어 모든 변화를 선언하지 않는다.

'''
    text += table(['반복 계절식', '블록 길이', '통과 발전소'], counts)
    text += f'''

네 설정을 모두 통과한 곳은 {s['counts']['all_four_significant']}개소다. 그중 두 식의 탐색일이 60일 이내이고 변화 방향이 같은 곳은 {s['counts']['date_and_direction_pass']}개소, 주 분석의 90일 지속/2%p 크기까지 통과한 곳은 **{s['counts']['robust_candidates']}개소/{s['counts']['candidate_stations']}지점**이었다. 미탐지는 안정성의 증명이 아니다.

다음 값은 원본 행이 아닌 분석 통계다. 90일 차이는 계절 제거 동료 차이의 직후 평균에서 직전 평균을 뺀 CF %p이고, 최대 p는 네 설정의 보정 p 중 가장 큰 값이다.

'''
    text += table(['발전소', 'B 지점', '3차 탐색일', '6차 탐색일', '주 분석 90일 차이', '최대 p', '판단'], rows)
    text += '''

### 두 후보는 계절 제거 후 상대 차이에서 선택됐다

PV112는 주 분석 탐색일 2023-04-22 전후 90일 차이가 +3.945%p였다. 6차 식에서는 2023-04-14가 선택됐다. PV117은 2023-11-26 전후 +2.011%p였고 6차 식에서는 2023-12-06이 선택됐다. 이는 원래 발전량의 증가율, 발전 효율의 개선, 기록 시각 오류를 뜻하지 않는다. 본인과 동료의 변화가 모두 상대 차이에 들어간다.

PV117은 사전 크기 기준 2%p를 약 0.011%p 웃돈다. PV112의 최대 보정 p는 0.046으로 0.05에 가깝다. 두 후보 모두 이번 기준을 통과했다는 범위로 해석하며 강한 임계값 독립적 증거라고 부르지 않는다. 499회 기준 p의 간격과 최소값은 0.002다. p=0.002도 정확히 0이거나 미래 재현이 확실하다는 뜻이 아니다.

'''
    text += table(['후보', '동료 수', '전체 전후 차이 (%p)', '가까운/중간/먼 30일 차이 (%p)', 'p: 3차7일/3차14일/6차7일/6차14일'], detail)
    text += '''

세 30일 차이가 모두 같은 방향이라는 지속 기준을 썼다. 각 30일 차이가 모두 2%p 이상인 것은 아니다. 탐색일과 전후 차이는 같은 자료에서 선택한 기술 통계이며 사후 선택을 보정한 효과 신뢰구간은 계산하지 않았다.

### 원래 경로와 대조하면 실제 변화일로 단정하기 어렵다

그림을 확인한 뒤 해석 검증을 보완했다. 추가 계산 전에 계획 이력에 이유를 기록하고, 이미 선택된 두 후보의 같은 전후 90일에서 원래 상대 차이와 적합 계절 성분의 차이를 기술 대조했다. 선택 기준을 바꾸거나 새 계절식을 추가 적합하지 않았다.

'''
    text += table(['후보', '원래 상대 차이의 90일 변화 (%p)', '적합 계절 성분의 90일 변화 (%p)', '계절 제거 후 90일 변화 (%p)'], raw_rows)
    text += '''

PV112는 원래 상대 차이의 두드러진 구간이 2024년 봄에 집중돼 있다. 두 해에 공통인 계절식은 그 현상을 두 해의 같은 계절에 나누어 반영하므로, 현상이 약한 2023년 봄에는 음의 잔차를 만들 수 있다. 이 경로에서 선택된 2023-04-22를 실제 수준 변경일로 받아들이면 안 된다. 원래 상대 차이와 계절 제거 후의 전후 차이도 서로 다른 값을 보인다. 3차와 6차가 모두 통과했다는 사실은 두 식이 공유한 반복 계절 가정까지 검증하지 않는다.

PV117도 크기 임계값에 가깝고 연말에는 상대 차이가 다시 줄어드는 경로를 보인다. 따라서 **두 곳은 사전 지속 기준 통과 후보이며, 지속적인 실제 구조 변화가 확정된 두 발전소라는 결론은 아니다.** 반복 계절과 국소 현상의 분리는 DA-009의 선택 후보로 등록했으며 추가 실험은 실행하지 않았다.

### 개별 발전소의 독립 사건 수로 세면 안 된다

WB007의 PV059와 PV246은 서로가 유일한 동료여서 상대 차이가 정확히 부호만 반대다. 같은 날짜와 통계로 둘 다 통과하지만 같은 비교 한 쌍을 두 독립 사건으로 세면 안 된다. 두 후보 모두 90일 크기는 약 0.708%p라 지속 후보에서 제외됐다. 이처럼 같은 지점 비교는 공통 변동을 줄이면서도 동료 의존성을 만든다. 전체 분석에는 이런 두 발전소 지점이 5쌍 있다.

WB004의 PV220은 두 계절식의 탐색일이 2024-04-26과 2024-02-22로 64일 차이였다. 네 설정의 통계 기준을 통과했어도 사전 날짜 기준을 넘으므로 지속 후보에 포함하지 않았다. 날짜 하나만 골라 실제 변경일로 고정하기 어렵다는 예다.

DA-002에서 연간 정규화 곡선 유형이 유지됐다는 결과와 이번 결과는 함께 성립할 수 있다. 연간 모양의 소속이 같아도 동료 대비 수준은 기간 중 달라질 수 있다. 두 유형이 모든 시간 구조의 안정성을 보장하는 것은 아니다.

### 검증과 실행 이력

합성 테스트 9개가 통과했다. 원본 일별 집계 58,480셀과 동료 차이 51,170셀을 별도 계산하고, SVD 최소제곱 잔차와 직접 전후 평균으로 두 계절식의 전체 탐색 통계를 검산했다. 네 설정 280개 전역 p, 저장 참조 분포 1,996개와 최초 재표집 4개의 별도 재구성, 후보 판정과 거울 관계를 확인했다. 전체 1,996회를 독립 재실행했다고 주장하지 않는다. 원본 두 파일과 계산 코드 해시를 전후 확인했다.

첫 계산은 pandas 날짜 검증 함수의 인자 호환성 오류로 통계 계산 전에 중단했다. 날짜 목록을 직접 비교하도록 고친 뒤 동일 계약의 계산을 완료했다. 검산에서는 사전 문서 복사본의 LF/CRLF 차이를 정규화해 원래 내용 해시를 확인했고, 이 분석이 생성한 지점 문자열 배열의 저장 형식을 명시해 읽었다. 통계나 임계값을 바꾸지는 않았다.

그림 8개를 로컬에 생성해 파일과 갤러리 링크 16개를 확인했다. 절반 크기 검토 시트 두 장에서 원래 경로, 계절식과 잔차, 필터, 전역 참조와 민감도를 확인했다. 겹친 두 발전소 표기를 묶어 수정한 뒤 다시 확인했다. 개별 패널과 재표집 결과, 그림은 `data/analysis/2026-10-21_temporal_relative/`에 로컬로만 보존한다.

## 한계 및 다음 단계

이번 종료 범위는 같은 B 지점 동료 대비 수준의 반복 계절 제거 후 단일 변화다. 기상 관계 계수 전체와 지점 공통 수준의 안정성을 검증한 결과는 아니다. 같은 B 지점이라도 실제 일사와 방향, 지역 기상 차이는 남는다. 정적 용량의 과거 적용과 두 해로 추정한 계절식에도 한계가 있다.

공동 블록 재표집은 잔차의 대략적인 정상성과 선택한 의존성 범위를 가정한다. 변화가 포함된 잔차를 다시 표집하므로 전역 경험적 p는 참조 기준이며 유한표본 오류율의 보장을 뜻하지 않는다. 7/14일보다 긴 의존성과 변동성 변화, 단기 이상과 왕복 변화, 완만한 이동, 여러 변화는 이번 단일 분할 분석으로 충분히 다루지 못한다. 원인을 밝혀야 종료되는 작업으로 확장하지 않는다.

분석 관점의 결론은 **일중 곡선의 연간 유형만으로 모든 발전소의 상대 수준이 시간 내내 일정하다고 가정하기 어렵다**는 것이다. 시간별 수준 변화가 향후 설명 변수나 학습 기간 판단에 도움이 될 가능성은 있으나 성능 개선은 미검증이다. 이번 결과로 값을 보정하거나 기간을 삭제하지 않는다.

정한 분석과 민감도, 독립 검산 및 한계 기록을 마쳐 DA-004를 종료한다. 다음 우선 후보는 DA-005 A/B의 공통/고유 정보 구조다. 원인 추적이 필요하면 기존 EDA-009와 DA-008의 추가 자료 보류 범위에 연결하며 자동 실행하지 않는다.

## 관련 코드

- [계산](../scripts/temporal_relative_structure.py)
- [독립 검산](../scripts/verify_temporal_relative_structure.py)
- [그림과 결과 요약](../scripts/report_temporal_relative_structure.py)
- [합성 검증](../tests/test_temporal_relative_structure.py)
'''
    REPORT.write_text(prefix + text, encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['figures', 'report', 'all'])
    args = parser.parse_args()
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    if args.mode in ['figures', 'all']:
        figures(s)
    if args.mode in ['report', 'all']:
        report(s)
    print('Saved DA-004 rendering complete.')
