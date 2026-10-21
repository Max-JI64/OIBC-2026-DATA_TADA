"""Render verified DA-003 tables and conditional association curves, no refitting."""
import argparse
import json
import sys

import numpy as np

from weather_conditional_relationships import (OUT, REPORT, VARIABLES, REFERENCE,
    CONTRASTS, design)

LABELS = {'observed_2024': '실측 2024', 'observed_2023': '실측 2023',
          'forecast_2024': '17시 예보 2024', 'forecast_2023': '17시 예보 2023'}
CONTRAST_LABELS = {
    'cloud_dry': '운량 25→75%, 강수 0', 'cloud_wet': '운량 25→75%, 강수 1 mm',
    'rain_cloud25': '강수 0→1 mm, 운량 25%', 'rain_cloud75': '강수 0→1 mm, 운량 75%',
    'wind_temp10': '풍속 1→5 m/s, 기온 10°C', 'wind_temp30': '풍속 1→5 m/s, 기온 30°C',
    'temp_wind1': '기온 10→30°C, 풍속 1 m/s', 'temp_wind5': '기온 10→30°C, 풍속 5 m/s',
    'humidity': '습도 40→80%', 'uv': 'UV 1→5'}
VAR_LABELS = {'cloud': '운량 (%)', 'humidity': '상대습도 (%)', 'temp': '기온 (°C)',
              'wind': '풍속 (m/s)', 'uv': 'UV', 'rain': '강수 (mm)'}
STATUS = {'same direction': '방향 반복', 'opposite direction': '방향 변화', 'uncertain': '불확실',
          'insufficient support': '표본 부족', 'fit or bootstrap failure': '적합 실패'}


def number(x):
    return '미산출' if x is None else f'{x:+.2f}'


def effect_text(value):
    if not value:
        return '미산출'
    interval = value['interval']
    text = number(value['pp']) + ' [' + ', '.join(number(x) for x in interval) + ']'
    return text + (' *' if not value['supported'] else '')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def curve_design(var, values):
    weather = np.tile(np.array([REFERENCE[v] for v in VARIABLES]), (len(values), 1))
    weather[:, VARIABLES.index(var)] = values
    x, _, _ = design(weather)
    ref, _, _ = design(np.array([REFERENCE[v] for v in VARIABLES]))
    return 100 * (x - ref)


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot, write_gallery
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    gallery = []
    colors = {'full': '#225f87', 'no_uv': '#bd7542', 'single': '#737d81'}
    for label, case in summary['cases'].items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        with np.load(OUT / f'{label}_results.npz') as saved:
            for ax, var in zip(axes.ravel(), VARIABLES):
                lo, _, hi = case['ranges'][var]
                grid = np.linspace(lo, hi, 81)
                if var == 'rain' and hi > .1:
                    grid = np.unique(np.r_[grid, .1, .100001])
                delta = curve_design(var, grid)
                for model, name in [('single_' + var, '단독'), ('no_uv', '공동, UV 제외'), ('full', '공동, UV 포함')]:
                    if model == 'no_uv' and var == 'uv':
                        continue
                    fit = case['models'][model]
                    if not fit['valid']:
                        continue
                    cols = fit['columns']
                    estimates = delta[:, cols] @ np.asarray(fit['beta'])
                    boots = saved['boot_' + model] @ delta[:, cols].T
                    bounds = np.nanquantile(boots, [.025, .975], axis=0)
                    color = colors['single'] if model.startswith('single') else colors[model]
                    ax.plot(grid, estimates, label=name, color=color, linewidth=1.7,
                            linestyle='--' if model.startswith('single') else '-')
                    if model == 'full':
                        ax.fill_between(grid, bounds[0], bounds[1], color=color, alpha=.18)
                ax.axhline(0, color='black', linewidth=.6)
                ax.set(xlabel=VAR_LABELS[var], ylabel='기준값 대비 발전 비율 차이 (pp)')
                ax.legend(fontsize=7)
        save_plot(fig, OUT, gallery, label + '_curves', LABELS[label] + ' / 조건부 관계 곡선',
                  '고정효과는 모든 식에서 통제했다. 다른 변수는 사전 기준값. 가로축은 해당 변수의 2~98백분위다. 음영은 공동식의 날짜 블록 95% 구간. 전체 공변량의 공동 지지를 보장하지 않는다.')
    for kind in ['observed', 'forecast']:
        fig, ax = plt.subplots(figsize=(13, 8))
        for j, year in enumerate([2023, 2024]):
            case = summary['cases'][f'{kind}_{year}']
            for i, c in enumerate(CONTRASTS):
                v = case['models']['full']['contrasts'][c['id']]
                point, (lo, hi) = v['pp'], v['interval']
                y = i + (j - .5) * .23
                color = '#1f6a8a' if j == 0 else '#b96c39'
                ax.plot([lo, hi], [y, y], color=color, linewidth=1.7)
                ax.scatter(point, y, color=color if v['supported'] else 'white', edgecolor=color, s=40,
                           label=str(year) if i == 0 else None, zorder=3)
        ax.axvline(0, color='black', linewidth=.8)
        ax.set_yticks(range(len(CONTRASTS)), [CONTRAST_LABELS[c['id']] for c in CONTRASTS])
        ax.invert_yaxis()
        ax.set_xlabel('공동 설명식의 대비 (pp)')
        ax.legend()
        save_plot(fig, OUT, gallery, kind + '_annual_contrasts',
                  ('실측' if kind == 'observed' else '17시 예보') + ' / 두 해의 조건부 대비',
                  '선은 날짜 블록 95% 구간. 빈 점은 근방 표본 기준 미통과. 인과 효과 또는 예측 성능이 아니다.')
    selected = ['cloud_dry', 'rain_cloud75', 'wind_temp30', 'humidity', 'uv']
    fig, axes = plt.subplots(1, len(selected), figsize=(17, 6))
    for ax, key in zip(axes, selected):
        vals, labels = [], []
        for label, case in summary['cases'].items():
            with np.load(OUT / f'{label}_results.npz') as saved:
                a = saved['site_values_' + key]
                if len(a):
                    vals.append(a)
                    labels.append(LABELS[label] + f'\nn={len(a)}')
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.tick_params(axis='x', rotation=60)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_title(CONTRAST_LABELS[key], fontsize=9)
        ax.set_ylabel('발전소별 조건부 대비 (pp)')
    save_plot(fig, OUT, gallery, 'site_heterogeneity', '발전소별 기술적 관계의 산포',
              '개별 근방 기준과 수치 적합성을 통과한 발전소만 표시. 수축 추정이나 확정 군집이 아니며 이상점은 그림에서만 생략했다.')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    keys = ['cloud_dry', 'rain_cloud75', 'humidity', 'wind_temp30']
    for ax, kind in zip(axes, ['observed', 'forecast']):
        for j, year in enumerate([2023, 2024]):
            case = summary['cases'][f'{kind}_{year}']
            for i, key in enumerate(keys):
                v = case['uv_adjustment'][key]
                point, (lo, hi) = v['pp_full_minus_no_uv'], v['interval']
                y = i + (j - .5) * .25
                color = '#1f6a8a' if j == 0 else '#b96c39'
                ax.plot([lo, hi], [y, y], color=color)
                ax.scatter(point, y, color=color, label=str(year) if i == 0 else None)
        ax.axvline(0, color='black', linewidth=.7)
        ax.set_yticks(range(len(keys)), [CONTRAST_LABELS[k] for k in keys])
        ax.set(title='실측' if kind == 'observed' else '17시 예보', xlabel='UV 포함식 - UV 제외식의 대비 차이 (pp)')
        ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, 'uv_adjustment', 'UV를 함께 고려했을 때 달라지는 관계',
              '동일 재표집을 이용한 짝지은 대비 차이와 95% 구간. UV 자체의 인과 효과나 변수를 넣은 예측 성능 개선이 아니다.')
    write_gallery(OUT, gallery, 'DA-003 기상과 발전량의 조건부 관계')


def report(summary):
    from eda_plot_utils import check_gallery
    from weather_conditional_relationships import dump
    dump(OUT / 'figure_verification.json', check_gallery(OUT, 8))
    check = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    header = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').split('## 결과')[0]
    header = header.replace('상태: DA-003 진행. 아래 계약을 계수와 조건부 관계 계산 전에 고정했다.',
                            '상태: DA-003 완료. 아래 계약은 계수 계산 전에 고정했으며 설명 변수와 비교 기준을 바꾸지 않았다.')
    body = ['## 결과', '',
        '기상을 하나씩 볼 때의 관계 크기는 여러 변수를 함께 고려하면 크게 달라졌다. 특히 실측 강수의 음의 관계는 '
        '공동 설명식에서 작아졌으며, 예보 강수의 추가 관계는 두 해 모두 뚜렷하지 않았다. 예보 습도의 음의 관계는 반복됐지만 '
        '예보 풍속과 운량의 개별 관계는 같은 수준으로 확정할 수 없었다. 이는 단순 상관에서 본 각 변수를 독립 정보로 읽기 어렵다는 결과다.', '',
        '### 실제 표본과 실행 규모', '']
    rows = []
    for label, case in summary['cases'].items():
        a = case['audit']
        rows.append([LABELS[label], a['sites'], a['stations'], a['balanced_dates'], f"{a['balanced_rows']:,}",
                     a['excluded_dates'], a['zero_cf_rows']])
    body += [table(['자료', '발전소', '고유 기상 지점', '날짜', '발전소-시간 행', '제외 날짜', '0 발전 행'], rows), '',
        '선택한 여섯 기상 변수와 발전량은 네 자료-연도 모두 정한 비교 범위를 통과해 날짜 제외가 없었다. '
        '1을 넘는 발전 비율도 이번 표본에서는 없었다. 같은 발전량이 실측식과 예보식에 각각 쓰였으므로 '
        '네 표본을 합쳐 독립 관측 수로 세지 않는다. 기상도 27개 지점 값이 80개 발전소에 반복 연결되며 '
        '행 독립 표준오차 대신 전체 발전소를 함께 옮긴 날짜 블록을 사용했다.', '',
        '완료된 주 실행에서는 연도/자료별 8개 설명식, 총 32개를 적합했다. 각 설명식의 블록 재표집 200회, 총 6,400개와 '
        '발전소별 공동식 320개도 모두 수치 기준을 통과했다. 표준화 교차곱의 조건수는 계산에 문제가 없는 범위였지만 '
        '관측의 인과적 식별이나 모든 공변량 조합의 지지를 보장하는 것은 아니다.', '',
        '### 2024년 단독 관계와 공동 관계의 차이', '',
        '아래 값의 단위는 발전 비율의 pp다. 단독식도 같은 발전소×월×시간 고정효과를 통제했다. '
        '별도 표시가 없으면 공동식은 UV를 포함한 여섯 기상 변수 및 사전 교호작용을 함께 포함한다. '
        '괄호가 아닌 대괄호는 날짜 블록 95% 백분위 구간이며 인과 효과의 신뢰구간으로 해석하지 않는다.', '']
    rows = []
    keys = ['cloud_dry', 'rain_cloud75', 'humidity', 'wind_temp10', 'temp_wind1', 'uv']
    variable_map = {c['id']: c['variable'] for c in CONTRASTS}
    for kind in ['observed', 'forecast']:
        case = summary['cases'][kind + '_2024']
        for key in keys:
            single = case['models']['single_' + variable_map[key]]['contrasts'][key]
            reduced = case['models']['no_uv']['contrasts'].get(key)
            full = case['models']['full']['contrasts'][key]
            rows.append(['실측' if kind == 'observed' else '17시 예보', CONTRAST_LABELS[key],
                         number(single['pp']), number(reduced['pp']) if reduced else '해당 없음', effect_text(full)])
    body += [table(['자료', '대비', '단독식', '공동 UV 제외', '공동 UV 포함 [95% 구간]'], rows), '',
        '실측 강수 0→1 mm의 단독 관계는 -23.47 pp였지만, 운량 75% 조건에서 다른 기상을 함께 고려한 관계는 '
        '-2.19 pp [-3.06, -1.32]였다. 강수가 동반하는 흐림, 습도, UV와의 공유 정보를 함께 고려하면 '
        '강수 자체에 남는 조건부 관계가 훨씬 작아진다. 이것은 독립적인 물리적 강수 효과를 분리했다는 뜻은 아니다.', '',
        '예보 강수는 같은 조건에서 +0.64 pp [-1.02, +2.42]였다. 0을 포함하므로 양의 효과가 있다고 해석하지 않으며 '
        '단독식의 -15.04 pp를 독립 설명력으로 간주하지도 않는다. 현재 식에서 불확실하다는 결과이며 '
        '예보 강수의 비선형 정보가 전혀 없다는 증거나 입력에서 삭제하라는 결론은 아니다.', '',
        '### 두 해에 반복된 관계와 남은 불확실성', '']
    for kind in ['observed', 'forecast']:
        body += ['#### ' + ('실측' if kind == 'observed' else '17시 예보'), '']
        rows = []
        for c in CONTRASTS:
            key = c['id']
            v23 = summary['cases'][kind + '_2023']['models']['full']['contrasts'][key]
            v24 = summary['cases'][kind + '_2024']['models']['full']['contrasts'][key]
            rows.append([CONTRAST_LABELS[key], effect_text(v23), effect_text(v24), STATUS[summary['replication'][kind][key]]])
        body += [table(['대비', '2023 pp [95% 구간]', '2024 pp [95% 구간]', '사전 기준 판단'], rows), '']
    body += ['별표 *는 근방 표본 기준을 통과하지 못한 대비다. 전체 조건부 대비는 40개이며 '
        '이를 다중 비교 보정한 확증적 검정으로 사용하지 않았다. 방향 반복도 이 표본, 설명식과 재표집 설정에 한정한다.', '',
        '예보 습도 40→80%의 관계는 2023년 -28.26 pp, 2024년 -30.43 pp로 반복됐고 두 구간 모두 0을 제외했다. '
        '두 해 모두 발전소별 기술적 적합 80개 전부에서 음의 방향이었다. 습도가 운량이나 UV와 완전히 같은 정보만 '
        '제공하는 것은 아니라는 근거지만, 습도 자체가 그만큼 발전을 감소시키는 인과 효과라고 말할 수는 없다. '
        '미측정 일사와 실제 날씨 차이를 대리하는 관계가 남을 수 있다.', '',
        '실측의 무강수 운량 관계는 두 해 모두 음의 구간을 유지했다. 반면 예보는 여러 기상을 함께 고려한 '
        '운량 25→75% 대비의 구간이 두 해 모두 0을 포함했다. 동일한 변수명이라도 실측 관계와 예보에 남는 '
        '조건부 정보는 다르다. 실측/예보 식은 동일한 물리 효과나 공급사 성능을 직접 비교한 결과가 아니다.', '',
        '예보 풍속은 10°C에서 2024년 -3.58 pp, 2023년 +1.19 pp였고 후자는 구간에 0을 포함했다. '
        '이를 보편적인 냉각 효과나 강풍 피해로 설명할 근거가 없다. 30°C의 풍속 관계도 두 해 모두 불확실했다. '
        '온도에 따라 바람 관계가 달라지는지의 교호작용 자체는 별도의 검증 없이 확정하지 않는다.', '',
        '기온 10→30°C에서 양의 대비가 남는 경우도 있었다. 이 수치는 같은 달/시간의 작은 변화에서 추정한 '
        '공통 곡선을 넓은 범위에 적용한 결과이며 미측정 일사, 계절 안의 날씨 변화와 공변량의 불완전한 공동 지지가 '
        '영향을 줄 수 있다. 태양광 모듈 온도나 실제 효율을 관측한 자료가 아니므로 기온 상승이 효율을 높인다는 결론은 내리지 않는다.', '',
        '### UV를 고려했을 때 줄어드는 운량의 추가 관계', '']
    rows = []
    for label, case in summary['cases'].items():
        v = case['uv_adjustment']['cloud_dry']
        rows.append([LABELS[label], number(v['pp_full_minus_no_uv']),
                     '[' + ', '.join(number(x) for x in v['interval']) + ']', v['paired_success']])
    body += [table(['자료', '운량 대비의 변화: UV 포함 - 제외 (pp)', '짝지은 95% 구간', '재표집 성공'], rows), '',
        '운량의 음의 대비는 UV를 포함하면 네 경우 모두 0에 가까워졌으며 짝지은 차이의 구간도 0을 제외했다. '
        'DA-001에서 함께 움직였던 운량/UV의 공유 정보가 발전량 관계를 해석할 때에도 중요하다는 결과다. '
        '다만 UV가 매개 변수나 일사 대리 변수일 수도 있어 이 비교를 운량의 총 인과 효과 추정으로 사용하지 않는다. '
        '실측의 무강수 운량은 UV를 고려해도 음의 관계가 남았으며 예보에서는 별도로 확정하기 어려웠다.', '',
        '### 드문 조건에서는 전체 풀링과 개별 발전소 지지가 다르다', '']
    rows = []
    for label, case in summary['cases'].items():
        s = case['support']['rain_cloud25']['endpoints'][1]
        h = case['heterogeneity']['contrasts']['rain_cloud25']
        rows.append([LABELS[label], s['rows'], s['dates'], s['sites'], '통과' if s['supported'] else '미통과', h['sites']])
    body += [table(['자료', '낮은 운량의 강수 1 mm 근방 행', '날짜', '발전소', '전체 근방 지지', '대비 양 끝 개별 지지 발전소'], rows), '',
        '예보에서 낮은 운량과 강수가 함께 있는 근방은 2024년 99행/17날짜, 2023년 64행/9날짜뿐이었다. '
        '따라서 그 조건의 추정값은 해석을 제한했다. 실측의 전체 표본은 기준을 통과했지만 개별 발전소 기준을 '
        '통과한 곳은 2024년 3개, 2023년 0개였다. 풀링된 평균 관계가 있어도 모든 발전소에 같은 수준으로 '
        '근거가 있다는 뜻은 아니다. 같은 지점의 반복 행을 독립 날씨 사건으로 세지 않는다.', '',
        '발전소별 기상 관계의 부호와 산포는 별도 로컬 결과에 보관했다. 이는 주어진 관측 노출과 추정 잡음이 '
        '함께 들어간 기술적 차이이며 설비 유형, 확정된 반응 군집 또는 발전소별 유의한 차이로 해석하지 않는다.', '',
        '## 한계 및 다음 단계', '',
        '분석 대상은 두 해 완전 기록 공통 80개소이며 전체 253개소를 대표한다고 보장할 수 없다. '
        '연도별로 동일한 기저와 식을 다시 적합했으며 독립 미래 시험은 아니다. 정적 용량/기상 매핑을 과거에 적용하는 가정, '
        '실측/예보 강수 정의 차이, 예보 오차와 미측정 실제 일사량의 영향이 남는다.', '',
        '고정효과는 발전소별 월/시간 평균 차이를 통제하지만 날짜별 전국/지역 공통 요인과 모든 시간 변화 교란을 '
        '제거하지 않는다. 근방 지지 기준도 대상 변수와 교호작용 조건만 확인한 것이며 모든 기상 및 고정효과 셀 안의 '
        '공동 지지를 보장하지 않는다. 특히 큰 기온 대비는 같은 달/시간의 실제 기온 분포가 충분히 겹치지 않을 수 있다. '
        '7일 블록 길이와 제한된 스플라인/교호작용은 사전 고정했고 다른 설정의 민감도는 수행하지 않았다.', '',
        'DA-003은 정한 범위에서 종료한다. 후속 DA-002에서는 발전소의 곡선과 기상 반응 유형을 볼 수 있지만 '
        '이번 발전소별 계수를 그대로 군집화하기보다 표본 지지와 추정 잡음을 먼저 고려해야 한다. '
        '엄격한 달력 셀 안 공동 지지와 날짜 공통 요인 민감도는 DA-007 선택 후보로 기록하며 이번 종료의 필수 조건으로 늘리지 않는다.', '',
        '모델링 활용은 아직 가설이다. 예보 습도와 공유 날씨 정보, 드문 강수 조합 및 불안정한 풍속 관계를 '
        '입력 표현과 표본 조건의 후보 근거로 삼을 수 있지만 예측 성능 향상은 검증하지 않았다. '
        '발전량 예측 모델 학습/튜닝/추론, 예측 파일 생성, V2/제공 평가 구간 분석, 공식 제출은 모두 0회다.', '',
        '## 실행과 검증', '',
        '정규 CPython 3.13.1의 승격 Start-Process로 실행했다. 본 분석은 세 번 시도했다. 첫 시도는 '
        'pandas의 읽기 전용 배열에 풍속 단위를 적용하는 준비 단계에서 종료 코드 1로 실패했다. '
        '복사 배열을 명시해 수정했다. 두 번째는 재표집에서 어느 달의 가중치가 0일 때의 처리를 '
        '12번째 재표집 전에 멈춰 종료 코드 1로 끝났다. 관측이 없는 고정효과 셀은 기여가 0이므로 '
        '해당 셀을 생략하도록 수정했고, 명시적 가중 더미 회귀와 동치인 검증을 추가했다. 재표집을 다시 뽑거나 '
        '통계 계약을 바꾸지는 않았다. 중단된 두 번째 시도에서는 점 적합 8개와 완료 재표집 적합 88개가 추가로 수행됐다.', '',
        '수정 후 합성 검증 9개와 세 번째 본 실행은 종료 코드 0으로 완료됐다. 검증은 스플라인 꼬리의 선형성, '
        '명시적 더미 OLS와 가중 고정효과 제거의 동치, 빈 달의 처리, 교차곱, 교호작용 대비, 수치 부적합 거부와 '
        '근방 표본/재표집 날짜 수를 확인했다. 첫 버전의 8개 검증 통과만으로 실제 자료 처리 전체가 검증됐다고 표현하지 않는다.', '',
        f"저장 패널을 다시 읽어 점 적합과 첫 재표집의 full/no_uv 총 {check['direct_svd_fits']}개를 "
        f"직접 평균 제거와 SVD 최소제곱으로 재계산했다. 계수 최대 절대 차이는 {check['max_direct_beta_absolute_error']:.3g}였다. "
        f"대비와 구간 {check['point_and_interval_contrasts']}건, 근방 지지 {check['support_checks']}건, "
        '짝지은 UV 차이, 개별 발전소 요약과 재표집 합계를 검증했다. 분석 코드 해시가 실행 전후 동일하며 '
        '원본 4개 파일의 SHA-256도 분석 전후와 재검증 때 동일했다.', '',
        '그림 8개와 갤러리 링크 16개를 검사하고 모두 50% 크기의 4개 묶음 두 장에서 직접 검토했다. '
        '원본 크기 전체 픽셀 검사를 뜻하지 않는다. 곡선의 범례와 단위, 빈 점의 표본 부족 표시와 '
        '연도별 구간을 구분해 확인했다.', '',
        '## 관련 코드와 로컬 결과', '',
        '- [분석 코드](../scripts/weather_conditional_relationships.py)',
        '- [합성 검증](../tests/test_weather_conditional_relationships.py)',
        '- [독립 저장 결과 검산](../scripts/verify_weather_conditional_relationships.py)',
        '- [보고서와 그림 생성](../scripts/report_weather_conditional_relationships.py)',
        '- [계산 전 고정 보고서](../data/analysis/2026-10-21_weather_conditional/preregistered_report.md)',
        '- [전체 결과](../data/analysis/2026-10-21_weather_conditional/summary.json)',
        '- [검증 결과](../data/analysis/2026-10-21_weather_conditional/verification.json)',
        '- [그림 갤러리](../data/analysis/2026-10-21_weather_conditional/index.html)',
        '- [분석 계획](데이터분석_진행계획.md)', '',
        '방법 참고: [제한적 3차 스플라인 기저](https://search.r-project.org/CRAN/refmans/Hmisc/html/rcspline.eval.html), '
        '[OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html). '
        '실제 적합은 numpy의 표준화 교차곱 풀이이며 SVD와 가중 더미 회귀로 검산했다.', '',
        '원본/전처리 행은 보고서에 복사하지 않았다. 패널, 계수와 개별 발전소 결과 등 데이터 출력은 로컬 data/에만 보관한다. '
        '공개 가능한 코드와 해석 보고서, 로그만 로컬 커밋하며 push는 하지 않는다.', '']
    text = header + '\n'.join(body)
    assert '\u00b7' not in text
    REPORT.write_text(text, encoding='utf-8')
    print(str(REPORT))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['figures', 'report'])
    args = parser.parse_args()
    data = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    if args.mode == 'figures':
        figures(data)
    else:
        report(data)
