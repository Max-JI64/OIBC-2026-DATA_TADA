"""Render saved DA-012 estimates and checks; no new statistical fits."""
import argparse
import json

import numpy as np

from type_weather_response import ROOT, OUT, REPORT, dump

CASES = ['observed_2024', 'observed_2023', 'forecast_2024', 'forecast_2023']
LABELS = ['실측 2024', '실측 2023', '17시 예보 2024', '17시 예보 2023']
EFFECTS = ['calendar', 'date_hour']
ENAMES = {'calendar': '월/시간', 'date_hour': '월/시간+날짜/시간'}
COLORS = ['#246c88', '#b26039']
VNAMES = {'cloud': '운량', 'humidity': '습도'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def num(x):
    return f'{x:+.3f}'


def interval(r, key, block):
    lo, hi = r['bootstrap'][str(block)][key]['interval']
    return f'[{num(lo)}, {num(hi)}]'


def direction(r, key):
    signs = []
    for block in ['7', '14']:
        lo, hi = r['bootstrap'][block][key]['interval']
        signs.append(1 if lo > 0 else -1 if hi < 0 else 0)
    return signs[0] if signs[0] == signs[1] else 0


def get(s, case, sample, effect):
    return s['cases'][case]['samples'][sample][effect]


def figures(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot, write_gallery, check_gallery
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    gallery = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, mode in enumerate(['absolute', 'relative']):
        for col, var in enumerate(VNAMES):
            ax = axes[row, col]
            for offset, sample, color, name in [(-.13, 'all', COLORS[0], '전체 80개소'),
                                                (.13, 'shared', COLORS[1], '같은 B 7지점/41개소')]:
                values = [get(s, c, sample + '_' + mode, 'date_hour')['metrics']['delta_' + var] for c in CASES]
                ax.plot(values, np.arange(4) + offset, 'o', color=color, label=name)
            ax.axvline(0, color='#888888', lw=.8)
            ax.set(yticks=range(4), yticklabels=LABELS, title=VNAMES[var] + (' / CF' if row == 0 else ' / 준거 상대값'),
                   xlabel='C2-C1 기울기 차이 (기상 10%p당)')
            ax.invert_yaxis()
            ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, '01_scope_scale', '비교 범위와 척도에 따른 차이: 날짜/시간 추가 식',
              '점 추정치다. 전체는 발전소 동일 가중, 같은 지점은 지점 동일 가중이다. 두 식의 날짜 효과도 같은 비교량이 아니다. 위 행은 CF %p, 아래 행은 고정 준거의 %다.')

    for mode, number, title in [('absolute', '02', '같은 B 지점 유형 간 차이: CF'),
                               ('relative', '03', '같은 B 지점 유형 간 차이: 고정 준거 상대값')]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 7))
        for ax, var in zip(axes, VNAMES):
            for ei, effect in enumerate(EFFECTS):
                for i, case in enumerate(CASES):
                    r = get(s, case, 'shared_' + mode, effect)
                    v = r['metrics']['delta_' + var]
                    y = i + (ei - .5) * .28
                    for block, width in [('14', 3.5), ('7', 1.2)]:
                        lo, hi = r['bootstrap'][block]['delta_' + var]['interval']
                        ax.hlines(y, lo, hi, color=COLORS[ei], lw=width, alpha=.35 if block == '14' else 1)
                    ax.plot(v, y, 'o', color=COLORS[ei], ms=5, label=ENAMES[effect] if i == 0 else None)
            ax.axvline(0, color='#888888', lw=.8)
            ax.set(yticks=range(4), yticklabels=LABELS, title=VNAMES[var],
                   xlabel='기상 10%p당 C2-C1 차이 (' + ('CF %p' if mode == 'absolute' else '고정 준거의 %') + ')')
            ax.invert_yaxis()
            ax.legend(fontsize=8)
        save_plot(fig, OUT, gallery, number + '_shared_' + mode, title,
                  '가는 선은 7일, 굵고 옅은 선은 14일 날짜 블록의 개별 95% 백분위 구간이다. 날짜와 모든 지점을 함께 재표집했고 다중 비교 보정 구간은 아니다.')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, mode in enumerate(['absolute', 'relative']):
        for col, var in enumerate(VNAMES):
            ax = axes[row, col]
            for i, case in enumerate(CASES):
                point = get(s, case, 'shared_' + mode, 'date_hour')['metrics']['delta_' + var]
                sens = s['cases'][case]['point_sensitivities'][mode + '_date_hour']
                loo = [x['metrics']['delta_' + var] for x in sens['leave_one_station_out']]
                strict = sens['min_two_each']['metrics']['delta_' + var]
                ax.hlines(i, min(loo), max(loo), color='#7295a4', lw=3, label='한 지점 제외 범위' if i == 0 else None)
                ax.plot(point, i, 'o', color=COLORS[0], label='기본 7지점' if i == 0 else None)
                ax.plot(strict, i, 'D', color=COLORS[1], label='각 유형 2개 이상 4지점' if i == 0 else None)
            ax.axvline(0, color='#888888', lw=.8)
            ax.set(yticks=range(4), yticklabels=LABELS, title=VNAMES[var] + (' / CF %p' if row == 0 else ' / 준거의 %'),
                   xlabel='기상 10%p당 C2-C1 기울기 차이')
            ax.invert_yaxis()
            ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, '04_station_sensitivity', '지점 구성 민감도: 날짜/시간 추가 식',
              '선은 한 지점을 차례로 제외한 7개 점 추정치의 최소/최대이며 불확실성 구간이 아니다. 4지점 부분집합도 별도 구간 없는 기술 비교다.')
    write_gallery(OUT, gallery, 'DA-012 발전 유형별 기상 반응 차이')
    return check_gallery(OUT, 4)


def report(s, v, qa, visual_checked):
    prefix = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').split('## 결과')[0]
    prefix = prefix.replace('상태: DA-012 진행, 메타 가용성 점검 완료.', '상태: DA-012 계산과 독립 검산 완료.')
    text = prefix + '''## 결과

### 1. 결론과 비교량

두 곡선 유형이 운량/습도에 서로 다르게 반응하는 기상 집단이라고 일반화할 근거는 이번 범위에서 확보하지 못했다. 전체 80개소의 상대 척도에서 보이는 차이가 같은 B 지점 비교에서는 유지되지 않았고, 같은 지점 안에서도 연도, 실측/예보, 날짜 요인과 지점 구성에 따라 달라졌다. 일부 조건의 차이는 남아 있으므로 두 유형의 반응이 같다고 입증한 결과도 아니다.

유형 간 차이는 C2의 기울기에서 C1 기울기를 뺀 값이다. 양수는 C2 기울기가 더 크다는 뜻이며, 그 자체로 발전량 증가를 뜻하지 않는다. 예를 들어 두 유형 기울기가 모두 음수라면 양의 차이는 C2 쪽 음의 관계가 더 약하다는 뜻이다.

전체 식의 날짜/시간 효과는 두 유형에 공통인 CF 변동을 제거한다. 같은 지점 차이 식의 날짜/시간 효과는 유형 간 격차에 공통인 날짜 변동을 제거한다. 따라서 두 식은 같은 표본에 동일한 조정만 가한 전후 비교가 아니다. 표본과 가중뿐 아니라 날짜 조정의 의미도 함께 달라진다.

### 2. 가용성과 실제 실행 범위

'''
    text += table(['범위', 'B 지점', 'C1/C2 개소', '합계', '가중'], [
        ['전체', '27', '47/33', '80', '발전소 동일'],
        ['같은 지점 주 비교', '7', '27/14', '41', '지점 동일'],
        ['각 유형 2개소 이상 민감도', '4', '22/11', '33', '지점 동일']])
    text += '\n\n' + table(['자료/연도', '균형 날짜', '발전소 시간 셀', '제외 날짜'], [
        [label, s['cases'][case]['audit']['balanced_dates'], s['cases'][case]['audit']['balanced_rows'],
         s['cases'][case]['audit']['excluded_dates']] for case, label in zip(CASES, LABELS)])
    text += '''

각 연도의 모든 날짜와 9~16시를 사용했다. 같은 시각 기상이 여러 발전소에 반복되므로 표의 발전소 시간 셀을 독립 기상 관측 수로 읽지 않는다. 같은 지점 표본 7곳은 공간적으로도 독립임을 보장하지 않는다. 유형과 준거는 사전 고정했고 새 군집이나 예측 모형을 만들지 않았다.

### 3. 전체 80개소: 유형별 CF 기울기

기상 10%p당 CF %p 단위다. 운량/습도 외 5개 기상을 함께 통제한다. 아래는 점 추정치이며 유형 차이의 두 블록 구간과 표본/척도 민감도를 이어서 본다.

'''
    text += table(['자료/연도', '고정효과', '변수', 'C1', 'C2', 'C2-C1', '차이의 7일 구간', '차이의 14일 구간'], [
        [label, ENAMES[effect], VNAMES[var], num(r['metrics']['C1_' + var]), num(r['metrics']['C2_' + var]),
         num(r['metrics']['delta_' + var]), interval(r, 'delta_' + var, 7), interval(r, 'delta_' + var, 14)]
        for case, label in zip(CASES, LABELS) for effect in EFFECTS
        for r in [get(s, case, 'all_absolute', effect)] for var in VNAMES])
    text += '''

### 4. 같은 B 지점 안의 유형 차이

다음 두 표는 같은 7지점/41개소의 결과다. CF 표는 기상 10%p당 CF %p, 상대 표는 기상 10%p당 고정 2023년 평균 CF의 % 단위다. 상대 표의 숫자는 현재 시점 발전량의 변화율이나 인과효과가 아니다. 발전소마다 다른 준거로 먼저 나눈 뒤 유형 평균을 내므로 전체 표를 상수 하나로 배율 변환한 결과도 아니다.

'''
    for mode, heading in [('absolute', 'CF 척도'), ('relative', '고정 준거 상대 척도')]:
        text += f'#### {heading}\n\n'
        text += table(['자료/연도', '고정효과', '변수', 'C2-C1', '7일 구간', '14일 구간', '두 블록 방향'], [
            [label, ENAMES[effect], VNAMES[var], num(r['metrics']['delta_' + var]), interval(r, 'delta_' + var, 7),
             interval(r, 'delta_' + var, 14), {1: '양', -1: '음', 0: '불확실'}[direction(r, 'delta_' + var)]]
            for case, label in zip(CASES, LABELS) for effect in EFFECTS
            for r in [get(s, case, 'shared_' + mode, effect)] for var in VNAMES]) + '\n\n'
    text += '''실측 CF의 운량 차이는 두 해와 두 날짜 조정 모두 개별 구간이 0을 포함했다. 실측 CF의 습도는 월/시간 식의 2023년에서 양의 차이가 남았으나 2024년에는 블록 길이에 따라 0 포함 여부가 달랐고 날짜/시간을 추가한 두 해에서는 불확실했다.

17시 예보 CF의 습도는 월/시간 식에서 두 해 모두 양의 차이가 남았지만 상대 척도에서는 두 해 모두 불확실했다. 날짜/시간 추가 후 예보 운량은 2024년만, 예보 습도는 2023년만 두 블록에서 양의 차이가 남았다. 상대 척도에서도 이 두 조건의 양의 차이는 남았으나, 이것을 모든 기간과 자료에서 반복되는 유형 특성으로 확대할 수 없다.

다음은 두 해/두 자료의 네 경우 모두에서 두 블록 구간이 같은 방향으로 0을 제외했는지를 요약한 표다. 이 기준은 다중 비교 보정 검정이나 등가성 검정이 아니다.

'''
    repetition = []
    for sample in ['all_absolute', 'all_relative', 'shared_absolute', 'shared_relative']:
        for effect in EFFECTS:
            row = [sample, ENAMES[effect]]
            for var in VNAMES:
                dirs = [direction(get(s, c, sample, effect), 'delta_' + var) for c in CASES]
                row.append(('양' if dirs[0] == 1 else '음') + ' / 4경우 반복' if len(set(dirs)) == 1 and dirs[0] else '4경우 공통 방향 미확보')
            repetition.append(row)
    text += table(['표본/척도', '고정효과', '운량 차이', '습도 차이'], repetition)
    text += '''

### 5. 지점 구성에 대한 민감도

각 칸은 `기본 7지점 / 각 유형 2개소 이상 4지점 / 한 지점 제외 최소~최대`다. 점 추정치만 비교하며 마지막 범위를 신뢰구간으로 해석하지 않는다. CF와 상대 척도의 단위는 위와 같다.

'''
    rows = []
    for case, label in zip(CASES, LABELS):
        for mode in ['absolute', 'relative']:
            for effect in EFFECTS:
                sens = s['cases'][case]['point_sensitivities'][mode + '_' + effect]
                cells = [label, 'CF' if mode == 'absolute' else '준거 상대', ENAMES[effect]]
                for var in VNAMES:
                    key = 'delta_' + var
                    point = get(s, case, 'shared_' + mode, effect)['metrics'][key]
                    strict = sens['min_two_each']['metrics'][key]
                    loo = [x['metrics'][key] for x in sens['leave_one_station_out']]
                    cells.append(f'{num(point)} / {num(strict)} / {num(min(loo))}~{num(max(loo))}')
                rows.append(cells)
    text += table(['자료/연도', '척도', '고정효과', '운량', '습도'], rows)
    text += '''

특히 날짜/시간 추가 식의 실측 CF에서는 기본 7지점의 운량/습도 차이가 두 해 모두 양수였지만, 각 유형 2개소 이상인 4지점에서는 두 변수 모두 음수였다. 예보 CF의 2024년 습도도 같은 제한에서 부호가 바뀌었다. 제한 표본은 지점을 바꾸므로 이 결과만으로 어느 추정치가 참인지 고를 수는 없다. 다만 소수 발전소가 한 유형을 대표하는 지점의 포함 여부에 관계 크기와 방향이 민감함을 확인했다.

한 지점을 제외한 점 추정치가 모두 같은 부호인 조건도 있다. 예를 들어 날짜/시간 추가 식의 2024년 예보 운량은 CF와 상대 척도 모두 양수다. 한 지점 제외의 방향 유지와 전체 날짜 재표집 구간, 다른 연도/자료의 재현은 서로 다른 확인이며 하나만으로 안정적인 일반 특성을 선언하지 않는다.

### 6. 실행과 검증

'''
    bootstrap_success = [stat['success'] for case in s['cases'].values() for sample in case['samples'].values()
                         for fit in sample.values() for block in fit['bootstrap'].values() for stat in block.values()]
    text += f'''정규 CPython {s['python'].split()[0]}에서 본 계산 1회와 독립 검산을 실행해 모두 종료 코드 0을 확인했다. 합성 경계와 계수 동치를 다룬 테스트 8개도 통과했다. 원본과 유형 배정, 고정 준거를 다시 대조했고 원본 해시는 유지됐다.

- 기본 식 32개와 지점 민감도 점 적합 128개를 계산했다. 4자료/연도 × 4표본/척도 × 2고정효과 × 2블록 × 100회의 재표집으로 6,400개 식을 적합했다. 모든 저장 계수의 블록별 성공 횟수는 최소 {min(bootstrap_success)}회, 최대 {max(bootstrap_success)}회였다.
- 독립 검산은 원본 발전량 {v['source_generation_cells']:,}개 셀과 발전소에 연결한 기상 {v['source_weather_cells']:,}개 셀을 재구성했다. 기상 셀은 공유 지점 값이 발전소 수만큼 반복된 검산량이며 독립 관측 수가 아니다.
- 기본 공분산 블록 {v['point_crosses']}개와 민감도 적합 {v['sensitivity_point_fits']}개를 별도 투영/최소제곱으로 확인했다. 직접 더미 방식의 월/시간 블록 점검 {v['explicit_dummy_month_hour_checks']}개, 각 블록 첫 추출의 계수 재계산 {v['bootstrap_crosses']}개, 날짜 가중 추출 {v['date_weight_draws']}개와 저장 구간 전체를 대조했다. 기본 공분산 블록 최대 절대 오차는 {v['max_point_cross_error']:.2e}였다.
- 전체 표본의 같은 지점/유형 압축은 원래 발전소 기울기와 동치임을 확인했다. 압축 반응의 제곱합을 원래 발전소의 설명력으로 쓰지 않았으며 R²나 예측 성능을 보고하지 않는다.
- 그림 {qa['figures']}개와 갤러리 연결 {qa['gallery_links']}개를 검사했다. '''
    text += '네 그림을 모은 검토 이미지에서 글자, 범례, 축과 구간 표시를 직접 확인했다.\n' if visual_checked else '직접 시각 검토는 아직 완료하지 않았다.\n'
    text += '''
7/14일 각각 100회의 개별 백분위 구간은 제한된 재표집에 따른 기술 구간이다. 레이블과 준거의 추정 불확실성, 새 기상 지점 모집단에 대한 공간 표집 불확실성, 다중 비교를 반영한 확정 검정으로 취급하지 않는다.

## 한계와 종료 판단

유형은 발전량에서 만들었고 두 해의 형태를 이미 본 상태다. 이번에는 고정 유형의 다른 기상 관계를 살폈으며 독립된 새 기간 검증은 아니다. 기상 자료는 설비 위치의 실제 일사/설비 조건을 모두 측정하지 않는다. 선형 설명식, 정적 용량의 과거 적용과 고정 준거 가정, 7개 공유 지점과 불균형한 유형별 발전소 수가 결론의 범위를 제한한다. 기존 DA-007의 엄격한 단독 기상 대비 지지 부족도 해소한 분석이 아니다.

기상과의 관계가 전혀 없다는 결론도, 유형별 관계가 같다는 결론도 내리지 않는다. 이번 결과는 곡선의 두 유형을 일관된 기상 반응의 두 집단으로 곧바로 해석하는 판단을 보류하게 한다. 개별 발전소의 모든 반응 이질성을 검정한 결과도 아니다. 설비 방향, 장애, 시각 오류나 원인은 확정하지 않는다.

사전에 정한 가용성, 전체/같은 지점 비교, 척도/날짜/연도/블록/지점 민감도와 독립 검산 및 그림 확인 범위에서 DA-012를 종료한다. 원인 미확정을 이유로 새 필수 분석을 추가하지 않는다. 남은 DA-006, DA-009와 기존 EDA-004는 각각 변수 압축, 변화일 활용, 공동 변동 해석 목적에 따라 검토할 선택 후보로 유지한다. 모델링 재개 지시 전에는 새 학습, 튜닝과 발전량 예측으로 연결하지 않는다.

## 관련 코드와 산출물

- [본 계산](../scripts/type_weather_response.py), [합성 테스트](../tests/test_type_weather_response.py)
- [독립 검산](../scripts/verify_type_weather_response.py), [보고서와 그림 생성](../scripts/report_type_weather_response.py)
- [로컬 그림 갤러리](../data/analysis/2026-10-21_type_weather_response/index.html), [집계 결과](../data/analysis/2026-10-21_type_weather_response/summary.json), [검산 결과](../data/analysis/2026-10-21_type_weather_response/verification.json)
- [DA-002 유형](2026-10-21_08_발전소별_발전패턴_유형_분석.md), [DA-007 공동 지지/날짜 요인](2026-10-21_13_공동지지와_날짜요인의_조건부관계_분석.md), [진행 계획](데이터분석_진행계획.md)

원본 행, 연결 행과 계산 출력은 로컬 전용이다. 보고서는 집계 결과만 담는다. 코드/보고서/계획/로그만 커밋하며 그림과 데이터 출력은 Git에 포함하지 않는다.

방법 확인: [statsmodels 교호작용 예제](https://www.statsmodels.org/stable/examples/notebooks/generated/interactions_anova.html). 실제 계산과 검산은 위 코드에서 수행했다.
'''
    REPORT.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual-checked', action='store_true')
    args = parser.parse_args()
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    v = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    qa = figures(s)
    report(s, v, qa, args.visual_checked)
    dump(OUT / 'report_checks.json', dict(qa, visual_checked=args.visual_checked))
    print(json.dumps(dict(qa, visual_checked=args.visual_checked)))


if __name__ == '__main__':
    main()
