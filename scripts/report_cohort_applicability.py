"""Describe and visualize saved DA-010 results; no new explanatory fits."""
import json

import numpy as np
import pandas as pd

from cohort_applicability import OUT, REPORT, COHORTS
from weather_latent_structure import dump

COLORS = {'L80': '#246b85', 'M68': '#c47735', 'S105': '#6c9065'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def interval(point, bounds):
    return f'{point:+.3f} [{bounds[0]:+.3f}, {bounds[1]:+.3f}]'


def figures(s, p, w, inspection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot, write_gallery, check_gallery
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    gallery = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    counts = np.array([[inspection['months'][c][12 + m]['sites_ge14days'] for m in range(12)] for c in COHORTS])
    im = axes[0].imshow(counts / np.array([80, 68, 105])[:, None], aspect='auto', cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(12):
            axes[0].text(j, i, str(counts[i, j]), ha='center', va='center', color='white' if counts[i, j] / [80, 68, 105][i] > .65 else 'black')
    axes[0].set(xticks=range(12), xticklabels=range(1, 13), yticks=range(3), yticklabels=COHORTS,
                xlabel='2024년 월', title='월 14일 이상 완전 날짜: 수와 집단 내 비율')
    fig.colorbar(im, ax=axes[0], shrink=.7)
    a = np.array([inspection['cohorts'][c]['current_groups'].get('A', 0) for c in COHORTS])
    b = np.array([inspection['cohorts'][c]['current_groups'].get('B', 0) for c in COHORTS])
    axes[1].bar(COHORTS, a, color='#547d98', label='현재 그룹 A')
    axes[1].bar(COHORTS, b, bottom=a, color='#b38456', label='현재 그룹 B')
    for i, (aa, bb) in enumerate(zip(a, b)):
        axes[1].text(i, aa / 2, str(aa), ha='center', va='center', color='white')
        axes[1].text(i, aa + bb / 2, str(bb), ha='center', va='center', color='white')
    axes[1].set(ylabel='발전소 수', title='현재 그룹 구성은 기상 제공처 구분과 다름')
    axes[1].legend()
    save_plot(fig, OUT, gallery, '01_coverage_composition', '관측 기간 집단의 가용성과 구성',
              '기록 시작은 실제 가동일이 아니다. 짧은 집단의 계절별 표본 구성이 다르다.')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    for row, cohort in enumerate(['M68', 'S105']):
        for col, metric in enumerate(['morning_diff_pp', 'cf_diff_pp']):
            ax = axes[row, col]
            for ref, style in [('all_L', '-o'), ('same_B_L', '--s')]:
                a = p.loc[p.cohort.eq(cohort) & p.reference.eq(ref)]
                g = a.groupby('month')[metric].agg(['mean', 'count'])
                ax.plot(g.index, g['mean'], style, label=ref)
                for m, r in g.iterrows():
                    ax.annotate(str(int(r['count'])), (m, r['mean']), textcoords='offset points',
                                xytext=(0, 7 if ref == 'all_L' else -14), ha='center', fontsize=7)
            ax.axhline(0, color='#777', linewidth=.6)
            ax.set(xlabel='2024년 월', ylabel='대상 - 기준 (%p)', xticks=range(1, 13),
                   title=f'{cohort}: ' + ('오전 7~11시 비중' if col == 0 else '9~16시 평균 CF'))
            ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, '02_calendar_profiles', '정확히 같은 날짜에서 비교한 형태와 수준',
              '숫자는 대상 발전소 수다. same_B_L은 같은 B 지점의 L80 동료가 2개소 이상인 부분집합이다.')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, ref in zip(axes, ['all_L', 'same_B_L']):
        for cohort in ['M68', 'S105']:
            a = p.loc[p.cohort.eq(cohort) & p.reference.eq(ref)]
            ax.scatter(a.morning_diff_pp, a.cf_diff_pp, s=12, alpha=.3, color=COLORS[cohort], label=cohort)
        ax.axhline(0, color='#777', linewidth=.6)
        ax.axvline(0, color='#777', linewidth=.6)
        ax.set(xlabel='오전 비중 차이 (%p)', ylabel='평균 CF 차이 (%p)', title=ref)
        ax.legend()
    save_plot(fig, OUT, gallery, '03_profile_heterogeneity', '발전소/월별 형태와 수준 차이의 산포',
              '집단 평균이 가까워도 개별 발전소가 같다는 뜻은 아니다. 점은 독립 표본으로 검정하지 않았다.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for cohort in COHORTS:
        a = w.loc[w.cohort.eq(cohort)]
        g = a.groupby('month').agg(candidate=('candidate_rows', 'sum'), supported=('supported_rows', 'sum'),
                                 passed=('passed', 'sum'), eligible=('passed', 'size'))
        axes[0].plot(g.index, 100 * g.supported / g.candidate, '-o', color=COLORS[cohort], label=cohort)
        axes[1].plot(g.index, g.passed, '-o', color=COLORS[cohort], label=cohort)
    axes[0].set(ylabel='행 기준 공동 지지율 (%)', ylim=(0, 105), xlabel='2024년 월', xticks=range(1, 13))
    axes[1].set(ylabel='월 지지 기준 통과 발전소 수', xlabel='2024년 월', xticks=range(1, 13))
    for ax in axes:
        ax.legend()
    save_plot(fig, OUT, gallery, '04_weather_support', '같은 날짜/시간의 제한된 공동 기상 지지',
              '자기 B 지점을 제외한 L80 고유 지점이 2개 이상 기상 허용차를 통과해야 한다. 대상 밖 조건의 일반화를 보장하지 않는다.')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    for j, ax in enumerate(axes):
        for k, cohort in enumerate(COHORTS):
            r = s['regression'][cohort]
            if not r['valid']:
                continue
            for offset, block in [(-.06, '7'), (.06, '14')]:
                bound = r['bootstrap'][block]['intervals_pp'][j]
                y = k + offset
                ax.hlines(y, *bound, color=COLORS[cohort], linewidth=2, alpha=.7)
                ax.plot(r['beta_pp'][j], y, 'o' if block == '7' else 's', color=COLORS[cohort], markersize=5)
        ax.axvline(0, color='#777', linewidth=.7)
        ax.set(yticks=range(3), yticklabels=['L80', 'M68', 'S105 중 92개소'], xlabel='기상 10%p당 CF 차이 (%p)',
               title=['운량', '습도'][j] + ': 원=7일 블록, 사각=14일 블록')
    save_plot(fig, OUT, gallery, '05_conditional_slopes', '하반기 고정 집단의 제한된 조건부 관계',
              '발전소/월/시간 고정효과와 사전 선형 공동 식이다. 개별 95% 기술 구간이며 인과효과가 아니다.')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    for j, ax in enumerate(axes):
        for k, cohort in enumerate(['M68', 'S105']):
            if cohort not in s['contrasts']:
                continue
            r = s['contrasts'][cohort]
            for offset, block in [(-.06, '7'), (.06, '14')]:
                bound = r['bootstrap'][block]['intervals_pp'][j]
                ax.hlines(k + offset, *bound, color=COLORS[cohort], linewidth=2)
                ax.plot(r['difference_pp'][j], k + offset, 'o' if block == '7' else 's', color=COLORS[cohort])
        ax.axvline(0, color='#777', linewidth=.7)
        ax.set(yticks=range(2), yticklabels=['M68 - L80', 'S105 중 92 - L80'],
               xlabel='계수 차이 (%p / 기상 10%p)', title=['운량', '습도'][j] + ': 원=7일, 사각=14일')
    save_plot(fig, OUT, gallery, '06_slope_differences', '공동 날짜 재표집으로 비교한 집단 계수 차이',
              '구간이 0을 포함하는 결과는 동등성을 입증하지 않는다. 고정 지점망과 선택한 지지 범위에 한정한다.')
    write_gallery(OUT, gallery, 'DA-010 장기 관측 표본의 적용 범위')
    return check_gallery(OUT, 6)


def main():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    inspection = json.loads((OUT / 'inspection.json').read_text(encoding='utf-8'))
    verification = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    p = pd.DataFrame(s['profiles'])
    w = pd.DataFrame(s['support'])
    gallery = figures(s, p, w, inspection)
    rows = []
    for cohort in ['M68', 'S105']:
        for m in [1, 4, 7, 10, 12]:
            a = p.loc[p.cohort.eq(cohort) & p.reference.eq('all_L') & p.month.eq(m)]
            b = p.loc[p.cohort.eq(cohort) & p.reference.eq('same_B_L') & p.month.eq(m)]
            rows.append([cohort, m, len(a), f'{a.morning_diff_pp.mean():+.2f}',
                         f'{a.center_diff_minutes.mean():+.2f}', f'{a.cf_diff_pp.mean():+.2f}', len(b),
                         f'{b.cf_diff_pp.mean():+.2f}' if len(b) else '없음'])
    profile_table = table(['집단', '월', '대상 수', '오전 비중 차이 (%p)', '중심 차이 (분)',
                           '낮 CF 차이 (%p)', '같은 B 부분집합 수', '같은 B CF 차이 (%p)'], rows)
    rows = []
    for cohort in COHORTS:
        a = w.loc[w.cohort.eq(cohort) & w.month.ge(7)]
        rows.append([cohort, len(a), int(a.passed.sum()),
                     f'{100*a.supported_rows.sum()/a.candidate_rows.sum():.1f}',
                     int(s['fixed_h2_sites'][cohort]), s['regression'][cohort]['audit']['rows']])
    support_table = table(['집단', '하반기 가용 발전소/월', '월 지지 통과', '낮 행 공동 지지율 (%)',
                           '하반기 고정 발전소 수', '회귀 지지 행'], rows)
    median_rows = []
    weather_rows = []
    for cohort in ['M68', 'S105']:
        for month in [7, 10, 12]:
            for reference in ['all_L', 'same_B_L']:
                a = p.loc[p.cohort.eq(cohort) & p.month.eq(month) & p.reference.eq(reference)]
                if len(a):
                    median_rows.append([cohort, month, reference, len(a),
                        f'{a.morning_diff_pp.median():+.2f}', f'{a.cf_diff_pp.median():+.2f}',
                        '[' + ', '.join(f'{v:+.2f}' for v in a.cf_diff_pp.quantile([.25, .75])) + ']'])
            a = w.loc[w.cohort.eq(cohort) & w.month.eq(month) & w.passed]
            weather_rows.append([cohort, month, len(a), f'{a.cf_diff_pp.mean():+.2f}',
                                 f'{a.cf_diff_pp.median():+.2f}'])
    medians = table(['집단', '월', '기준', '대상 수', '오전 차이 중앙값 (%p)',
                     '낮 CF 차이 중앙값 (%p)', '낮 CF 차이 25/75백분위 (%p)'], median_rows)
    weather_differences = table(['집단', '월', '지지 통과 대상 수', '기상 대응 CF 차이 평균 (%p)',
                                 '중앙값 (%p)'], weather_rows)
    rows = []
    for cohort, r in s['regression'].items():
        if r['valid']:
            rows.append([cohort, r['audit']['supported_sites'], r['audit']['stations'],
                         interval(r['beta_pp'][0], r['bootstrap']['7']['intervals_pp'][0]),
                         interval(r['beta_pp'][1], r['bootstrap']['7']['intervals_pp'][1])])
    slopes = table(['집단', '발전소', 'B 지점', '운량 10%p당 CF (%p), 7일 구간',
                    '습도 10%p당 CF (%p), 7일 구간'], rows)
    rows = []
    for cohort, r in s['contrasts'].items():
        for j, label in enumerate(['운량', '습도']):
            rows.append([cohort + ' - L80', label, interval(r['difference_pp'][j], r['bootstrap']['7']['intervals_pp'][j]),
                         '[' + ', '.join(f'{v:+.3f}' for v in r['bootstrap']['14']['intervals_pp'][j]) + ']'])
    differences = table(['비교', '변수', '계수 차이와 7일 구간 (%p)', '14일 구간 (%p)'], rows)
    naive = []
    panel = np.load(OUT / 'panel.npz')
    for cohort in COHORTS:
        take = panel['cohorts'] == cohort
        means = np.nanmean(panel['cf'][:, take, 2:10], axis=(0, 2)) * 100
        naive.append({'cohort': cohort, 'observed_time_mean_cf_percent_equal_site': float(means.mean())})
    dump(OUT / 'report_summary.json', {'naive': naive, 'gallery': gallery,
        'coefficient_comparison_rows': rows, 'profile_records': len(p), 'support_records': len(w)})
    prefix = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').split('## 결과')[0]
    prefix = prefix.replace('상태: DA-010 진행, 가용성 점검 완료. 아래 계약은 발전량 크기/형태와 관계 계산 전 고정했다.',
                            '상태: DA-010 실행과 수치 검산 완료. 아래 계약은 발전량 크기/형태와 관계 계산 전 고정했다.')
    body = f'''## 결과

### 비교 가능한 범위는 집단과 계절에 따라 다르다

80/68/105개소의 구분과 그룹 구성 차이를 원본으로 확인했다. S105의 2023년 기록은 없고, 2024년 초에는 비교할 수 있는 발전소가 적다. 9~12월도 월 14일 기준에서는 104개소만 남는다. 하반기 여섯 달 모두 같은 기준을 통과한 집단은 S105 중 92개소다. 따라서 105개소 전체의 연간 유형이나 연도 재현을 이번 결과로 주장할 수 없다.

2024년 선택 11시간 격자는 {s['audit']['grid_cells_2024']:,}셀이고 원본 키가 없는 셀은 {s['audit']['missing_source_keys']:,}개다. 존재한 기록의 음수/비유한 발전량은 {s['audit']['invalid_present_energy']}개, 유효 0은 {s['audit']['valid_zero_cells']:,}개였으며 CF>1은 {s['audit']['above_one_cells']}개였다. 없는 기록을 0발전으로 채우지 않았다.

### 같은 날짜의 비교에서도 형태와 수준 차이를 따로 봐야 한다

아래는 정확히 같은 날짜에서 대상과 L80 평균을 비교한 발전소별 차이를 월 안에서 동일 가중한 값이다. 중심은 정규화 곡선의 시간 중심이며 실제 기록 시각 오류를 뜻하지 않는다. 같은 B 열은 서로 다른 부분집합의 결과다.

{profile_table}

M68은 L80 전체와 비교하면 대부분의 월에서 낮 CF가 조금 높지만, 같은 B 지점의 L80 동료와 비교한 20개소에서는 대부분 낮았다. 대상 발전소와 기준도 함께 바뀌는 비교이므로 이를 기상 조정만으로 부호가 바뀐 결과로 받아들이면 안 된다. 발전소 구성과 비교 기준을 고정하지 않은 수준 순위가 쉽게 달라질 수 있다는 점을 보여 준다.

S105의 월별 전체 대응에서는 7월 +1.98%p, 10월 +2.73%p, 12월 +0.81%p의 낮 CF 차이가 남았다. 오전 비중 차이는 같은 세 달에서 -0.50/-0.42/+0.41%p로 방향이 일정하지 않았다. 대상 수가 92/104/104개소로 달라지는 월별 기술값이며, 고정 집단의 계절 효과나 기록 길이의 효과는 아니다.

{medians}

개별 발전소/월의 산포도 함께 확인했다. 평균 차이가 작거나 부호가 달라지는 월만으로 같은 유형 또는 다른 유형을 확정하지 않는다. 같은 B 지점 비교에서도 실제 설비 특성과 미측정 일사 차이는 남으며, 대상 집단 전체를 대표하지 않는다. 2024년 전체 관측 시간의 단순 평균은 기록 시기가 달라 별도 로컬 요약에만 보관하고 위 날짜 대응 비교와 섞지 않았다.

### 공동 기상 지지는 확보됐지만 완전한 조건 일치는 아니다

{support_table}

지지율은 집단별로 월 가용 기준을 통과한 낮 행을 분모로 계산했다. 회귀는 하반기 고정 발전소 중 월 지지 기준도 통과한 행만 사용하므로 표의 두 표본을 같은 것으로 보지 않는다. 자기 B 지점을 제외한 서로 다른 L80 지점이 둘 이상 허용차를 통과했다는 기준이며, 모든 변수의 전체 결합 분포가 같아졌다는 뜻은 아니다.

조건을 통과한 기준 지점의 동일 가중 CF와 비교한 발전소/월 차이는 다음과 같다. 달력 비교와는 기준 발전소/지점의 가중도 달라 순수 기상 조정 효과를 계산한 결과로 부르지 않는다. 같은 B 지점 검토는 M68 20개소/S105 30개소에 한정된 민감도다.

{weather_differences}

### 하반기의 운량/습도 관계 방향은 반복되지만 크기는 그대로 옮기기 어렵다

모든 집단이 사전 표본/수치 기준을 통과했다. 각 집단의 고유 날짜는 184일이며 지점 수와 유효 행 수는 다르다. 다음 수치는 선형 공동 식에서 다른 변수를 고려한 10%p당 CF 차이로 DA-003의 비선형 40→80% 등 특정 대비와 같지 않다.

{slopes}

세 집단의 운량과 습도 계수는 모두 음수였고 7일 및 14일 날짜 블록의 개별 구간에서도 같은 방향이었다. 이는 이번 하반기와 지지 범위에서의 방향 재현이다. 실제 기상 변화의 인과효과나 예보 입력에서의 관계까지 재현한 결과는 아니다.

{differences}

M68의 습도 계수는 L80보다 덜 음수였고 그 차이의 7/14일 기술 구간은 모두 0을 제외했다. S105 중 92개소의 습도 계수 차이와 두 대상 집단의 운량 계수 차이는 구간에 0을 포함했다. 전자는 집단 구성과 조건/위치가 섞인 관측 차이이며, 후자는 관계의 동등성을 입증하지 않는다. 다중 비교를 보정한 발견이나 새 발전소 모집단에 대한 검정으로 해석하지 않는다.

기온/풍속 등은 함께 고려할 통제 변수로 투입했으며 이 표에서 물리 효과의 주 결론으로 삼지 않았다. 양의 기온 계수를 발전 효율 향상의 증거로 쓰지 않는다.

## 검증과 실행 이력

정규 CPython 3.13.1에서 합성 검증 8개와 본 계산은 종료 코드 0으로 완료됐다. 불균형/빈 달과 결측 밖 행의 가중 고정효과를 명시적 더미 OLS와 비교했고 기상 허용차의 경계, 강수 분류와 월 14일 기준을 확인했다. 본 계산은 사전 계약으로 1회 수행했다.

원본 키로 발전량 {verification['source_generation_cells']:,}셀과 변환 기상 {verification['source_weather_cells']:,}셀을 다시 구성했다. 형태 비교 {verification['profile_comparisons']:,}건, 기상 지지 발전소/월 {verification['weather_site_months']:,}건과 모든 기준 지점 수/마스크를 대조했다. 고정효과 제거 뒤 직접 SVD 최소제곱으로 점 적합 3개와 두 블록의 최초 재표집 각 3개, 총 {verification['direct_svd_fits']}개를 검산했다. 점 계수 최대 차이는 {verification['max_point_beta_difference_pp']:.2e}%p였다. 저장된 400개 날짜 가중치와 모든 저장 구간/집단 차이를 확인했으며 전체 1,200개 재표집 적합을 독립 재실행한 것은 아니다.

검산기는 두 차례 준비 단계에서 중단됐다. 첫 시도는 사전 보고서 복사본의 LF/CRLF 차이였으며 두 표현의 해시만 허용해 동일 내용의 사전 기록을 확인했다. 다음 시도는 pandas 읽기 전용 배열의 풍속 단위 변환이었으며 복사 배열을 명시해 수정했다. 검산기의 오류를 고쳤고 본 분석 식, 표본, 임계값이나 재표집은 변경하지 않았다. 원본 세 파일과 본 계산 코드의 해시는 전후 및 재검산 때 동일했다.

그림 6개와 갤러리 링크 12개의 파일 검사를 완료했다. 절반 크기 검토 시트 두 장에서 모든 그림의 표본 수, 단위, 산포와 7/14일 구간을 직접 확인했다. 회귀 그림에서 S105 중 92개소라는 표기를 보완했으며 원본 크기의 전체 픽셀 검사를 뜻하지 않는다.

## 한계 및 다음 단계

이번 결과는 장기 관측 표본을 전체 발전소로 그대로 일반화하기 어렵다는 점과, 어느 조건에서 비교 범위를 넓힐 수 있는지를 보여 준다. 관측 길이와 현재 그룹, 지역/기상 지점 구성 및 설비 특성이 섞여 있으므로 차이를 기록 길이의 효과로 해석하지 않는다. 달력/허용차 대응도 실제 일사와 설비 특성을 통제하지 못한다.

형태는 같은 날짜의 연속 요약값을 비교했을 뿐 DA-002의 두 군집을 새 발전소에서 검증하거나 배정한 결과가 아니다. 2024년의 일부 관측으로 2023년 관계나 연간 시간 안정성을 채우지 않았다. A/B 공유 구조와 FA의 추가 기상 지점에 대한 재현도 이번 범위 밖이다.

| 기존 결론 | 이번에 넓혀 확인한 부분 | 유지할 제한 |
|---|---|---|
| DA-002의 곡선 형태와 두 유형 | 다른 집단도 같은 날짜에서 형태/수준의 연속 차이를 비교할 수 있다. | 두 유형 자체의 전수 재현이나 신규 배정은 미검증이다. S105의 연간/연도 재현도 주장하지 않는다. |
| DA-003의 기상 조건부 관계 | 하반기 B 실측의 제한된 선형 식에서 세 집단의 운량/습도 음의 방향을 확인했다. | M68의 습도 크기 차이가 남고, 기존 비선형 곡선과 예보 관계 전체를 전이한 결과가 아니다. |
| DA-004의 상대 수준 시간 구조 | 관측 기간과 비교 동료 범위의 제약을 구분했다. | 새 집단의 변화점과 지속 변화는 검증하지 않았다. |
| DA-001/005의 기상 구조 | 추가 발전소의 기상 지지와 표본 범위를 확인했다. | FA/CCA를 추가 지점에서 재현한 결과는 아니다. |

날짜 블록 구간은 고정한 발전소와 지점망에 한정한다. 7/14일보다 긴 의존성, 공간 모집단 불확실성과 사후 표본 선택의 영향은 모두 반영하지 못한다. 설명식은 지지 행에서도 선형 계수를 공유하므로 지역별 이질성과 비선형 잔여 관계가 남을 수 있다.

적용 범위와 지지, 형태/수준 및 제한된 조건부 관계를 확인해 DA-010의 정한 범위를 종료한다. 다음 권장 후보는 DA-007 공동 지지와 날짜 요인이다. 이번 분석의 느슨한 기상 허용차를 엄격한 공동 지지 검증의 대체물로 쓰지 않는다. DA-011은 그 뒤 별도로 검토한다. 예측 모델링과 공식 제출은 수행하지 않았다.

## 관련 코드와 로컬 결과

- [본 계산과 가용성 점검](../scripts/cohort_applicability.py)
- [합성 검증](../tests/test_cohort_applicability.py)
- [독립 원본/수치 검산](../scripts/verify_cohort_applicability.py)
- [보고서와 그림 생성](../scripts/report_cohort_applicability.py)
- [계산 전 고정 기록](../data/analysis/2026-10-21_cohort_applicability/preregistered_report.md)
- [전체 로컬 결과](../data/analysis/2026-10-21_cohort_applicability/summary.json)
- [독립 검산 결과](../data/analysis/2026-10-21_cohort_applicability/verification.json)
- [그림 갤러리](../data/analysis/2026-10-21_cohort_applicability/index.html)
- [데이터 분석 계획](데이터분석_진행계획.md)

원본/전처리 행은 보고서에 복사하지 않았다. 패널과 개별 결과/집계 CSV, 재표집과 그림은 무시되는 로컬 data/ 아래에만 보존한다.
'''
    REPORT.write_text(prefix + body, encoding='utf-8')
    print(json.dumps({'naive': naive, 'gallery': gallery,
        'monthly_means': p.groupby(['cohort', 'reference', 'month'])[['morning_diff_pp', 'cf_diff_pp']].mean().round(3).reset_index().to_dict('records'),
        'support': w.groupby(['cohort', 'month']).agg(eligible=('passed','size'), passed=('passed','sum'),
            fraction=('support_fraction','mean')).round(3).reset_index().to_dict('records')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
