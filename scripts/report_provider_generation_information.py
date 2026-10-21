"""Generate DA-011 interpretation and figures from saved explanatory results."""
import json

import numpy as np

from provider_generation_information import OUT, REPORT, PRIMARY, REPS
from weather_latent_structure import dump

LABELS = {'10_2024': '10시/2024', '10_2023': '10시/2023', '17_2024': '17시/2024', '17_2023': '17시/2023'}
VARIABLES = {'A_cloud': 'A 운량', 'B_cloud': 'B 운량', 'B_uv': 'B UV'}
EFFECTS = {'calendar': '발전소/월/시간', 'date_hour': '날짜/시간 추가'}
COLORS = ['#246b85', '#bc743d', '#6c9065', '#8a7196']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def interval(value, bounds, scale=1):
    if bounds is None:
        return f'{scale*value:+.3f} [구간 없음]'
    return f'{scale*value:+.3f} [{scale*bounds[0]:+.3f}, {scale*bounds[1]:+.3f}]'


def direction(case, effect, name):
    bounds = [case['effects'][effect]['bootstrap'][str(block)][name + '_beta_pp']['interval'] for block in [7, 14]]
    if any(v is None for v in bounds):
        return '구간 부족'
    if all(v[1] < 0 for v in bounds):
        return '음의 방향'
    if all(v[0] > 0 for v in bounds):
        return '양의 방향'
    return '불확실'


def partial_residuals(label, name):
    p = np.load(OUT / f'{label}_panel.npz')
    x = p['x'][:, p['site_pair']].copy()
    y = p['site_response'].copy()
    for month in np.unique(p['months']):
        take = p['months'] == month
        x[take] -= x[take].mean(axis=0)
        y[take] -= y[take].mean(axis=0)
    # Complete balanced original-site panel: exact two-way projection.
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    x = x.reshape(-1, 13)
    y = y.ravel()
    mat = np.load(OUT / f'{label}_cross.npz')['date_hour']
    col = PRIMARY[name]
    others = [j for j in range(13) if j != col]
    bx = np.linalg.solve(mat[np.ix_(others, others)], mat[others, col])
    by = np.linalg.solve(mat[np.ix_(others, others)], mat[others, -1])
    return x[:, col] - x[:, others] @ bx, y - x[:, others] @ by


def figures(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot, write_gallery, check_gallery
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    labels = list(s['cases'])
    gallery = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, effect in zip(axes, EFFECTS):
        for j, model in enumerate(['A', 'B', 'AB']):
            v = [100 * s['cases'][label]['effects'][effect]['metrics']['R2_' + model] for label in labels]
            ax.bar(np.arange(4) + (j - 1) * .23, v, width=.23, label=model, color=COLORS[j])
        ax.set(xticks=range(4), xticklabels=[LABELS[v] for v in labels], ylabel='고정효과 제거 CF 분산의 설명률 (%)',
               title=EFFECTS[effect], ylim=(0, 65))
        ax.legend()
    save_plot(fig, OUT, gallery, '01_block_explained_variation', '같은 표본에서 A만, B만, A+B의 표본 내 설명량',
              '개별 발전소 잔차를 보존했다. 두 패널은 고정효과와 분모가 달라 R² 감소를 기상 정보 손실로 해석하지 않는다.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, effect in zip(axes, EFFECTS):
        for offset, metric, title, color in [(-.12, 'delta_R2_A_given_B', 'B에 A 추가', COLORS[0]),
                                             (.12, 'delta_R2_B_given_A', 'A에 B 추가', COLORS[1])]:
            values = [100 * s['cases'][label]['effects'][effect]['metrics'][metric] for label in labels]
            ax.bar(np.arange(4) + offset, values, width=.24, color=color, label=title)
        ax.set(xticks=range(4), xticklabels=[LABELS[v] for v in labels], ylabel='추가 ΔR² (%p)', title=EFFECTS[effect])
        ax.legend()
    save_plot(fig, OUT, gallery, '02_conditional_block_increment', '상대 제공처를 고려한 뒤의 추가 표본 내 설명량',
              '양수라는 사실만으로 유용성이나 예측 개선을 입증하지 않는다. 변수 목록의 크기와 표현에도 의존한다.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, name in zip(axes, PRIMARY):
        for i, label in enumerate(labels):
            for offset, effect, color in [(-.1, 'calendar', COLORS[0]), (.1, 'date_hour', COLORS[1])]:
                r = s['cases'][label]['effects'][effect]
                key = name + '_beta_pp'
                point, bounds = r['metrics'][key], r['bootstrap']['7'][key]['interval']
                ax.hlines(i + offset, *bounds, color=color, linewidth=2)
                ax.plot(point, i + offset, 'o', color=color, label=EFFECTS[effect] if i == 0 else None)
        ax.axvline(0, color='#888', linewidth=.7)
        ax.set(yticks=range(4), yticklabels=[LABELS[v] for v in labels], title=VARIABLES[name],
               xlabel='CF 차이 (%p / ' + ('UV 1)' if name == 'B_uv' else '운량 10%p)'))
        ax.legend(fontsize=7)
    save_plot(fig, OUT, gallery, '03_primary_partial_coefficients', 'A+B 나머지 변수까지 고려한 주 변수의 부분계수',
              '7일 공동 블록의 개별 95% 구간이다. 14일 구간과 두 해/라운드 판정은 보고서에서 별도 대조한다.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label = '17_2024'
    for ax, name in zip(axes, PRIMARY):
        rx, ry = partial_residuals(label, name)
        multiplier = 1 if name == 'B_uv' else 10
        r = s['cases'][label]['effects']['date_hour']['metrics']
        expected_r = np.corrcoef(rx, ry)[0, 1]
        np.testing.assert_allclose(expected_r, r[name + '_partial_correlation'], atol=1e-10)
        ax.hexbin(rx * multiplier, ry * 100, gridsize=45, mincnt=1, bins='log', cmap='Blues', linewidths=0)
        left, right = ax.get_xlim()
        beta = r[name + '_beta_pp'] / multiplier
        ax.plot([left, right], [left * beta, right * beta], color='#ba6538', linewidth=1.8)
        ax.axvline(0, color='#888', linewidth=.5)
        ax.axhline(0, color='#888', linewidth=.5)
        ax.set(title=f'{VARIABLES[name]}, 부분상관 {expected_r:+.3f}', ylabel='발전량 CF 잔여 (%p)',
               xlabel=VARIABLES[name] + (' 잔여 (지수)' if name == 'B_uv' else ' 잔여 (%p)'))
    save_plot(fig, OUT, gallery, '04_partial_residual_example', '2024년 17시 예보의 날짜/시간 조정 부분회귀 예시',
              '설명 변수와 발전량 양쪽에서 같은 나머지 변수를 제거했다. 밀도는 원래 80개소 행이며 독립 관측 수를 뜻하지 않는다.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, PRIMARY):
        all_values = []
        for j, effect in enumerate(EFFECTS):
            x, y = [], []
            for label in labels:
                r = s['cases'][label]['effects'][effect]
                x.append(r['metrics'][name + '_beta_pp'])
                y.append(r['matching']['metrics'][name + '_beta_pp'])
            ax.scatter(x, y, color=COLORS[j], marker='o' if j == 0 else 's', label=EFFECTS[effect])
            all_values += x + y
        lo, hi = min(all_values), max(all_values)
        gap = max((hi - lo) * .1, .02)
        ax.plot([lo - gap, hi + gap], [lo - gap, hi + gap], '--', color='#777', linewidth=.8)
        ax.set(title=VARIABLES[name], xlabel='전체 29쌍 부분계수', ylabel='고유 지점 매칭 22쌍 부분계수')
        ax.legend(fontsize=7)
    save_plot(fig, OUT, gallery, '05_matching_sensitivity', 'A/B 지점을 한 번씩 사용한 매칭 부분집합의 점 민감도',
              '고유 22쌍 안 발전소 빈도 가중을 유지했다. 지점 반복과 구성이 함께 바뀌며 독립 지역 검증은 아니다.')

    fig, ax = plt.subplots(figsize=(11, 5.3))
    states = ['A0_B0', 'A1_B0', 'A0_B1', 'A1_B1']
    values = np.array([[s['cases'][label]['rain_support']['states'][state]['pair_hour_rows'] for state in states] for label in labels])
    percentages = 100 * values / values.sum(axis=1, keepdims=True)
    im = ax.imshow(percentages, aspect='auto', cmap='Blues', vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{percentages[i,j]:.1f}%\n{values[i,j]:,}', ha='center', va='center',
                    color='white' if percentages[i,j] > 60 else 'black')
    ax.set(yticks=range(4), yticklabels=[LABELS[v] for v in labels], xticks=range(4),
           xticklabels=['둘 다 비강수', 'A만 강수', 'B만 강수', '둘 다 강수'])
    fig.colorbar(im, ax=ax, label='고유 쌍/시각 행 비율 (%)')
    save_plot(fig, OUT, gallery, '06_rain_disagreement', '예보 강수 여부의 일치와 불일치',
              '기준은 0.1mm 초과다. 공유 지점 반복과 시간 의존성이 있으며 불일치 자체가 정확도나 오류를 뜻하지 않는다.')
    write_gallery(OUT, gallery, 'DA-011 A/B 잔여 변동과 발전량')
    return check_gallery(OUT, 6)


def main():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    v = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    gallery = figures(s)
    def metric_range(metric, effect='date_hour', scale=100, digits=2):
        values = [scale * case['effects'][effect]['metrics'][metric] for case in s['cases'].values()]
        return f'{min(values):.{digits}f}~{max(values):.{digits}f}'
    audit_rows, provenance_rows, block_rows, primary_rows, residual_rows, rain_rows = [], [], [], [], [], []
    for label, case in s['cases'].items():
        a = case['audit']
        audit_rows.append([LABELS[label], a['dates'], a['excluded_dates'], f"{a['site_hour_rows']:,}", a['valid_zero_cf']])
        provenance_rows.append([LABELS[label], a['providers']['wa']['missing_pair_hour_keys'],
            a['providers']['wb']['missing_pair_hour_keys'],
            sum(a['providers']['wa']['invalid_unique_station_cells'].values()),
            sum(a['providers']['wb']['invalid_unique_station_cells'].values()),
            a['providers']['wa']['nonhour_rows'] + a['providers']['wb']['nonhour_rows']])
        for effect, r in case['effects'].items():
            m = r['metrics']
            block_rows.append([LABELS[label], EFFECTS[effect], *[f"{100*m['R2_'+k]:.2f}" for k in ['A','B','AB']],
                f"{100*m['delta_R2_A_given_B']:.2f}", f"{100*m['delta_R2_B_given_A']:.2f}",
                f"{100*m['partial_R2_A_given_B']:.2f}", f"{100*m['partial_R2_B_given_A']:.2f}"])
            for name in PRIMARY:
                key = name + '_beta_pp'
                primary_rows.append([LABELS[label], EFFECTS[effect], VARIABLES[name],
                    interval(m[key], r['bootstrap']['7'][key]['interval']),
                    '[' + ', '.join(f'{x:+.3f}' for x in r['bootstrap']['14'][key]['interval']) + ']', direction(case, effect, name)])
                if effect == 'date_hour':
                    residual_rows.append([LABELS[label], VARIABLES[name], f"{100*m[name+'_remaining_weather_fraction']:.1f}",
                        f"{m[name+'_remaining_weather_sd']:.2f}", f"{m[name+'_partial_correlation']:+.3f}",
                        f"{100*m[name+'_delta_R2']:.3f}", f"{100*m[name+'_partial_R2']:.3f}",
                        f"{m[name+'_opposite_beta_pp']:+.3f}", f"{m[key]:+.3f}"])
        for provider in ['A', 'B']:
            rain = case['rain_support'][provider]
            rain_rows.append([LABELS[label], provider, rain['disagreement_rows'], rain['disagreement_dates'],
                rain['supported_cells'], '통과' if rain['eligible'] else '부족',
                f"{100*case['effects']['calendar']['metrics'].get(provider+'_rain_delta_R2',np.nan):.4f}",
                f"{100*case['effects']['date_hour']['metrics'].get(provider+'_rain_delta_R2',np.nan):.4f}"])
    audit_table = table(['자료', '유효 날짜', '제외 날짜', '발전소/시간 행', '유효 0'], audit_rows)
    provenance = table(['자료', '없는 A 쌍/시각 키', '없는 B 쌍/시각 키', 'A 유효 범위 밖 셀',
                        'B 유효 범위 밖 셀', '비정시 행'], provenance_rows)
    blocks = table(['자료', '통제', 'A R² (%)', 'B R² (%)', 'A+B R² (%)', 'A|B ΔR² (%p)',
                    'B|A ΔR² (%p)', 'A|B 부분 R² (%)', 'B|A 부분 R² (%)'], block_rows)
    primary = table(['자료', '통제', '변수', '부분계수, 7일 구간', '14일 구간', '두 블록 판단'], primary_rows)
    residuals = table(['자료', '변수', '잔여 기상 분산 (%)', '잔여 기상 SD', '부분상관', 'ΔR² (%p)',
                       '부분 R² (%)', '반대쪽만 통제 계수', 'A+B 나머지 통제 계수'], residual_rows)
    rain_table = table(['자료', '추가 강수', '자기만 강수 행', '날짜', '지지 셀', '보조 기준',
                        '기본 ΔR² (%p)', '날짜 추가 ΔR² (%p)'], rain_rows)
    all_success = all(r['success'] == REPS for case in s['cases'].values() for effect in case['effects'].values()
                      for block in effect['bootstrap'].values() for r in block.values())
    states = {name: {effect: sorted(set(direction(case, effect, name) for case in s['cases'].values()))
                    for effect in EFFECTS} for name in PRIMARY}
    largest_match_change = {name: max(abs(case['effects'][effect]['metrics'][name+'_beta_pp'] -
        case['effects'][effect]['matching']['metrics'][name+'_beta_pp']) for case in s['cases'].values() for effect in EFFECTS)
                           for name in PRIMARY}
    p = np.load(OUT / '17_2024_panel.npz')
    matching_sites = int(p['matching'][p['site_pair']].sum())
    dump(OUT / 'report_verification.json', {'gallery': gallery, 'direction': states,
        'all_bootstrap_metrics_success': all_success, 'max_matching_beta_changes_pp': largest_match_change,
        'matching_sites': matching_sites})
    assert all_success
    prefix = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').split('## 결과')[0]
    prefix = prefix.replace('상태: DA-011 진행. 아래 범위와 기준은 새 결과 계산 전에 고정했다.',
                            '상태: DA-011 계산, 독립 검산과 그림 검토 완료. 아래 범위와 기준은 새 결과 계산 전에 고정했다.')
    body = f'''## 결과

### 기상 자료만 분석한 표본과 발전량을 연결한 표본은 다르다

같은 80개 발전소의 고유 A/B 쌍은 {s['pairs']}개이며 A {s['a_stations']}/B {s['b_stations']}지점이다. DA-005는 148개소에 연결된 52쌍의 기상 자체를 분석했으므로 두 결과를 같은 표본으로 합치지 않는다.

{audit_table}

{provenance}

2023년에는 두 시각 각각 하루가 기상 키 부족으로 제외됐다. 2024년은 10시 366일, 17시 365일을 사용했다. 17시에는 선택한 A 습도 한 셀이 분석용 0~100 범위를 벗어나 해당 날짜를 균형 표본에서 제외했다. 이는 원본의 최종 삭제/보정 판단이 아니다. 비정시 행과 발전량 키 누락은 없었으며 정적 용량 대비 CF>1도 없었다. 유효 0은 유지했다. 없는 키와 유효 범위 밖 셀은 관측 단위가 다르므로 단순 합산하지 않는다. 제외일이 다른 두 시각의 차이를 정확한 예보 갱신 효과로 해석하지 않는다.

### A/B는 서로를 고려한 뒤에도 표본 내 추가 설명량이 남았다

다음 R²의 분모는 원래 발전소 CF에서 해당 고정효과를 뺀 제곱합이다. 발전소 평균 곡선이나 전체 원시 발전량의 설명률, 예측 점수가 아니다. ΔR²는 이 분모에 대한 증가분이고 부분 R²는 상대 제공처 식에 남은 잔차를 분모로 삼는다.

{blocks}

날짜/시간을 추가한 식에서는 B에 A를 더한 ΔR²가 약 {metric_range('delta_R2_A_given_B')}%p, A에 B를 더한 값은 약 {metric_range('delta_R2_B_given_A')}%p였다. 기본 통제 식에서는 각각 약 {metric_range('delta_R2_A_given_B', 'calendar')}%p와 {metric_range('delta_R2_B_given_A', 'calendar')}%p였다. 이는 같은 표본의 선형 설명량 비교이며 A가 더 정확한 제공처라는 판정이 아니다. 변수 수와 표현도 다르고 날짜 추가 시 잔여 발전량의 분모도 달라진다.

양쪽 모두 추가 변수로 표본 내 설명량이 늘었다는 사실만으로 독립적인 실제 정보나 미래 예측 개선을 확정하지 않는다. 아래 주 변수의 방향/불확실성과 기간 민감도를 함께 본다.

### A 운량과 B UV의 잔여 관계는 반복됐고 B 운량은 조건에 따라 불확실했다

운량의 단위는 10%p당 CF %p, B UV는 지수 1당 CF %p다. 모든 값은 상대 제공처뿐 아니라 같은 제공처의 나머지 변수까지 고려한 A+B 식의 부분계수다.

{primary}

A 운량은 두 해/두 시각에서 기본과 날짜 추가 식 모두 음수였고 7/14일 구간이 모두 0을 제외했다. 기본 식의 계수는 약 {metric_range('A_cloud_beta_pp', 'calendar', 1)}%p, 날짜 추가 후에는 약 {metric_range('A_cloud_beta_pp', scale=1)}%p였다. B UV는 같은 모든 조건에서 양수로 반복됐지만 기본 식의 크기는 2023년 약 1.69~1.74%p에서 2024년 약 0.60~0.62%p로 달랐다. 날짜 추가 후에는 약 {metric_range('B_uv_beta_pp', scale=1)}%p였다. 모든 연도/설명식에서 같은 크기의 관계라고 보지 않는다.

B 운량은 2023년 17시의 기본 식에서 두 블록 구간이 0을 포함했다. 날짜 추가 후에는 두 해/두 시각 모두 음의 방향이었지만, 이를 조정 전의 불확실성을 없앤 단일 정답으로 취급하지 않는다. 날짜 추가 전후는 비교에 쓰는 변동도 달라진다.

### 남는 기상 변동의 크기와 발전량의 추가 설명량은 별개다

다음 표는 날짜/시간 추가 식이다. 잔여 기상 분산은 고정효과 제거 후 해당 변수의 변동 중 A+B의 나머지 열로 설명되지 않는 비율이다. SD의 단위는 운량 %p, B UV 지수이며 선택 행 수로 나눈 잔여 RMS다. 고정효과 제거 때문에 가중 평균은 0이다.

{residuals}

날짜 추가 후 B UV에는 약 {metric_range('B_uv_remaining_weather_fraction', digits=0)}%의 조건부 잔여 기상 분산이 남았지만, 이 열 하나를 추가한 발전량 ΔR²는 약 {metric_range('B_uv_delta_R2')}%p였다. 부분상관도 약 +0.04 수준이다. A 운량의 ΔR²는 약 {metric_range('A_cloud_delta_R2')}%p, B 운량은 약 {metric_range('B_cloud_delta_R2')}%p였다. 큰 기상 잔여 분산을 곧바로 큰 발전량 설명력으로 읽으면 안 된다.

반대 제공처만 고려한 계수와 A+B 나머지 열까지 고려한 계수는 크기가 달랐다. 예를 들어 B UV의 양의 관계 일부는 같은 B 제공처의 다른 날씨와도 연결돼 있다. 이것을 비음수의 고유/공유 정보 기여율로 나누거나 물리적 원인을 확정하지 않는다. DA-005의 잔여 비율과는 표본, 변환과 통제 범위가 달라 직접 전후 비교하지 않는다.

### 강수 불일치는 관측되지만 추가 선형 설명량은 작았다

모든 자료/연도에서 양쪽 강수의 보조 가용성 기준을 통과했다. 이 기준은 같은 셀에서 상대 제공처 강수 상태만 고정했으며 운량/습도 등 모든 날씨의 엄격한 공동 지지를 확인한 기준은 아니다. DA-007의 부족 판단을 뒤집지 않는다.

{rain_table}

A/B 불일치 자체는 어느 제공처가 틀렸다는 뜻이 아니다. A 강수 두 열의 날짜 추가 ΔR²는 약 {metric_range('A_rain_delta_R2')}%p, B 강수 두 열은 약 {metric_range('B_rain_delta_R2', digits=3)}%p였다. 강수의 0 집중과 정의/위치 차이를 고려해야 하며, 작은 B 강수의 추가 설명량을 강수가 발전량에 영향을 주지 않는다는 증거로 해석하지 않는다. 강수 조건별 인과효과나 예보 정확도를 추정하지 않았다.

### 공유 지점을 한 번씩 남긴 부분집합에서도 주 방향은 같았다

최대 이분 매칭은 {s['matching_pairs']}쌍, 해당 발전소 {matching_sites}개소를 남겼다. 이 집합에서도 A/B 운량의 점 계수는 음수, B UV는 양수였다. 전체와의 최대 점 계수 차이는 A 운량 {largest_match_change['A_cloud']:.3f}%p, B 운량 {largest_match_change['B_cloud']:.3f}%p, B UV {largest_match_change['B_uv']:.3f}%p였다. 별도 구간은 계산하지 않아 재표집 방향까지 확인한 결과는 아니다. 지점 반복과 표본 구성이 함께 바뀐 민감도이며 새로운 지역의 독립 검증은 아니다.

## 검증과 실행 이력

정규 CPython 3.13.1의 승격 Start-Process로 실행했다. 합성 검증 8개는 최종 통과했다. 최초에는 절편 없는 합성 자료의 잔여 RMS를 평균 제거 표준편차와 비교한 기대값이 한 검증에서 달랐다. 비교량을 잔여 제곱평균의 제곱근으로 바로잡았고 계산 코드는 변경하지 않았다. 실제 자료 계산 전 수정이며 분석 범위/기준을 변경하지 않았다.

본 계산은 1회로 완료됐다. 모든 강수 보조 기준이 통과해 설명식은 A/B/A+B, 세 주 변수의 제외/반대쪽 추가 식과 두 강수 제외 식 등 11개다. 네 자료의 두 고정효과에서 점 적합 88개, 매칭 점 적합 88개를 수행했다. 두 블록 각 100회의 재표집에서 총 17,600개 식 적합에 해당하는 설명량/주 관계를 계산했고 모든 저장 지표의 재표집 100회씩이 유효했다. 이는 17,600개의 서로 독립된 분석이 아니다.

독립 검산에서 원본 키의 발전량 {v['source_generation_cells']:,}셀과 변환 기상 {v['transformed_weather_cells']:,}셀을 재구성하고 제외일/표본을 대조했다. 강수 보조 조건 8건을 별도 집계로 확인했으며 다른 증가 경로 알고리즘으로 최대 매칭 크기를 검증했다. 원래 발전소 자료의 명시적 더미 최소제곱으로 점 교차곱 16개와 두 블록 최초 표본 교차곱 16개를 대조했다. 모든 점 적합 176개와 양쪽 잔여화 48건의 계수, 원래 발전소 잔차를 분모로 한 설명량을 확인했다. 원래 발전소와 압축 계산의 교차곱 최대 차이는 {v['max_original_site_cross_difference']:.2e}였다.

날짜 가중치 800개와 모든 저장 구간을 확인했다. 전체 재표집 설명식을 독립 재실행한 것은 아니다. 원본 네 파일과 본 계산 및 의존 코드의 해시는 전후 및 검산 때 동일했고 사전 기록 내용도 줄바꿈 차이를 구분해 확인했다. 본 계산과 독립 검산은 종료 코드 0이었다.

그림 6개와 갤러리 링크 12개를 검사했다. 부분회귀 예시 그림의 잔여 상관은 저장된 부분상관과 일치시켰다. 절반 크기의 검토 시트 두 장에서 모든 그림의 단위, 서로 다른 분모, 구간과 매칭/강수 표기를 직접 확인했다. 원본 크기의 전체 픽셀 검사를 뜻하지 않는다. 보고서의 날짜 제외 사유와 요약 범위도 저장된 집계와 대조해 보완했다.

## 한계 및 다음 단계

이번 결론은 A/B 기상 차이 중 일부가 다른 관측 변수와 달력/날짜 요인을 고려한 발전량 잔여와도 관련된다는 것이다. 특히 A 운량과 B UV의 방향이 반복됐지만 추가 설명량과 크기 안정성에는 차이가 있었다. 표본 내 ΔR²와 부호 재현은 미래 예측 효용, 정확도 순위나 인과적 정보 분해를 뜻하지 않는다.

전체 표본의 선형 조건부 관계이며 큰 대비의 엄격한 공동 지지는 이번에도 확보했다고 주장하지 않는다. 비선형 공유, 측정 오차와 억제 관계, 미측정 실제 일사/설비/운영 조건 및 지점 구성 차이가 남는다. 80개소의 결론을 전체 253개소로 확대하지 않는다. 두 해/두 시각은 이미 개발에 사용한 자료이며 각각의 설명식을 독립적인 외부 재현으로 세지 않는다.

원래 발전소 빈도 가중과 공동 날짜 블록은 고정 지점망의 조건부 비교를 유지할 뿐 새 지점 모집단의 불확실성을 포괄하지 않는다. 날짜 추가 식의 R² 분모도 달라 두 식의 설명률 차이를 기상 정보량 감소로 해석하지 않는다.

양방향 설명량과 주 잔여 관계, 강수 가용성 및 사전 민감도/검산을 확인해 DA-011을 종료한다. 종합 검토에서 권장한 DA-010 → DA-007 → DA-011 세 범위는 완료했다. 다음 선택 후보로 DA-012 발전소 유형별 기상 반응 차이를 검토할 수 있고, DA-009는 변화일을 실제 활용하려 할 때 우선 재검토한다. 후보를 자동 필수화하거나 예측 모델링을 재개하지 않는다.

## 관련 코드와 로컬 결과

- [계산](../scripts/provider_generation_information.py)
- [합성 검증](../tests/test_provider_generation_information.py)
- [독립 원본/수치 검산](../scripts/verify_provider_generation_information.py)
- [보고서와 그림 생성](../scripts/report_provider_generation_information.py)
- [계산 전 고정 보고서](../data/analysis/2026-10-21_provider_generation_information/preregistered_report.md)
- [전체 로컬 결과](../data/analysis/2026-10-21_provider_generation_information/summary.json)
- [독립 검산 결과](../data/analysis/2026-10-21_provider_generation_information/verification.json)
- [그림 갤러리](../data/analysis/2026-10-21_provider_generation_information/index.html)
- [데이터 분석 계획](데이터분석_진행계획.md)

원본/전처리 행은 보고서에 복사하지 않았다. 개별 패널과 집계/재표집, 지점 쌍 CSV 및 그림은 무시되는 로컬 data/ 아래에만 보존한다.
'''
    REPORT.write_text(prefix + body, encoding='utf-8')
    print(json.dumps({'gallery': gallery, 'direction': states, 'max_matching_change': largest_match_change,
                      'matching_sites': matching_sites, 'all_bootstrap_success': all_success}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
