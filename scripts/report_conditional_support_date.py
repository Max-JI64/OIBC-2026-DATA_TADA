"""Report saved DA-007 calculations without changing support or fitting models."""
import json

import numpy as np

from conditional_support_date import OUT, REPORT, SPECS, REPS
from weather_latent_structure import dump

SAMPLE_NAMES = {'cloud_wide': '운량 25→75', 'humidity_wide': '습도 40→80',
                'cloud_local': '운량 40→60', 'humidity_local': '습도 50→70'}
CASE_NAMES = {'observed_2024': '실측 2024', 'observed_2023': '실측 2023',
              'forecast_2024': '예보 2024', 'forecast_2023': '예보 2023'}
COLORS = ['#246b85', '#bc743d', '#6c9065', '#8a7196']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def cell(point, bounds):
    if point is None or bounds is None:
        return '추정 불가'
    return f'{point:+.3f} [{bounds[0]:+.3f}, {bounds[1]:+.3f}]'


def make_figures(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot, write_gallery, check_gallery
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    gallery = []
    labels = list(s['cases'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, key, title in zip(axes, ['site_hour_rows', 'calendar_cells'],
                              ['공동 지지 발전소 가중 행 수', '살아 있는 지점/월/시간 셀 수']):
        values = np.array([[s['cases'][case]['samples'][sample]['support'][key] for sample in SPECS] for case in labels])
        im = ax.imshow(values, aspect='auto', cmap='Blues', vmin=0)
        for i in range(len(labels)):
            for j in range(len(SPECS)):
                ax.text(j, i, str(values[i, j]), ha='center', va='center',
                        color='white' if values[i, j] > .65 * max(values.max(), 1) else 'black')
        ax.set(yticks=range(len(labels)), yticklabels=[CASE_NAMES[v] for v in labels],
               xticks=range(4), xticklabels=list(SAMPLE_NAMES.values()), title=title)
        ax.tick_params(axis='x', labelrotation=20)
        fig.colorbar(im, ax=ax, shrink=.8)
    save_plot(fig, OUT, gallery, '01_joint_support', '같은 달력 셀 안에서 양 끝의 다른 기상까지 맞춘 지지',
              '모든 대비가 사전 해석 기준을 미통과했다. 행 1,000개, 셀 20개 외 지점 5곳과 날짜 60일도 요구한다.')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, kind in enumerate(['observed', 'forecast']):
        for col in range(2):
            ax = axes[row, col]
            for k, year in enumerate([2024, 2023]):
                r = s['cases'][f'{kind}_{year}']['samples']['all']
                for shift, effect, color in [(-.1, 'calendar', '#246b85'), (.1, 'date_hour', '#bc743d')]:
                    value = r['fits'][effect]['beta_pp'][col]
                    bounds = r['fits'][effect]['bootstrap']['7']['intervals_pp'][col]
                    ax.hlines(k + shift, *bounds, color=color, linewidth=2)
                    ax.plot(value, k + shift, 'o', color=color, label=effect if k == 0 else None)
            ax.axvline(0, color='#888', linewidth=.7)
            ax.set(yticks=[0, 1], yticklabels=['2024', '2023'], xlabel='10%p당 CF 차이 (%p)',
                   title=('실측' if kind == 'observed' else '17시 예보') + ': ' + ['운량', '습도'][col])
            ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, '02_date_sensitivity', '같은 전체 표본과 선형 식의 날짜 조정 전후',
              '구간은 7일 공동 날짜 블록의 개별 95% 백분위다. 날짜×시간을 추가하면 비교에 쓰는 변동이 달라진다.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, ax in enumerate(axes):
        values = [100 * s['cases'][label]['samples']['all']['remaining_variance_ratio'][col] for label in labels]
        ax.bar(range(4), values, color=COLORS)
        for i, value in enumerate(values):
            ax.text(i, value + 1, f'{value:.1f}%', ha='center')
        ax.set(xticks=range(4), xticklabels=[CASE_NAMES[v] for v in labels], ylim=(0, 100),
               ylabel='추가 조정 뒤 남는 변동 (%)', title=['운량', '습도'][col])
    save_plot(fig, OUT, gallery, '03_remaining_variation', '날짜와 시간의 공통 변동을 뺀 뒤 남는 설명 변수 변동',
              '분모는 발전소/월/시간 조정 후 변동이다. 남는 분산 비율은 인과 식별 강도나 예측 정보량의 비율이 아니다.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for col, ax in enumerate(axes):
        for i, label in enumerate(labels):
            r = s['cases'][label]['samples']['all']
            point = r['fits']['date_hour']['beta_pp'][col] - r['fits']['calendar']['beta_pp'][col]
            for shift, block, marker in [(-.07, '7', 'o'), (.07, '14', 's')]:
                bound = r['date_difference'][block]['intervals_pp'][col]
                ax.hlines(i + shift, *bound, color=COLORS[i], linewidth=2)
                ax.plot(point, i + shift, marker, color=COLORS[i], markersize=5)
        ax.axvline(0, color='#888', linewidth=.7)
        ax.set(yticks=range(4), yticklabels=[CASE_NAMES[v] for v in labels],
               xlabel='날짜 추가 - 기본 계수 (%p / 10%p)', title=['운량', '습도'][col])
    save_plot(fig, OUT, gallery, '04_paired_changes', '날짜 추가에 따른 계수 차이의 짝지은 재표집',
              '원은 7일, 사각은 14일 블록의 개별 구간이다. 차이는 같은 인과효과의 편향 제거량이 아니다.')
    write_gallery(OUT, gallery, 'DA-007 공동 지지와 날짜 요인')
    return check_gallery(OUT, 4)


def main():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    verify = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    gallery = make_figures(s)
    support_rows, slope_rows, difference_rows = [], [], []
    for label, case in s['cases'].items():
        for sample in SPECS:
            a = case['samples'][sample]['support']
            support_rows.append([CASE_NAMES[label], SAMPLE_NAMES[sample], a['stations'], a['dates'],
                                 a['calendar_cells'], a['station_hour_rows'], a['site_hour_rows']])
        r = case['samples']['all']
        for col, variable in enumerate(['운량', '습도']):
            base, date = r['fits']['calendar'], r['fits']['date_hour']
            slope_rows.append([CASE_NAMES[label], variable,
                cell(base['beta_pp'][col], base['bootstrap']['7']['intervals_pp'][col]),
                cell(date['beta_pp'][col], date['bootstrap']['7']['intervals_pp'][col]),
                '[' + ', '.join(f'{v:+.3f}' for v in date['bootstrap']['14']['intervals_pp'][col]) + ']',
                f"{100 * r['remaining_variance_ratio'][col]:.1f}"])
            difference_rows.append([CASE_NAMES[label], variable,
                cell(date['beta_pp'][col] - base['beta_pp'][col], r['date_difference']['7']['intervals_pp'][col]),
                '[' + ', '.join(f'{v:+.3f}' for v in r['date_difference']['14']['intervals_pp'][col]) + ']'])
    support_table = table(['자료', '노출 대비 (%)', '지점', '날짜', '달력 셀', '고유 지점/시간 행', '발전소 가중 행'], support_rows)
    slope_table = table(['자료', '변수', '기본, 7일 구간', '날짜/시간 추가, 7일 구간',
                         '날짜/시간 추가, 14일 구간', '남는 분산 (%)'], slope_rows)
    diff_table = table(['자료', '변수', '계수 차이, 7일 구간', '14일 구간'], difference_rows)
    bootstrap_success = {label: {effect: case['samples']['all']['fits'][effect]['bootstrap']
                                  for effect in ['calendar', 'date_hour']} for label, case in s['cases'].items()}
    assert all(v == REPS for case in bootstrap_success.values() for effect in case.values()
               for block in effect.values() for v in block['success_by_term'])
    status = {}
    for kind in ['observed', 'forecast']:
        status[kind] = {}
        for col, variable in enumerate(['cloud', 'humidity']):
            bounds = [s['cases'][f'{kind}_{year}']['samples']['all']['fits']['date_hour']['bootstrap'][block]['intervals_pp'][col]
                      for year in [2024, 2023] for block in ['7', '14']]
            status[kind][variable] = 'negative in both years/blocks' if all(b[1] < 0 for b in bounds) else 'uncertain'
    dump(OUT / 'report_verification.json', {'gallery': gallery, 'direction_status': status,
                                          'all_bootstrap_terms_success': True})
    prefix = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').split('## 결과')[0]
    prefix = prefix.replace('상태: DA-007 진행. 아래 범위와 판정은 새 계산 전에 고정했다.',
                            '상태: DA-007 계산, 독립 검산과 그림 검토 완료. 아래 범위와 판정은 새 계산 전에 고정했다.')
    body = f'''## 결과

### 같은 달력 셀의 엄격한 공동 지지는 사전 표본 기준에 못 미쳤다

네 자료/연도의 원본을 재구성해 DA-003 패널과 일치시켰다. 실측/예보의 연도별 원래 발전소 행은 2024년 234,240개, 2023년 233,600개다. 이는 각각 같은 발전량을 사용한 두 설명 분석이며 독립적인 네 발전량 자료로 합산하지 않는다.

나머지 기상 조건을 함께 맞추고 반대 끝 날짜 두 개 이상이 연결되는 행을 남기자, 넓은 대비와 제한 대비 모두 사전 해석 기준을 통과하지 못했다.

{support_table}

이에 따라 공동 지지로 제한한 16개 표본에서는 설명 계수를 적합하지 않았다. 지점 5곳/날짜 60일/달력 셀 20개/발전소 가중 행 1,000개의 공동 조건 중 하나 이상이 부족했다. 습도 40→80%는 네 자료 중 세 자료에서 남는 셀이 없었다. 2023년 예보에서도 7개 셀, 25날짜에 한정됐다.

이 결과는 이번에 정한 같은 달력 셀과 기상 허용차에서 근거가 부족하다는 뜻이다. 해당 날씨가 물리적으로 불가능하다거나 원래 자료에 습도/운량 변동이 없다는 뜻은 아니다. 제한 대비는 다른 노출 구간과 더 좁은 양 끝 범위를 쓰므로 넓은 대비 표본의 상위 집합이 아니며, 지지가 더 많아야 할 이유도 없다. 좋은 결과를 얻기 위해 사후에 허용차나 셀 기준을 완화하지 않았다.

DA-003에서 풀링한 근방 표본은 충분해 보였어도, 같은 지점/월/시간 안의 여러 기상 조건을 동시에 맞춘 넓은 대비는 별도의 근거가 필요하다. 따라서 기존 큰 습도 대비를 그만큼의 독립적인 실제 습도 효과로 해석하는 데 제약이 확인됐다.

### 전체 표본에서는 날짜 조정 뒤에도 선형 음의 관계가 남았다

엄격한 공동 지지 표본을 만들 수 없었으므로 다음 날짜 민감도는 전체 표본에서의 별도 결과다. 이 표를 공동 지지가 충분한 표본의 추정치로 읽으면 안 된다. 같은 선형 식과 같은 행에서 발전소/월/시간 고정효과만 둔 결과와 날짜/시간 고정효과를 추가한 결과를 비교했다.

단위는 기상 변수 10%p당 CF의 %p 차이다. 구간은 개별 기술 구간이며 열의 7/14일은 공동 날짜 블록 길이다. 남는 분산은 날짜 추가 후 설명 변수의 잔여 제곱합을 기본 조정 후 제곱합으로 나눈 값이다.

{slope_table}

실측과 17시 예보에서 운량/습도의 날짜 추가 후 계수는 두 해 모두 음수였고, 7/14일 구간 모두 0을 제외했다. 실측의 운량 계수는 날짜 추가 후 2024년 약 -1.20%p, 2023년 -0.98%p였다. 습도는 약 -2.45/-2.23%p였다. 예보에서는 운량 약 -0.52/-0.46%p, 습도 약 -4.73/-4.19%p였다.

두 변수의 남는 분산은 기본 고정효과 조정 후의 약 17~27%였다. 전국적으로 함께 변하는 날짜/시간 성분을 제거한 뒤에도 음의 관계가 남지만, 비교가 주로 같은 날짜/시간 안의 지점 차이에 의존한다. 측정 위치/정의 차이와 미측정 실제 날씨 등은 여전히 관계에 들어갈 수 있다.

### 계수의 크기 변화와 해석 대상의 변화를 함께 보아야 한다

{diff_table}

날짜 추가 후 계수에서 기본 계수를 뺀 차이는 운량/습도에서 모두 양수였고 7/14일 구간 모두 0을 제외했다. 두 종류 자료에서 음의 관계 절댓값이 작아지는 모습이 두 해에 반복됐다. 이는 같은 표본에서 날짜 공통 변동의 처리에 따라 선형 관계 크기가 달라진다는 결과다. 그 차이를 미측정 교란의 정확한 제거량이나 동일한 인과효과의 편향 크기로 해석하지 않는다.

DA-003의 특정 기준점에서 계산한 비선형 운량/습도 대비와 이번 선형 계수는 같은 수치가 아니다. 이번 예보 운량 선형 계수가 음수라는 이유로 앞선 특정 비선형 대비의 불확실 결론을 대체하지 않는다. 모형과 비교량을 정한 범위별로 유지한다.

## 검증과 실행 이력

정규 CPython 3.13.1의 승격 Start-Process로 실행했다. 합성 검증 첫 시도에서 정수 빈도 가중치 나눗셈의 자료형 오류가 발견돼 4개 검증이 중단됐다. 가중치 배열을 실수로 명시한 뒤 같은 합성 검증 9개가 모두 통과했다. 이는 실제 자료 계산 전 수정이며 분석 범위나 판정 기준은 바꾸지 않았다.

본 계산은 1회로 완료했고, 전체 표본의 점 적합 8개와 두 블록 각 100회에 따른 재표집 적합 총 1,600개가 모든 항에서 성공했다. 공동 지지 제한 표본 16개는 해석 기준 부족으로 적합하지 않았다. 고유 날짜를 전 지점과 두 설명식에 공동 적용했으며 지점별 독립 표준오차를 사용하지 않았다.

별도 검산에서 원본 키로 발전량 {verify['generation_cells']:,}셀과 고유 지점 기상 {verify['weather_cells']:,}셀을 재구성했다. 독립적인 집합 연결/제거 구현으로 공동 지지 마스크 {verify['support_masks']}개와 모든 남은 셀/날짜/연결 수를 대조했다. 명시적 월/지점 및 날짜 더미의 직접 최소제곱으로 점 교차곱 8개와 각 블록 최초 표본의 교차곱 16개를 검산했다. 원래 80개 발전소와 지점 평균/빈도 가중의 동치도 8개 월/시간 점검에서 확인했다. 점 계수 최대 차이는 {verify['max_point_beta_difference_pp']:.2e}%p였다.

날짜 가중치 800개와 전체 저장 구간 및 짝지은 차이를 검산했다. 전체 재표집 적합을 독립적으로 다시 수행한 것은 아니다. 원본 네 파일과 본 계산 및 두 의존 코드의 해시를 전후 확인했다. 계산 전 보고서의 내용 해시도 LF/CRLF 차이를 구분해 확인했다. 별도 검산은 종료 코드 0으로 완료됐다.

그림 4개와 갤러리 링크 8개의 파일 검사를 마쳤다. 절반 크기 검토 시트 한 장에서 모든 그림의 양 끝 표본 수, 계수와 구간, 남는 분산 및 날짜 추가 차이를 직접 확인했다. 원본 크기의 전체 픽셀 검사를 뜻하지 않는다.

## 한계 및 다음 단계

이번 결과는 관측 관계의 방향과 큰 조건부 대비의 식별을 나누어 볼 근거다. 전체 선형 관계의 음의 방향은 날짜 조정에도 반복됐지만, 나머지 기상 조건을 함께 맞춘 같은 달력 셀의 넓은 대비와 사전 제한 대비에는 충분한 표본이 없었다. 따라서 습도나 운량을 다른 조건과 독립적으로 크게 바꿨을 때의 실제 효과를 수치로 확정하지 않는다.

날짜/시간 추가 식도 인과 식별을 보장하지 않는다. 조건부 평균의 비선형성, 지역별 반응 차이와 실제 일사의 불완전한 대리, 설비 정보 부재는 남는다. 같은 B 지점의 여러 발전소를 빈도 가중해 이전 표본과 같은 비교량을 유지했지만, 27개 고유 날씨 지점의 범위를 넘는 공간 일반화 구간은 아니다.

날짜 블록은 고정한 지점망과 지지 선택에 조건부이며, 7/14일보다 긴 시간 의존성과 공간 모집단 불확실성을 모두 반영하지 않는다. 두 해는 이미 분석한 개발 자료여서 새로운 독립 미래 검증이 아니다. 다중 비교를 보정한 발견이나 동일성 검정도 수행하지 않았다.

사전 공동 지지의 부족, 전체 표본의 날짜 민감도와 남는 변동, 두 해/두 블록 재현 및 수치 검산을 확인해 DA-007을 종료한다. 다음 권장 후보는 DA-011 A/B 잔여 변동과 발전량 관계다. 다음 분석에서도 큰 표본 내 설명량을 독립적인 기상 효과로 해석하지 않고 자료별 비교량과 지지 한계를 명시한다. 예측 모형 학습/튜닝/추론, 새 예측 파일과 공식 제출은 수행하지 않았다.

## 관련 코드와 로컬 결과

- [계산](../scripts/conditional_support_date.py)
- [합성 검증](../tests/test_conditional_support_date.py)
- [독립 원본/수치 검산](../scripts/verify_conditional_support_date.py)
- [보고서와 그림 생성](../scripts/report_conditional_support_date.py)
- [계산 전 고정 보고서](../data/analysis/2026-10-21_conditional_support_date/preregistered_report.md)
- [전체 로컬 결과](../data/analysis/2026-10-21_conditional_support_date/summary.json)
- [독립 검산 결과](../data/analysis/2026-10-21_conditional_support_date/verification.json)
- [그림 갤러리](../data/analysis/2026-10-21_conditional_support_date/index.html)
- [데이터 분석 계획](데이터분석_진행계획.md)

원본/전처리 행은 보고서에 복사하지 않았다. 패널과 지지 연결, 재표집 및 그림은 무시되는 로컬 data/ 아래에만 보존한다.
'''
    REPORT.write_text(prefix + body, encoding='utf-8')
    print(json.dumps({'gallery': gallery, 'direction': status}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
