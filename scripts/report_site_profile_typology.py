"""Draw and describe saved DA-002 results without fitting new partitions."""
import argparse
import json
import sys

import numpy as np

from site_profile_typology import OUT, REPORT, HOURS, align_labels, dump

COLORS = ['#286582', '#b86c39', '#719468', '#8776a2', '#b85054']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def figures(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot, write_gallery
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    gallery = []
    p24 = np.load(OUT / 'panel_2024.npz')
    p23 = np.load(OUT / 'panel_2023.npz')
    r = np.load(OUT / 'results.npz')
    labels = np.asarray(s['diagnostic_partition']['labels'])
    k = s['evaluation_k']
    profiles = p24['profiles']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for q, ax in enumerate(axes.ravel()):
        for c in range(k):
            z = profiles[labels == c, q] * 100
            for row in z:
                ax.plot(HOURS, row, color=COLORS[c], alpha=.12, linewidth=.6)
            ax.plot(HOURS, z.mean(axis=0), color=COLORS[c], linewidth=2.3, label=f'C{c+1}, n={len(z)}')
        ax.set(title=f'{q+1}분기', xlabel='구간 시작 시각', ylabel='선택 11시간 내 상대 비중 (%)')
        ax.legend()
    save_plot(fig, OUT, gallery, '01_profiles', '2024년 분기별 발전 곡선 형태',
              '각 발전소/분기의 7~17시 평균 곡선을 합이 1이 되도록 정규화했다. 설비 방향이나 효율을 뜻하지 않는다.')
    fig, ax = plt.subplots(figsize=(10, 7))
    xy24 = r['pca_coordinates']
    xy23 = (p23['features'] - p24['features'].mean(axis=0)) @ r['pca_axes'].T
    for c in range(k):
        take = labels == c
        for a, b in zip(xy24[take], xy23[take]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COLORS[c], alpha=.12, linewidth=.7)
        ax.scatter(xy24[take, 0], xy24[take, 1], color=COLORS[c], label=f'C{c+1} 2024', s=24)
        ax.scatter(xy23[take, 0], xy23[take, 1], color=COLORS[c], marker='x', label=f'C{c+1} 2023', s=24)
    ev = s['pca']['explained_variance_ratio_first_two']
    ax.set(xlabel=f'2024 PC1 ({100*ev[0]:.1f}%)', ylabel=f'2024 PC2 ({100*ev[1]:.1f}%)')
    ax.legend(ncol=2, fontsize=8)
    save_plot(fig, OUT, gallery, '02_pca', '분기 곡선 차이의 연속 좌표와 두 해의 변화',
              'PCA는 보조 표시다. 군집은 44개 원래 특징에서 계산했으며 이 2차원에서 선택하지 않았다.')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fits = s['candidates']
    for fit in fits:
        color = COLORS[0] if fit['admissible'] else '#bb4848'
        axes[0].scatter(fit['k'], fit['silhouette'], color=color, s=60)
        axes[0].annotate('/'.join(map(str, fit['sizes'])), (fit['k'], fit['silhouette']), xytext=(0, 12),
                         textcoords='offset points', ha='center', fontsize=8)
    scores = [fit['silhouette'] for fit in fits]
    axes[0].set(xlabel='k', ylabel='평균 실루엣', xticks=[2, 3, 4, 5], title='빨간 점: 최소 크기 미통과',
                ylim=(min(scores) - .025, max(scores) + .05))
    null = r['null_max_silhouette']
    axes[1].hist(null[np.isfinite(null)], bins=22, color='#9baeb7', alpha=.8)
    axes[1].axvline(s['reference']['silhouette_q05_median_q95'][2], color='#ba6c39', linestyle='--', label='참조 95백분위')
    axes[1].axvline(s['diagnostic_partition']['silhouette'], color=COLORS[0], linewidth=2, label='관측 후보')
    axes[1].set(xlabel='허용 후보의 최대 실루엣', ylabel='참조 자료 수')
    axes[1].legend(fontsize=8)
    save_plot(fig, OUT, gallery, '03_selection', '군집 수 선택과 비군집 참조',
              '분할이 가능하다는 사실과 유형의 증거를 구분한다. Gaussian 참조는 평균/공분산을 맞춘 기술적 비교다.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    replica = np.asarray(s['replication_2023']['aligned_labels'])
    contingency = np.zeros((k, k), dtype=int)
    np.add.at(contingency, (labels, replica), 1)
    im = axes[0].imshow(contingency, cmap='Blues')
    for i in range(k):
        for j in range(k):
            axes[0].text(j, i, str(contingency[i, j]), ha='center', va='center')
    axes[0].set(xticks=range(k), yticks=range(k), xticklabels=[f'C{i+1}' for i in range(k)],
                yticklabels=[f'C{i+1}' for i in range(k)], xlabel='2023 정렬 군집', ylabel='2024 군집',
                title=f"연도 ARI {s['replication_2023']['ari']:.3f}")
    fig.colorbar(im, ax=axes[0], shrink=.7)
    order = np.lexsort((xy24[:, 0], labels))
    im = axes[1].imshow(r['coassignment'][np.ix_(order, order)], vmin=0, vmax=1, cmap='viridis')
    axes[1].set(xlabel='발전소 순서', ylabel='발전소 순서', title='날짜 블록 200회 공동 배정 비율')
    fig.colorbar(im, ax=axes[1], shrink=.7)
    save_plot(fig, OUT, gallery, '04_stability', '연도 재현과 날짜 블록의 배정 안정성',
              '같은 80개소에서의 기술적 반복이다. 실제 미래 예측 성능이나 모든 자료 변화에 대한 안정성을 뜻하지 않는다.')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for q, ax in enumerate(axes.ravel()):
        for c in range(k):
            all_shape = profiles[labels == c, q].mean(axis=0) * 100
            weather_shape = r['weather_shapes_2024'][labels == c, q].mean(axis=0) * 100
            ax.plot(HOURS, all_shape, color=COLORS[c], label=f'C{c+1} 전체')
            ax.plot(HOURS, weather_shape, color=COLORS[c], linestyle='--', label=f'C{c+1} 조건 50%')
        ax.set(title=f'{q+1}분기', xlabel='구간 시작 시각', ylabel='선택 시간 내 상대 비중 (%)')
        ax.legend(fontsize=7)
    save_plot(fig, OUT, gallery, '05_weather', '무강수/평균 운량 50% 이하의 곡선 민감도',
              '원래 2024 군집의 같은 발전소로 곡선을 비교했다. 실제 맑은 날 판정이나 완전한 날씨 통제가 아니다.')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    stations = p24['stations']
    station_names = np.unique(stations)
    counts = np.array([[np.sum((stations == st) & (labels == c)) for c in range(k)] for st in station_names])
    im = axes[0].imshow(counts, aspect='auto', cmap='Blues')
    axes[0].set(xlabel='관측 형태 군집', ylabel='고유 B 기상 지점 순서', xticks=range(k), xticklabels=[f'C{i+1}' for i in range(k)])
    fig.colorbar(im, ax=axes[0], shrink=.7, label='발전소 수')
    sd = s['station_diagnostics']
    axes[1].bar([0, 1], [sd['same_label_given_same_station'], sd['same_label_given_different_station']], color=COLORS[:2])
    lo, _, hi = sd['permutation_same_station_quantiles']
    axes[1].axhspan(lo, hi, color='gray', alpha=.18, label='크기 보존 순열 95% 범위')
    axes[1].set(xticks=[0, 1], xticklabels=['같은 기상 지점 쌍', '다른 기상 지점 쌍'],
                ylim=(0, 1), ylabel='같은 군집 비율')
    axes[1].legend(fontsize=8)
    save_plot(fig, OUT, gallery, '06_station', '기상 지점 공유와 형태 분할의 관계',
              '동일 지점의 연관은 실제 날씨, 위치와 설비 요인을 분리하지 못한다. 쌍은 독립 관측이 아니다.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    levels = p24['cf'].mean(axis=(0, 2))
    axes[0].boxplot([levels[labels == c] for c in range(k)], tick_labels=[f'C{c+1}' for c in range(k)], showfliers=True)
    axes[0].set(ylabel='선택 11시간의 평균 용량 대비 발전 비율', xlabel='형태 군집')
    amplitude_labels = np.asarray(s['amplitude_sensitivity']['partition']['labels'])
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (labels, amplitude_labels), 1)
    im = axes[1].imshow(counts, cmap='Oranges')
    for i in range(k):
        for j in range(k):
            axes[1].text(j, i, str(counts[i, j]), ha='center', va='center')
    axes[1].set(xlabel='평균 수준을 남긴 분할', ylabel='정규화 형태 분할', xticks=range(k), yticks=range(k),
                xticklabels=[f'수준 {i+1}' for i in range(k)], yticklabels=[f'C{i+1}' for i in range(k)],
                title=f"ARI {s['amplitude_sensitivity']['ari_vs_shape']:.3f}")
    fig.colorbar(im, ax=axes[1], shrink=.7)
    save_plot(fig, OUT, gallery, '07_amplitude', '발전 수준과 곡선 형태의 구분',
              '수준을 남겨 얻은 분할로 주 결과를 교체하지 않았다. 용량 대비 값은 실제 설비 효율을 뜻하지 않는다.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    time_center = (profiles * HOURS).sum(axis=2).mean(axis=1)
    morning = profiles[:, :, HOURS <= 11].sum(axis=2).mean(axis=1) * 100
    for c in range(k):
        take = labels == c
        axes[0].scatter(time_center[take], morning[take], s=25, color=COLORS[c], label=f'C{c+1}')
        axes[1].hist(time_center[take], bins=12, alpha=.55, color=COLORS[c], label=f'C{c+1}')
    axes[0].set(xlabel='구간 시작 시각으로 가중한 곡선 시간 중심', ylabel='7~11시 상대 비중 (%)')
    axes[1].set(xlabel='곡선 시간 중심', ylabel='발전소 수')
    axes[0].legend()
    axes[1].legend()
    save_plot(fig, OUT, gallery, '08_descriptors', '관측 곡선의 시간 중심과 오전 비중',
              '차이는 시간당 자료에서 계산한 곡선 요약이다. 시간 기록이 그만큼 어긋났다는 증거가 아니다.')
    write_gallery(OUT, gallery, 'DA-002 발전소 곡선 유형 분석')
    p24.close()
    p23.close()
    r.close()


def report(s):
    from eda_plot_utils import check_gallery
    dump(OUT / 'figure_verification.json', check_gallery(OUT, 8))
    check = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    header = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').split('## 결과')[0]
    header = header.replace('상태: DA-002 진행. 아래 계약을 발전 곡선과 군집 결과 계산 전에 고정했다.',
                            '상태: DA-002 완료. 아래 계약은 발전 곡선과 군집 결과 계산 전에 고정했고 변경 없이 실행했다.')
    selected = s['diagnostic_partition']
    boot = s['bootstrap']
    weather = s['weather_comparison']
    station = s['station_diagnostics']
    descriptors = s['cluster_descriptors']
    time_difference = 60 * (descriptors[1]['time_center_median'] - descriptors[0]['time_center_median'])
    morning_difference = 100 * (descriptors[0]['morning_share_median'] - descriptors[1]['morning_share_median'])
    body = ['## 결과', '',
        '이번 곡선 표현과 사전 기준에서는 47개소/33개소의 두 관측 유형 후보가 남았다. '
        '2023년의 독립 재군집과 2024년 날짜 블록 200회에서 같은 80개소의 구분이 반복됐다. '
        '운량 50% 이하의 무강수 기록 조건에서도 대부분 유지됐다. 다만 정확히 두 개의 자연 집단이나 '
        '실제 설비 종류를 입증한 것은 아니며, 더 세밀한 작은 집단과 연속적인 차이는 남는다.', '',
        '### 실제 사용 표본과 곡선 표현', '']
    rows = []
    for year, a in s['audit'].items():
        rows.append([year, a['sites'], a['stations'], a['balanced_dates'], f"{a['balanced_rows']:,}",
                     a['excluded_dates'], a['zero_rows'], a['above_one_rows']])
    body += [table(['연도', '발전소', '고유 B 지점', '날짜', '11시간 행', '제외 날짜', '0 기록', '비율 1 초과'], rows), '',
        '한 분기의 날짜들에서 시간대별 평균을 구한 뒤 그 11시간 합으로 나눴다. 각 분기 합은 1이고 '
        '네 분기를 같은 비중으로 연결했다. 용량으로 나눈 값을 다시 분기 합으로 나누므로 정적 용량의 '
        '상수배도 소거된다. 따라서 이 군집은 큰 발전소와 작은 발전소의 구분이나 생산량 순위가 아니라 '
        '선택 시간 안에서 발전이 배분되는 형태의 구분이다. 높은 발전일이 분기 평균 곡선에 더 많이 기여할 수 있고 '
        '기록된 0도 유지했으므로 모든 운영 영향을 제거한 순수 설비 곡선이라고 말할 수는 없다.', '',
        '### 군집 수와 비군집 참조', '']
    rows = [[v['k'], '/'.join(map(str, v['sizes'])), f"{v['silhouette']:.4f}",
             '통과' if v['admissible'] else '최소 크기 미달'] for v in s['candidates']]
    body += [table(['k', '군집 크기', '평균 실루엣', '사전 크기 기준'], rows), '',
        'k=3은 실루엣이 더 높았지만 5개소 집단이 포함돼 최소 8개소 기준을 통과하지 못했다. '
        'k=4/5도 작은 집단이 남았다. 따라서 허용된 후보 중 k=2를 선택했다. 더 작은 집단이 실제로 '
        '존재하지 않는다는 뜻은 아니며 최소 집단 크기라는 작업 목적이 선택에 영향을 줬다.', '',
        f"관측 후보의 실루엣은 {selected['silhouette']:.4f}, Gaussian 참조 최대 실루엣의 95백분위는 "
        f"{s['reference']['silhouette_q05_median_q95'][2]:.4f}였다. 199개 참조 모두 허용 후보가 있었고 "
        '이번 분리 기준을 통과했다. 하나의 타원형 연속 분포를 억지로 잘랐을 때와 비교한 결과이며 '
        '모든 비군집 분포를 배제하거나 자연 군집의 존재를 확증한 검정은 아니다. '
        f"참조에서 음의 특징 성분은 평균 {100*s['reference']['negative_feature_share_mean']:.3f}%였다. "
        '실제 발전 곡선의 물리적 생성 모형을 사용한 것은 아니다.', '',
        '### 두 유형은 오전 배분과 시간 중심이 다르다', '']
    rows = []
    for d in descriptors:
        cf = d['day_cf_q25_median_q75']
        rows.append([f"C{d['cluster']}", d['sites'], f"{100*d['morning_share_median']:.2f}%",
            f"{d['time_center_median']:.3f}", f"{cf[1]:.4f} [{cf[0]:.4f}, {cf[2]:.4f}]",
            f"{d['capacity_kw_median']:.3f}", d['stations']])
    body += [table(['유형 후보', '발전소', '7~11시 상대 비중 중앙값', '시간 중심 중앙값', '평균 발전 비율 중앙값 [IQR]', '용량 중앙값 kW', 'B 지점 수'], rows), '',
        f"C1은 C2보다 오전 상대 비중 중앙값이 약 {morning_difference:.2f} pp 높고 시간 중심은 약 "
        f"{time_difference:.1f}분 앞섰다. 시간 중심은 구간 시작 시각을 곡선 비중으로 가중한 요약이며 "
        '실제 측정 간격이나 시각 오류를 추정한 값이 아니다. C1을 동향 설비, C2를 서향 설비처럼 '
        '명명할 근거는 없다. 방향, 경사, 지형, 제약/운영과 기록 생성 과정의 영향은 분리하지 못했다.', '',
        '평균 발전 비율의 분포는 겹쳤다. 수준을 남긴 용량 대비 곡선을 같은 k로 다시 나눴을 때 주 형태 '
        f"분할과의 ARI는 {s['amplitude_sensitivity']['ari_vs_shape']:.3f}였다. 발전 수준과 시간 배분은 "
        '다른 축의 차이이며 하나를 다른 하나의 등급으로 해석하지 않는다. 두 유형 사이의 값 차이를 '
        '설비 효율이나 수익 우열로 판단하지 않았다.', '',
        '### 연도와 날짜 블록에서의 재현', '',
        f"2023년 독립 재군집은 동일한 47/33개소로 나뉘었고 2024년과 ARI {s['replication_2023']['ari']:.3f}였다. "
        f"2024년 날짜 블록은 {boot['success']}/{boot['attempts']}회 성공했고, ARI의 5백분위/중앙값/95백분위는 "
        '/'.join(f'{v:.3f}' for v in boot['ari_q05_median_q95']) + '였다. '
        '최소 군집 크기 통과율도 100%였으며 순열을 정렬한 발전소별 배정은 모두 원래 분할과 같았다.', '',
        '따라서 이 집단과 관측 기간에서 일시적인 날짜 구성이나 군집 번호의 순열만으로 생긴 차이라고 '
        '보기는 어렵다. 그렇더라도 재표집은 같은 2024년 기록을 반복 사용한 것이며 이미 확인한 두 해는 '
        '새로운 미래 시험이 아니다. 계측 방식이나 운영이 바뀐 미래에도 동일한 유형이 유지된다는 보장은 없다.', '',
        '### 제한적인 기상 조건에서도 구분이 남는다', '',
        '평균 운량 25% 이하 조건은 두 해/모든 분기에서 10일 이상 확보한 발전소가 3개뿐이라 '
        '군집 비교를 수행하지 않았다. 이 부족을 유지하고, 계산 전에 가용성을 확인한 운량 50% 이하 '
        '무강수 기록 조건에서 80개소 모두를 비교했다. 조건 판정은 9~16시의 같은 날짜이고 '
        '곡선은 그 날짜의 7~17시 전체를 사용했다.', '',
        f"2024년 전체 날짜 분할과 조건 50% 날짜 분할의 ARI는 {weather['full_vs_moderate_ari']:.3f}, "
        f"조건 50%의 두 해 분할끼리는 {weather['moderate_annual_ari']:.3f}였다. "
        '단순히 흐린 날의 빈도 차이만으로 두 형태가 나타났다고 보기는 어렵다는 보조 근거다. '
        '그러나 실제 맑음, 일사량, 강수 정의와 현장 날씨를 완전히 맞춘 분석은 아니며 분기 안의 날짜 구성도 다를 수 있다.', '',
        '### 공유 기상 지점과의 관계도 일부 남는다', '',
        f"같은 B 기상 지점을 공유한 {station['same_station_pairs']}개 발전소 쌍에서 같은 군집 비율은 "
        f"{100*station['same_label_given_same_station']:.1f}%, 다른 지점의 {station['different_station_pairs']}개 쌍에서는 "
        f"{100*station['same_label_given_different_station']:.1f}%였다. 군집 크기를 보존한 라벨 순열의 "
        f"같은 지점 비율 95% 범위는 {100*station['permutation_same_station_quantiles'][0]:.1f}~"
        f"{100*station['permutation_same_station_quantiles'][2]:.1f}%였다. AMI는 {station['ami']:.3f}였다. "
        '지점과 분할의 연관을 기상 원인으로 확정하거나 쌍 독립 p값으로 바꾸지는 않았다.', '',
        f"같은 기상 지점 평균을 제거한 뒤에도 전체 곡선 분산의 {100*station['within_station_variance_fraction']:.1f}%가 남았고 "
        f"{station['stations_with_multiple_labels']}개 기상 지점에 두 군집이 함께 존재했다. 공유 기상 지점만으로 "
        '구분을 완전히 설명할 수는 없다. 반면 제공 지점의 실제 대표성, 위치와 설비 요인을 분리하지 못해 '
        '기상과 무관한 순수한 설비 유형이라고도 결론 내리지 않는다.', '',
        '## 한계 및 다음 단계', '',
        '사전 최소 크기, Euclidean 거리와 KMeans의 볼록한 분할, 선택 11시간과 분기 표현에 한정된 '
        '유형 후보다. k=3의 작은 집단, 단일 곡선의 연속 차이와 분기 안의 변화는 남겨 두었다. '
        'PCA 첫 두 축의 전체 분산 비중은 '
        '/'.join(f'{100*v:.1f}%' for v in s['pca']['explained_variance_ratio_first_two']) +
        '였지만 PCA에서 집단을 선택한 것은 아니다.', '',
        '80개소는 두 해 완전 기록 교집합이다. 전체 253개소나 기록이 짧은 발전소의 유형은 '
        '검증하지 않았다. 용량과 기상 지점 매핑은 정적 메타를 과거에 적용했고 실제 설비 방향, '
        '경사와 운영 이력은 없으므로 분할 원인을 특정하지 않았다. 시간 중심의 약 15분 차이만으로 '
        '어느 군집의 시각을 이동하거나 원본 기록을 수정하지 않는다.', '',
        'DA-002는 이번 표현에서 시간과 제한 기상 민감도를 통과한 두 관측 유형 후보를 기록하고 종료한다. '
        '발생 원인과 기록 시각의 관계는 DA-008에 추가 자료 필요 보류 질문으로 남긴다. '
        '다음 우선 후보는 DA-004 시간에 따른 구조 변화이며 실제 범위와 검증 기준은 시작 전에 별도로 정한다.', '',
        '모델링 활용은 가설 단계다. 동일한 시간 특징이 모든 발전소의 곡선을 똑같이 표현하는지 '
        '향후 확인할 근거는 생겼으나, 유형별 별도 모델이나 교호작용이 성능을 개선한다고 검증하지 않았다. '
        '발전소 배정 최적화와 발전량 예측 학습/튜닝/추론, V2/평가 구간 및 공식 제출은 모두 0회다.', '',
        '## 실행과 검증', '',
        '정규 CPython 3.13.1의 승격 Start-Process 경로로 가용성 점검 두 번, 합성 검증, 본 계산과 '
        '독립 검산, 그림/보고서 작성을 실행했고 종료 코드는 모두 0이었다. 군집 계산의 실패 재시도는 없었다. '
        '25% 조건의 부족과 50% 조건 보완은 군집 결과를 보기 전에 기록했으며 군집 수나 재현 기준을 사후 변경하지 않았다.', '',
        '합성 검증 9개로 규모/분기 배수에 대한 불변성, 명시적 날짜 반복과 가중 곡선의 동치, '
        '기상 조건의 날짜 단위 선택, 빈 분기와 0 곡선 거부, 알려진 두 곡선 회복, 최소 크기 선택 정책, '
        '번호 순열, 참조의 분기 합 제약과 재표집 날짜 수를 확인했다.', '',
        '본 실행의 KMeans 호출은 후보 4회, Gaussian 참조 796회, 2023년 반복 1회, 날짜 블록 200회, '
        f"기상 조건 2회와 수준 민감도 1회로 총 {sum(s['fit_calls'].values()):,}회였다. 각 호출의 n_init는 20이다. "
        '이 수는 탐색/설명 목적의 군집 적합이며 발전량 예측 모델 학습 횟수가 아니다.', '',
        f"독립 검산에서는 연도 곡선 {check['annual_profiles_reconstructed']}개와 조건 곡선 "
        f"{check['weather_profiles_reconstructed']}개를 별도 평균 수식으로 재구성했다. "
        f"분할 {check['partitions_checked_with_direct_silhouette_and_nearest_centers']}개는 거리 행렬로 직접 계산한 "
        '실루엣과 최근접 중심/관성, 크기를 확인했다. 날짜 블록 200개의 ARI와 공동 배정, 첫 재표집 분할, '
        '참조 분위수, 수준 민감도 및 PCA 고유값도 재확인했다. 원본 3개 파일과 분석 코드의 SHA-256은 실행 전후와 검증 시점에 동일했다.', '',
        '그림 8개와 갤러리 링크 16개를 검사하고 모든 그림을 50% 크기의 4개 묶음 두 장에서 직접 확인했다. '
        '원본 크기 전체 픽셀 검사를 뜻하지 않는다. 분기 곡선, 선택 미달 후보, 연도/기상 비교와 '
        '단위/범례를 구분해 확인했다.', '',
        '## 관련 코드와 로컬 결과', '',
        '- [분석 코드](../scripts/site_profile_typology.py)',
        '- [합성 검증](../tests/test_site_profile_typology.py)',
        '- [독립 검산](../scripts/verify_site_profile_typology.py)',
        '- [보고서와 그림 생성](../scripts/report_site_profile_typology.py)',
        '- [계산 전 고정 보고서](../data/analysis/2026-10-21_site_typology/preregistered_report.md)',
        '- [전체 결과](../data/analysis/2026-10-21_site_typology/summary.json)',
        '- [검증 결과](../data/analysis/2026-10-21_site_typology/verification.json)',
        '- [그림 갤러리](../data/analysis/2026-10-21_site_typology/index.html)',
        '- [분석 계획](데이터분석_진행계획.md)', '',
        '방법 참고: [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), '
        '[실루엣](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), '
        '[ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html). '
        '실제 사용한 버전은 1.7.1이며 실행 시 설정을 명시했다.', '',
        '원본/전처리 행과 개별 발전소 배정은 보고서에 복사하지 않았다. 패널과 배정, 개별 곡선 및 그림은 '
        '로컬 data/에만 보관하고 공개 가능한 코드와 해석 보고서, 로그만 로컬 커밋한다. push는 하지 않는다.', '']
    text = header + '\n'.join(body)
    assert '\u00b7' not in text
    REPORT.write_text(text, encoding='utf-8')
    print(str(REPORT))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['figures', 'report'])
    args = parser.parse_args()
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    if args.mode == 'figures':
        figures(summary)
    else:
        report(summary)
