"""EDA of weather-provider pairs assigned to the same plants. No model fitting.

Repeated plant mappings are collapsed to unique A/B pairs. All data-bearing
outputs remain in ignored data/. Differences are disagreements, not errors.
"""
import gc
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from basic_shape_eda import ROOT, SOURCE, digest, fmt, md_table, log
from eda_plot_utils import save_plot, write_gallery, check_gallery

OUT = ROOT / 'data/eda/2026-10-19_weather_provider_pairs'
REPORT = ROOT / 'reports/2026-10-19_16_AB기상지점쌍_관계_EDA.md'
FILES = ['sites.csv', 'README.md', *[f'{s}/forecast_{p}.parquet' for s in ['train', 'test'] for p in ['wa', 'wb']],
         'train/observed_wa.parquet', 'train/observed_wb.parquet']
META = {'temp': ('기온', '°C'), 'humidity': ('습도', '%p'), 'cloud': ('운량', '%p'),
        'speed': ('풍속', 'm/s'), 'direction': ('풍향', '°'), 'rain': ('강우/강수 참고', 'mm'),
        'snow': ('강설 참고', 'mm')}
LINEAR = ['temp', 'humidity', 'cloud', 'speed']
COLORS = ['#26718c', '#d7804f']
TABLES = {k: [] for k in ['summary', 'pair_metrics', 'monthly', 'events', 'audit', 'wind', 'quality']}
GALLERY = []
INFO = {}


def columns(provider, kind):
    return {v: provider + '_' + suffix for v, suffix in {
        'temp': 'temp', 'humidity': 'humidity' if provider == 'wa' else 'rel_hum',
        'cloud': 'cloud', 'speed': 'wind_speed', 'direction': 'wind_dir',
        'rain': 'precip_1h' if provider == 'wb' and kind == 'observed' else 'rain',
        **({'snow': 'snow'} if kind == 'forecast' else {})}.items()}


def wrapped(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def correlation(x, y):
    return float(pd.Series(np.asarray(x)).corr(pd.Series(np.asarray(y)))) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else np.nan


def permitted(x, variable):
    good = np.isfinite(x)
    if variable in ['humidity', 'cloud']:
        good &= x.between(0, 100)
    elif variable == 'speed':
        # Provider-specific upper bounds are applied before unit conversion.
        good &= x.ge(0)
    elif variable == 'direction':
        good &= x.between(0, 360)
    elif variable in ['rain', 'snow']:
        good &= x.ge(0)
    return good


def load(provider, split, kind, lo, hi, round_number):
    mapping = columns(provider, kind)
    key = provider + '_station'
    selected = [key, 'ts', *mapping.values()] + (['round'] if kind == 'forecast' else [])
    data = pd.read_parquet(SOURCE / f'{split}/{kind}_{provider}.parquet', columns=selected)
    begin = pd.Timestamp(lo, tz='Asia/Seoul')
    end = pd.Timestamp(hi, tz='Asia/Seoul')
    data = data.loc[data.ts.gt(begin) & data.ts.le(end)].copy()
    if kind == 'forecast':
        data = data.loc[data['round'].eq(round_number)].drop(columns='round')
    nonhour = int(data.ts.ne(data.ts.dt.floor('h')).sum())
    data = data.loc[data.ts.eq(data.ts.dt.floor('h'))]
    duplicate_rows = int(data.duplicated([key, 'ts'], keep=False).sum())
    conflict_cells = 0
    if duplicate_rows:
        repeated = data.duplicated([key, 'ts'], keep=False)
        grouped = data.loc[repeated].groupby([key, 'ts'], observed=True)[list(mapping.values())]
        # dropna=False makes a missing-versus-present conflict explicit too.
        conflicts = grouped.nunique(dropna=False).gt(1)
        conflict_cells = int(conflicts.to_numpy().sum())
        fixed = grouped.first().mask(conflicts).reset_index()
        data = pd.concat([data.loc[~repeated], fixed], ignore_index=True)
    assert not data.duplicated([key, 'ts']).any()
    data = data.rename(columns={c: provider + '_' + v for v, c in mapping.items()})
    if provider == 'wb':
        data['wb_speed'] /= 3.6
        if kind == 'forecast':
            data['wb_snow'] *= 10
    return data, {'provider': provider, 'nonhour_rows': nonhour, 'duplicate_rows': duplicate_rows,
                  'conflict_cells': conflict_cells, 'station_rows': len(data)}


def joined(pairs, split, kind, lo, hi, r, label):
    a, aa = load('wa', split, kind, lo, hi, r)
    b, bb = load('wb', split, kind, lo, hi, r)
    times = pd.date_range(pd.Timestamp(lo, tz='Asia/Seoul') + pd.Timedelta(hours=1),
                          pd.Timestamp(hi, tz='Asia/Seoul'), freq='h')
    grid = pd.MultiIndex.from_product([pairs.pair_id, times], names=['pair_id', 'ts']).to_frame(index=False)
    grid = grid.merge(pairs, on='pair_id', validate='many_to_one')
    a['_a_exists'] = 1
    b['_b_exists'] = 1
    grid = grid.merge(a, on=['wa_station', 'ts'], how='left', validate='many_to_one')
    grid = grid.merge(b, on=['wb_station', 'ts'], how='left', validate='many_to_one')
    assert len(grid) == len(pairs) * len(times) and not grid.duplicated(['pair_id', 'ts']).any()
    counts = {'both_keys': int((grid._a_exists.eq(1) & grid._b_exists.eq(1)).sum()),
              'only_a': int((grid._a_exists.eq(1) & grid._b_exists.isna()).sum()),
              'only_b': int((grid._a_exists.isna() & grid._b_exists.eq(1)).sum()),
              'neither_key': int((grid._a_exists.isna() & grid._b_exists.isna()).sum())}
    TABLES['audit'].append({'dataset': label, 'expected': len(grid), 'hours': len(times), 'pairs': len(pairs), **counts,
                            **{f'{x["provider"]}_{k}': v for x in [aa, bb] for k, v in x.items() if k != 'provider'}})
    start = grid.ts - pd.Timedelta(hours=1)
    grid['month'] = start.dt.month
    grid['hour'] = start.dt.hour
    del a, b
    gc.collect()
    return grid


def pair_summary(z, variable):
    z = z.copy()
    z['delta'] = wrapped(z.x, z.y) if variable == 'direction' else z.x - z.y
    z['absolute'] = z.delta.abs()
    z['square'] = z.delta ** 2
    z['equal'] = z.absolute.le(1e-9)
    z['both_zero'] = z.x.eq(0) & z.y.eq(0)
    group = z.groupby('pair_id', observed=True)
    p = group.agg(n=('x', 'size'), mean_a=('x', 'mean'), mean_b=('y', 'mean'),
                  mean_difference=('delta', 'mean'), mean_absolute_difference=('absolute', 'mean'),
                  mean_square=('square', 'mean'), equal_share=('equal', 'mean'), both_zero_share=('both_zero', 'mean'))
    p['root_mean_square_difference'] = np.sqrt(p.pop('mean_square'))
    p['pearson'] = np.nan
    p['spearman'] = np.nan
    p['within_pearson'] = np.nan
    p['day_pearson'] = np.nan
    if variable != 'direction':
        centered = z[['x', 'y']] - z.groupby(['pair_id', 'month', 'hour'], observed=True)[['x', 'y']].transform('mean')
        z['cx'] = centered.x
        z['cy'] = centered.y
        for pid, rows in z.groupby('pair_id', observed=True):
            p.loc[pid, 'pearson'] = correlation(rows.x, rows.y)
            p.loc[pid, 'spearman'] = rows.x.corr(rows.y, method='spearman') if rows.x.nunique() > 1 and rows.y.nunique() > 1 else np.nan
            p.loc[pid, 'within_pearson'] = correlation(rows.cx, rows.cy)
            day = rows.hour.between(9, 16)
            p.loc[pid, 'day_pearson'] = correlation(rows.loc[day, 'x'], rows.loc[day, 'y'])
    else:
        z['cos_delta'] = np.cos(np.deg2rad(z.delta))
        z['within30'] = z.absolute.le(30)
        p['mean_cosine_difference'] = z.groupby('pair_id').cos_delta.mean()
        p['within30_share'] = z.groupby('pair_id').within30.mean()
    return z, p.reset_index()


def event_metrics(z, label, variable):
    for threshold in [0., .1, 1.]:
        a = z.x.gt(threshold)
        b = z.y.gt(threshold)
        table = pd.DataFrame({'pair_id': z.pair_id, 'neither': ~a & ~b, 'a_only': a & ~b,
                              'b_only': ~a & b, 'both': a & b})
        counts = table.iloc[:, 1:].sum()
        balanced = table.groupby('pair_id').mean().mean()
        union = balanced.a_only + balanced.b_only + balanced.both
        TABLES['events'].append({'dataset': label, 'variable': variable, 'threshold': threshold, 'n': len(z),
                                 **{k: int(v) for k, v in counts.items()},
                                 **{k + '_pair_share': float(v) for k, v in balanced.items()},
                                 'pair_balanced_union_overlap': float(balanced.both / union) if union else np.nan})


def analyze(data, label, kind, tag):
    views = {}
    for variable in columns('wa', kind):
        x, y = data['wa_' + variable], data['wb_' + variable]
        finite = np.isfinite(x) & np.isfinite(y)
        good = permitted(x, variable) & permitted(y, variable)
        if variable == 'speed':
            good &= x.le(50) & y.le(200 / 3.6)
        raw_diff = wrapped(x.loc[finite], y.loc[finite]) if variable == 'direction' else (x.loc[finite] - y.loc[finite]).to_numpy()
        TABLES['quality'].append({'dataset': label, 'variable': variable, 'grid_rows': len(data),
                                 'finite_pairs': int(finite.sum()), 'comparison_pairs': int(good.sum()),
                                 'range_excluded': int((finite & ~good).sum()),
                                 'finite_row_mean_abs_difference': float(np.abs(raw_diff).mean()),
                                 'finite_row_pearson': correlation(x.loc[finite], y.loc[finite]) if variable != 'direction' else np.nan})
        z = data.loc[good, ['pair_id', 'plants', 'month', 'hour']].copy()
        z['x'] = x.loc[good]
        z['y'] = y.loc[good]
        z, per_pair = pair_summary(z, variable)
        per_pair['dataset'] = label
        per_pair['variable'] = variable
        TABLES['pair_metrics'].extend(per_pair.to_dict('records'))
        balanced = per_pair.mean_absolute_difference.mean()
        mapped_counts = data[['pair_id', 'plants']].drop_duplicates().set_index('pair_id').plants
        positive = z.x.gt(0) & z.y.gt(0)
        TABLES['summary'].append({'dataset': label, 'variable': variable, 'unit': META[variable][1],
            'n': len(z), 'pairs': len(per_pair), 'mean_difference': per_pair.mean_difference.mean(),
            'mean_absolute_difference': balanced, 'pair_mad_p10': per_pair.mean_absolute_difference.quantile(.1),
            'pair_mad_p90': per_pair.mean_absolute_difference.quantile(.9),
            'plant_count_weighted_mad': np.average(per_pair.mean_absolute_difference, weights=per_pair.pair_id.map(mapped_counts)),
            'row_mean_absolute_difference': z.absolute.mean(),
            'pooled_pearson': correlation(z.x, z.y) if variable != 'direction' else np.nan,
            'median_pair_pearson': per_pair.pearson.median(), 'median_pair_spearman': per_pair.spearman.median(),
            'median_pair_within_pearson': per_pair.within_pearson.median(),
            'within_pairs_defined': int(per_pair.within_pearson.notna().sum()),
            'pair_equal_share': per_pair.equal_share.mean(), 'pair_both_zero_share': per_pair.both_zero_share.mean(),
            'both_positive_n': int(positive.sum()),
            'both_positive_spearman': z.loc[positive, 'x'].corr(z.loc[positive, 'y'], method='spearman') if positive.sum() > 2 and variable in ['rain', 'snow'] else np.nan})
        if variable in LINEAR:
            month = z.groupby(['pair_id', 'month'], observed=True).agg(n=('x', 'size'), mean_a=('x', 'mean'),
                        mean_b=('y', 'mean'), mean_difference=('delta', 'mean'), mean_absolute_difference=('absolute', 'mean'))
            month = month.reset_index()
            month['dataset'] = label
            month['variable'] = variable
            TABLES['monthly'].extend(month.to_dict('records'))
        if variable in ['rain', 'snow']:
            event_metrics(z, label, variable)
        if variable == 'direction':
            strong = data.loc[z.index, 'wa_speed'].between(1, 50, inclusive='right') & data.loc[z.index, 'wb_speed'].between(1, 200 / 3.6, inclusive='right')
            z['strong'] = strong
            for scope, selection in [('전체 시간', np.ones(len(z), dtype=bool)), ('양쪽 풍속 1 m/s 초과', strong.to_numpy())]:
                zz = z.loc[selection]
                gg = zz.groupby('pair_id').agg(n=('x', 'size'), mad=('absolute', 'mean'), cosine=('cos_delta', 'mean'), near=('within30', 'mean'))
                TABLES['wind'].append({'dataset': label, 'scope': scope, 'n': len(zz),
                    'pairs': len(gg), 'mean_absolute_angle': gg.mad.mean(), 'mean_cosine': gg.cosine.mean(), 'within30_share': gg.near.mean()})
        views[variable] = z
    plot_dataset(views, label, tag, kind)
    log(label + ' pair comparison complete')


def plot_dataset(views, label, tag, kind):
    rng = np.random.default_rng(20261019)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, v in zip(axes.flat, LINEAR):
        z = views[v]
        take = rng.choice(len(z), min(len(z), 60000), replace=False)
        xx, yy = z.x.iloc[take], z.y.iloc[take]
        ax.hexbin(xx, yy, gridsize=45, bins='log', mincnt=1, cmap='viridis')
        low, high = min(xx.min(), yy.min()), max(xx.max(), yy.max())
        ax.plot([low, high], [low, high], color='#a55043', lw=1)
        value_unit = '%' if v in ['humidity', 'cloud'] else META[v][1]
        ax.set_title(META[v][0]); ax.set_xlabel('A (' + value_unit + ')'); ax.set_ylabel('B (' + value_unit + ')')
    save_plot(fig, OUT, GALLERY, tag + '_density', label + ' / A와 B 값의 모양',
              '변수별 비교 범위 통과 쌍. 밀도만 고정 난수 최대 6만 행, 통계는 전체 유효 쌍. 붉은 선은 값 일치선.')
    p = pd.DataFrame(TABLES['pair_metrics']); p = p.loc[p.dataset.eq(label)]
    variables = [v for v in views if v != 'direction']
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for offset, metric, color, name in [(-.16, 'pearson', COLORS[0], '전체'), (.16, 'within_pearson', COLORS[1], '쌍/월/시간 평균 편차')]:
        values = [p.loc[p.variable.eq(v), metric].dropna().to_numpy() for v in variables]
        boxes = ax.boxplot(values, positions=np.arange(len(variables)) + offset, widths=.26, patch_artist=True,
                           manage_ticks=False, tick_labels=None)
        for box in boxes['boxes']: box.set_facecolor(color); box.set_alpha(.65)
        ax.plot([], [], color=color, lw=8, label=name)
    ax.set_xticks(range(len(variables)), [META[v][0] for v in variables]); ax.set_ylim(-1, 1.03)
    ax.axhline(0, color='#aaa', lw=1); ax.legend(); ax.set_ylabel('지점 쌍별 Pearson 분포')
    save_plot(fig, OUT, GALLERY, tag + '_pair_correlations', label + ' / 공통 패턴을 뺀 뒤의 관계',
              '87쌍을 같은 비중으로 표시. 상수이면 상관 미정의. 상자와 수염은 기술통계이며 신뢰구간이 아님.')
    m = pd.DataFrame(TABLES['monthly']); m = m.loc[m.dataset.eq(label)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, v in zip(axes.flat, LINEAR):
        z = m.loc[m.variable.eq(v)].groupby('month').mean_difference
        mean = z.mean(); lower = z.quantile(.1); upper = z.quantile(.9)
        ax.fill_between(mean.index, lower, upper, color=COLORS[0], alpha=.2)
        ax.plot(mean.index, mean, marker='o', color=COLORS[0]); ax.axhline(0, color='#999', lw=1)
        ax.set_title(META[v][0]); ax.set_xlabel('구간 시작 달력 월'); ax.set_xticks(range(1, 13)); ax.set_ylabel('A-B (' + META[v][1] + ')')
    save_plot(fig, OUT, GALLERY, tag + '_monthly', label + ' / 월별 제공사 차이',
              '선은 쌍별 월평균 차이의 평균, 띠는 쌍 사이 10~90백분위. 서로 다른 지점의 값 차이며 정확도 차이가 아님.')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    z = views['direction']
    for mask, name, color in [(np.ones(len(z), dtype=bool), '전체', COLORS[0]), (z.strong.to_numpy(), '양쪽 풍속 >1 m/s', COLORS[1])]:
        values = z.loc[mask, 'delta']; count, edges = np.histogram(values, bins=np.linspace(-180, 180, 73))
        axes[0, 0].stairs(count / len(values), edges, label=name, color=color)
    axes[0, 0].set_xlabel('풍향 A-B의 최단 각도 (°)'); axes[0, 0].set_ylabel('행 비중'); axes[0, 0].legend()
    e = pd.DataFrame(TABLES['events']); e = e.loc[e.dataset.eq(label) & e.variable.eq('rain')]
    bottom = np.zeros(len(e))
    for key, color in zip(['neither', 'a_only', 'b_only', 'both'], ['#d4d9dc', '#26718c', '#d7804f', '#514d73']):
        values = 100 * e[key + '_pair_share'].to_numpy()
        axes[0, 1].bar(np.arange(len(e)), values, bottom=bottom, label=key, color=color); bottom += values
    axes[0, 1].set_xticks(range(len(e)), ['>' + str(x) for x in e.threshold]); axes[0, 1].set_xlabel('강우/강수 참고 기준 (mm)')
    axes[0, 1].set_ylabel('쌍 균등 비중 (%)'); axes[0, 1].legend(fontsize=8)
    z = views['rain']; pos = z.loc[z.x.gt(0) & z.y.gt(0)]
    take = rng.choice(len(pos), min(len(pos), 60000), replace=False)
    axes[1, 0].hexbin(pos.x.iloc[take], pos.y.iloc[take], gridsize=40, bins='log', mincnt=1, cmap='viridis')
    axes[1, 0].set_xlabel('A 양수 강우 (mm)'); axes[1, 0].set_ylabel('B 양수 강우/강수 (mm)')
    if kind == 'forecast':
        e = pd.DataFrame(TABLES['events']); e = e.loc[e.dataset.eq(label) & e.variable.eq('snow')]
        axes[1, 1].plot(e.threshold, e.pair_balanced_union_overlap, marker='o', color=COLORS[0])
        axes[1, 1].set_xlabel('강설 기준 초과 (mm, B cm×10)'); axes[1, 1].set_ylabel('동시 초과 / 한쪽 이상 초과'); axes[1, 1].set_ylim(0, 1)
    else:
        z = views['cloud']
        curves = z.groupby(['pair_id', 'hour'])[['x', 'y']].mean().groupby('hour').mean()
        axes[1, 1].plot(curves.index, curves.x, label='A', color=COLORS[0]); axes[1, 1].plot(curves.index, curves.y, label='B', color=COLORS[1])
        axes[1, 1].set_xlabel('구간 시작 시간'); axes[1, 1].set_ylabel('쌍 균등 평균 운량 (%)'); axes[1, 1].legend()
    save_plot(fig, OUT, GALLERY, tag + '_wind_rain', label + ' / 풍향과 희소 강수의 차이',
              '풍향은 원형 차이. 강수 4분할은 A/B를 대칭 비교하며 적중률이 아님. 실측은 A 강우량과 B 1시간 강수량의 정의 차이가 있음.')


def overview(pairs, sites):
    s = pd.DataFrame(TABLES['summary'])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, col, title in zip(axes, ['median_pair_pearson', 'median_pair_within_pearson'], ['전체 관계', '쌍/월/시간 평균을 뺀 관계']):
        mat = s.loc[~s.variable.eq('direction')].pivot(index='variable', columns='dataset', values=col)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r'); ax.set_title(title)
        ax.set_yticks(range(len(mat)), [META[v][0] for v in mat.index]); ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=35, ha='right', fontsize=8)
        for i in range(len(mat)):
            for j in range(len(mat.columns)):
                if pd.notna(mat.iloc[i, j]): ax.text(j, i, f'{mat.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
    save_plot(fig, OUT, GALLERY, 'overview', 'A/B 지점 쌍 관계의 기간과 자료별 비교',
              '실측은 2024년, 예보는 최근/평가의 각 1년. 표본과 기간이 달라 실측 대비 예보 성능 비교가 아님. 흰 셀은 해당 변수 없음.')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].hist(pairs.plants, bins=np.arange(.5, pairs.plants.max() + 1.5)); axes[0].set_xlabel('한 A/B 쌍에 연결된 발전소 수'); axes[0].set_ylabel('쌍 수')
    axes[1].hist(pairs.groupby('wa_station').size(), bins=np.arange(.5, pairs.groupby('wa_station').size().max() + 1.5)); axes[1].set_xlabel('A 지점 하나에 연결된 B 지점 수')
    axes[2].hist(pairs.groupby('wb_station').size(), bins=np.arange(.5, pairs.groupby('wb_station').size().max() + 1.5)); axes[2].set_xlabel('B 지점 하나에 연결된 A 지점 수')
    save_plot(fig, OUT, GALLERY, 'mapping', '발전소 253개에서 만들어진 A/B 지점 연결',
              '메타 매핑의 고유 87쌍을 사용. 쌍이 달라도 A 또는 B 지점을 공유하므로 서로 독립인 87개 실험이 아님.')


def number(value, digits=3):
    return f'{value:.{digits}f}' if pd.notna(value) else '정의 안 됨'


def report():
    s = pd.DataFrame(TABLES['summary']); a = pd.DataFrame(TABLES['audit']); q = pd.DataFrame(TABLES['quality'])
    e = pd.DataFrame(TABLES['events']); w = pd.DataFrame(TABLES['wind'])
    def row(dataset, variable):
        return s.loc[s.dataset.eq(dataset) & s.variable.eq(variable)].iloc[0]
    recent = {v: row('최근 예보 17', v) for v in META}
    test = {v: row('평가 예보 17', v) for v in META}
    monthly = pd.DataFrame(TABLES['monthly'])
    temp_month = monthly.loc[monthly.dataset.eq('최근 예보 17') & monthly.variable.eq('temp')].groupby('month').mean_difference.mean()
    rain_events = e.loc[e.variable.eq('rain') & e.threshold.eq(.1)].set_index('dataset')
    findings = [
        '## 주요 결과의 해석', '',
        f"기온은 서로 비슷하게 움직였다. 최근/평가 17시 예보의 쌍별 편차 상관 중앙값은 {recent['temp'].median_pair_within_pearson:.3f}/{test['temp'].median_pair_within_pearson:.3f}였다. 다만 쌍 균등 평균 절대 차이는 {recent['temp'].mean_absolute_difference:.3f}/{test['temp'].mean_absolute_difference:.3f}°C로 0이 아니었다. 같은 값을 복사한 자료로 볼 수 없다.", '',
        f"전체 평균의 작은 차이는 월별 차이를 가린다. 최근 17시 기온의 연평균 A-B는 {recent['temp'].mean_difference:+.3f}°C였지만, 쌍별 월평균 차이의 평균은 {temp_month.idxmin()}월 {temp_month.min():+.3f}°C에서 {temp_month.idxmax()}월 {temp_month.max():+.3f}°C까지 방향이 달랐다. 하나의 고정 보정값을 적용할 근거로 사용하지 않았다.", '',
        f"습도는 편차 상관 중앙값 {recent['humidity'].median_pair_within_pearson:.3f}→{test['humidity'].median_pair_within_pearson:.3f}였지만 평균 절대 차이는 {recent['humidity'].mean_absolute_difference:.2f}→{test['humidity'].mean_absolute_difference:.2f}%p였다. 운량은 각각 {recent['cloud'].median_pair_within_pearson:.3f}→{test['cloud'].median_pair_within_pearson:.3f}, {recent['cloud'].mean_absolute_difference:.2f}→{test['cloud'].mean_absolute_difference:.2f}%p로, 기온보다 편차 상관이 낮았다. 단위가 다른 변수 사이의 절대 차이를 직접 비교한 결론은 아니다.", '',
        f"풍속과 풍향은 평가 구간에서 서로 더 가까운 모양이었다. 17시 풍속의 편차 상관 중앙값은 {recent['speed'].median_pair_within_pearson:.3f}→{test['speed'].median_pair_within_pearson:.3f}, 평균 절대 차이는 {recent['speed'].mean_absolute_difference:.3f}→{test['speed'].mean_absolute_difference:.3f} m/s였다. 풍향 평균 절대 차이도 {recent['direction'].mean_absolute_difference:.2f}→{test['direction'].mean_absolute_difference:.2f}도로 달랐다. 두 기간의 실제 날씨와 자료 생성 방식 차이를 분리하지 못했으므로 정확도 향상이나 공급사 통합으로 해석하지 않는다.", '',
        f"강우량은 최근/평가 17시 예보에서 함께 0인 비중이 {100*recent['rain'].pair_both_zero_share:.2f}%/{100*test['rain'].pair_both_zero_share:.2f}%였지만, 한쪽 이상이 0.1 mm를 초과하는 경우의 동시 초과 비율은 {100*rain_events.loc['최근 예보 17','pair_balanced_union_overlap']:.2f}%/{100*rain_events.loc['평가 예보 17','pair_balanced_union_overlap']:.2f}%였다. 양쪽 모두 양수인 행의 Spearman도 {recent['rain'].both_positive_spearman:.3f}/{test['rain'].both_positive_spearman:.3f}였다. 무강수의 큰 공통 비중과 강수 시의 불일치는 구분해야 한다.", '',
        f"강설은 함께 0인 비중이 {100*recent['snow'].pair_both_zero_share:.2f}%/{100*test['snow'].pair_both_zero_share:.2f}%였고 양쪽 양수 행은 {fmt(recent['snow'].both_positive_n)}/{fmt(test['snow'].both_positive_n)}개뿐이었다. 드문 양수값, 단위 환산 후 크기와 정의 미확정을 함께 고려해야 하며 높은 전체 일치율로 대체 가능성을 판단하지 않는다.", '']
    text = ['# A/B 기상 지점 쌍의 관계 EDA', '', f"작성: {INFO['finished_at']}. 실제 계산 완료.", '',
        '## 질문과 목적', '', '같은 발전소에 매핑된 A와 B 기상 자료가 얼마나 같은 움직임을 보이는지, 값의 차이가 계절과 시간 패턴을 구분한 뒤에도 남는지 확인했다. A-B의 차이는 서로 다른 지점과 자료 사이의 불일치이며 예보 오차나 제공사 우열이 아니다.', '',
        '## 사용 범위와 수행 방법', '',
        f"sites.csv의 253개 발전소가 사용하는 A 61개/B 73개 지점을 고유 조합 {INFO['pairs']}쌍으로 묶었다. 한 조합은 최대 {INFO['max_plants_per_pair']}개 발전소가 공유한다. 발전소 수로 같은 날씨를 복제하지 않고 쌍별 통계의 평균/중앙값을 주 결과로 사용했다. 같은 A 또는 B 지점을 공유하는 쌍들은 서로 독립이 아니다.", '',
        '- 예보는 최근 1년 2024-06~2025-05와 평가 1년 2025-06~2026-05를 각각 10시/17시 라운드로 구분했다. 실측은 별도의 2024년 전체를 참고 비교했다. 평가 구간의 실측은 없으며 추정해서 채우지 않았다.',
        '- 시간은 KST, ts - 1시간의 구간 시작 기준이다. 같은 쌍과 대상 ts만 비교하고 자료별 존재 키를 전체 시간 격자와 대조했다. 없는 기록을 0으로 채우지 않고 비정시를 반올림하지 않았다.',
        '- 풍속 B는 km/h를 3.6으로 나눠 m/s, 예보 강설 B는 cm를 10배해 mm로 맞췄다. 기온/습도/운량/풍속/풍향과 강우/강설을 비교했다. A 강우량과 B 실측 1시간 강수량은 정의가 달라 참고로만 보며, 강설도 적설 깊이와 물환산 등의 세부 정의가 같다고 확인하지 않았다.',
        '- A 지표기압과 B 실측 기압은 기준 정의를 같다고 확인하지 못했고 B 예보에는 대응 열이 없어 직접 비교하지 않았다. 최대/최저기온과 체감온도 등을 이름만으로 서로 대응시키지 않았다.',
        '- 변수별 유한한 원본 쌍의 민감도를 먼저 남기고, 주 비교에서는 운량/습도 0~100, 풍향 0~360도, A 풍속 0~50 m/s와 B 0~200 km/h, 강우/강설 0 이상을 사용했다. 임시 비교 범위이며 최종 삭제 규칙이 아니다. 중복 실측 키는 비교하는 열에서 값 충돌 또는 결측/비결측 충돌이 있으면 임시 NaN으로 두었다.',
        '- 지점 쌍별 전체 Pearson/Spearman, 시작 9~16시 Pearson, 같은 쌍/월/시간의 평균을 뺀 편차 Pearson을 계산했다. 평균 편차는 모형 적합이나 완전한 교란 통제가 아니다. 상관이 정의되지 않는 상수 쌍 수를 구분했다.',
        '- 평균 절대 차이는 쌍마다 계산한 뒤 같은 비중으로 평균했다. 발전소 수 가중과 행 가중 결과도 민감도로 남겼다. 회귀, 가설검정, p값, 변수 제거와 예측은 수행하지 않았다.',
        '- 풍향은 (A-B+180)%360-180의 최단 각도 차이와 cos 차이를 사용하고 원각도 Pearson은 계산하지 않았다. 양쪽 풍속이 모두 1 m/s를 초과하는 경우를 별도로 확인했다. 이 기준은 설명용이며 고요한 바람의 확정 정의가 아니다.',
        '- 강우/강설은 >0, >0.1, >1 mm의 네 가지 공동 상태와 동시 초과/한쪽 이상 초과 비율을 대칭적으로 비교했다. 함께 0인 비중과 양쪽 양수일 때의 Spearman도 구분했다.', '',
        '## 연결 키와 비교 표본', '', md_table(['자료', '예상 쌍-시각', '양쪽 키', 'A만', 'B만', '모두 없음'],
        [[r.dataset, fmt(r.expected), fmt(r.both_keys), fmt(r.only_a), fmt(r.only_b), fmt(r.neither_key)] for r in a.itertuples()]), '',
        '전체 격자와 실제 키의 차이는 변수 값 NaN과 다르다. 같은 A 결손이 여러 B 매핑 쌍에 걸치면 쌍-시각 결손 수는 원본 A 결손 행 수보다 많을 수 있다.', '',
        *findings,
        '## 쌍별 관계와 값의 차이', '', md_table(['자료', '변수', '유효 행', '쌍 수', '전체 r 중앙값', '편차 r 중앙값', '편차 정의 쌍', '평균 A-B', '평균 절대 차이', '단위'],
        [[r.dataset, META[r.variable][0], fmt(r.n), r.pairs, number(r.median_pair_pearson), number(r.median_pair_within_pearson),
          r.within_pairs_defined, number(r.mean_difference), number(r.mean_absolute_difference), r.unit] for r in s.itertuples()]), '',
        '전체 상관과 달력 평균을 뺀 관계를 함께 읽는다. 상관이 높아도 상수 차이와 값의 산포가 남을 수 있으므로 완전 중복이나 대체 가능성으로 확정하지 않는다. 서로 다른 변수의 절대 차이 숫자는 단위가 다르므로 우열 순서로 나열하지 않는다.', '',
        '## 변수별 원본 유한값과 비교 범위의 민감도', '', md_table(['자료', '변수', '유한 쌍', '범위 제외', '유한값 행 평균 절대차', '범위 통과 행 평균 절대차'],
        [[r.dataset, META[r.variable][0], fmt(r.finite_pairs), fmt(r.range_excluded), number(r.finite_row_mean_abs_difference),
          number(s.loc[s.dataset.eq(r.dataset) & s.variable.eq(r.variable), 'row_mean_absolute_difference'].iloc[0])] for r in q.itertuples()]), '',
        '이 표는 범위 점검의 영향만 보려고 행 가중을 동일하게 사용했다. 위 주 결과의 쌍 균등 평균과는 가중이 다르다. 원본 값은 수정하지 않았다.', '',
        '## 풍향과 풍속 조건', '', md_table(['자료', '조건', '행', '쌍 수', '쌍 균등 절대 각도 차이 °', 'cos 차이 평균', '30도 이내 %'],
        [[r.dataset, r.scope, fmt(r.n), r.pairs, number(r.mean_absolute_angle), number(r.mean_cosine), number(100*r.within30_share, 2)] for r in w.itertuples()]), '',
        'cos 차이는 +1이면 같은 방향, -1이면 반대 방향이다. 부분집합은 대상 날씨와 표본이 달라지므로 풍속 기준을 적용하면 정보가 좋아진다는 성능 주장은 하지 않는다.', '',
        '## 강수의 공통 0과 공동 초과', '', md_table(['자료', '변수', '기준 mm', '둘 다 비초과 %', 'A만 %', 'B만 %', '동시 초과 %', '동시/합집합 %'],
        [[r.dataset, META[r.variable][0], r.threshold, number(100*r.neither_pair_share, 2), number(100*r.a_only_pair_share, 2),
          number(100*r.b_only_pair_share, 2), number(100*r.both_pair_share, 2), number(100*r.pair_balanced_union_overlap, 2)] for r in e.itertuples()]), '',
        md_table(['자료', '변수', '함께 0 %', '양쪽 양수 행', '양쪽 양수 Spearman'],
        [[r.dataset, META[r.variable][0], number(100*r.pair_both_zero_share, 2), fmt(r.both_positive_n), number(r.both_positive_spearman)] for r in s.loc[s.variable.isin(['rain', 'snow'])].itertuples()]), '',
        '초과는 엄격한 > 비교다. 합집합 비중이 0이면 겹침 비율을 정의하지 않는다. 공통 0이 많다는 사실만으로 강수 상황에서 같은 정보를 준다고 결론내리지 않는다.', '',
        '## 가중 민감도', '', md_table(['자료', '변수', '쌍 균등 평균 절대차', '연결 발전소 수 가중', '쌍별 절대차 10~90백분위'],
        [[r.dataset, META[r.variable][0], number(r.mean_absolute_difference), number(r.plant_count_weighted_mad),
          number(r.pair_mad_p10) + ' ~ ' + number(r.pair_mad_p90)] for r in s.itertuples()]), '',
        '발전소 수 가중은 현재 메타의 연결 수를 반영한 설명용 민감도다. 발전소의 관측 가용성이나 설비용량 가중을 반영한 결과는 아니다.', '',
        '## 한계와 다음 단계', '',
        '서로 다른 지점의 날씨와 제공사의 측정/가공 방식 차이를 분리하지 못한다. 메타에는 발전소와 각 지점까지의 거리만 있어 A 지점과 B 지점 사이의 실제 거리를 추정하지 않았다. 강수/강설 정의의 완전한 일치도 확인하지 못했다. 상관이 낮다는 사실은 고유한 유용 정보가 있다는 증거가 아니며, 높다는 사실은 한 공급사를 제거해도 된다는 증거가 아니다.', '',
        '이번에 남아 있던 A/B 매핑 지점 쌍의 직접 비교를 실행했다. 특이 패턴의 생성 원인과 변수의 실제 추가 설명력은 여전히 미확정이다. 이 결과로 최종 전처리, 변수 선택이나 모델링을 시작하지 않았다. 다음 EDA는 남은 질문의 중요도를 다시 정해 진행한다.', '',
        '## 관련 코드와 로컬 결과', '', '- [분석 코드](../scripts/weather_provider_pairs_eda.py)',
        '- [전체 그림](../data/eda/2026-10-19_weather_provider_pairs/index.html)',
        '- [쌍별 집계](../data/eda/2026-10-19_weather_provider_pairs/pair_metrics.csv)',
        '- [월별 집계](../data/eda/2026-10-19_weather_provider_pairs/monthly.csv)', '',
        '원본 또는 전처리 데이터 행을 보고서에 복사하지 않았다. 모든 데이터 출력과 그림은 로컬 data/에만 보관한다.', '']
    if 'verification' in INFO:
        position = text.index('## 한계와 다음 단계')
        text[position:position] = ['## 실행과 검증', '',
            f"정규 CPython 3.13으로 계산과 검증을 실행했다. 자료 5종, 변수 조합 34개, 쌍별 조합 2,958개와 월별 쌍 집계 20,880개, 강수 기준 조합 27개를 확인했다. 키의 네 가지 상태 합계가 전체 격자와 같고, 쌍별 평균과 월별 가중 재집계가 일치했다. 풍향의 0/360도 경계와 원본 8개 파일의 SHA-256도 검사했다. 이 8개에는 단위 확인용 README.md와 데이터 7개가 포함된다. 그림 22개와 갤러리 링크 44개를 검사했다. 검증 시각은 {INFO['verified_at']}이며 상세 기록은 로컬 verification.json에 있다.", '']
    REPORT.write_text('\n'.join(text), encoding='utf-8')


def verify():
    info = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    data = {k: pd.read_csv(OUT/(k+'.csv'), encoding='utf-8-sig') for k in TABLES}
    s, p, a, e, q, m = [data[k] for k in ['summary', 'pair_metrics', 'audit', 'events', 'quality', 'monthly']]
    assert len(s) == 34 and len(a) == 5 and len(e) == 27
    assert a[['both_keys', 'only_a', 'only_b', 'neither_key']].sum(axis=1).eq(a.expected).all()
    assert a.expected.eq(a.pairs * a.hours).all() and a.pairs.eq(87).all()
    assert (q.finite_pairs - q.comparison_pairs).eq(q.range_excluded).all()
    for row in s.itertuples():
        z = p.loc[p.dataset.eq(row.dataset) & p.variable.eq(row.variable)]
        assert z.n.sum() == row.n and len(z) == row.pairs
        assert np.isclose(z.mean_absolute_difference.mean(), row.mean_absolute_difference)
        assert np.isclose(np.average(z.mean_absolute_difference, weights=z.n), row.row_mean_absolute_difference)
        assert np.allclose(z.root_mean_square_difference.ge(z.mean_absolute_difference - 1e-9), True)
        assert np.isclose(z.mean_difference.mean(), row.mean_difference)
        for col in ['pearson', 'spearman', 'within_pearson', 'day_pearson']:
            assert (z[col].isna() | z[col].abs().le(1.000001)).all()
        if row.variable in LINEAR:
            months = m.loc[m.dataset.eq(row.dataset) & m.variable.eq(row.variable)]
            assert months.n.sum() == row.n and len(months) == 87 * 12
            reaggregated = months.assign(total=months.n * months.mean_difference).groupby('pair_id').agg(n=('n','sum'), total=('total','sum'))
            check = z.set_index('pair_id')
            assert np.allclose(reaggregated.total/reaggregated.n, check.mean_difference.reindex(reaggregated.index))
    assert e[['neither', 'a_only', 'b_only', 'both']].sum(axis=1).eq(e.n).all()
    assert np.allclose(e[['neither_pair_share', 'a_only_pair_share', 'b_only_pair_share', 'both_pair_share']].sum(axis=1), 1)
    assert e.pair_balanced_union_overlap.dropna().between(0, 1).all()
    assert np.array_equal(wrapped([359, 1, 360], [1, 359, 0]), [-2, 2, 0])
    assert np.isclose(3.6/3.6, 1) and np.isclose(.1*10, 1)
    assert info['sha256'] == {f: digest(SOURCE/f) for f in FILES}
    assert '\u00b7' not in REPORT.read_text(encoding='utf-8')
    result = {**check_gallery(OUT, 22), 'datasets': len(a), 'variable_combinations': len(s), 'pair_variable_combinations': len(p),
              'monthly_pair_combinations': len(m), 'event_threshold_combinations': len(e), 'join_accounting': True,
              'pair_weight_and_month_accounting': True, 'circular_angle_boundary_checked': True, 'source_unchanged': True}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    assert sys.version_info[:2] == (3, 13)
    OUT.mkdir(parents=True, exist_ok=True)
    if '--verify-only' in sys.argv:
        verify(); return
    if '--report-only' in sys.argv:
        INFO.update(json.loads((OUT/'summary.json').read_text(encoding='utf-8')))
        for k in TABLES: TABLES[k].extend(pd.read_csv(OUT/(k+'.csv'), encoding='utf-8-sig').to_dict('records'))
        report(); return
    before = {f: digest(SOURCE/f) for f in FILES}
    sites = pd.read_csv(SOURCE/'sites.csv', encoding='utf-8-sig')
    pairs = sites.groupby(['wa_station', 'wb_station'], observed=True).size().rename('plants').reset_index()
    pairs['pair_id'] = np.arange(len(pairs))
    assert len(sites) == 253 and len(pairs) == 87 and pairs.plants.sum() == 253
    pairs.to_csv(OUT/'mapping.csv', index=False, encoding='utf-8-sig')
    INFO.update(pairs=len(pairs), max_plants_per_pair=int(pairs.plants.max()))
    configs = [('train', 'observed', '2024-01-01', '2025-01-01', 0, '2024 실측', 'observed')]
    configs += [(split, 'forecast', lo, hi, r, name + f' 예보 {r}', f'{split}_{r}')
                for split, lo, hi, name in [('train','2024-06-01','2025-06-01','최근'), ('test','2025-06-01','2026-06-01','평가')]
                for r in [10, 17]]
    for split, kind, lo, hi, r, label, tag in configs:
        frame = joined(pairs, split, kind, lo, hi, r, label)
        analyze(frame, label, kind, tag)
        del frame
        gc.collect()
    overview(pairs, sites)
    assert before == {f: digest(SOURCE/f) for f in FILES}
    INFO.update(sha256=before, finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes'))
    for k, rows in TABLES.items(): pd.DataFrame(rows).to_csv(OUT/(k+'.csv'), index=False, encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(INFO, ensure_ascii=False, indent=2), encoding='utf-8')
    write_gallery(OUT, GALLERY, 'A/B 기상 지점 쌍의 관계 EDA')
    report()
    INFO['verification'] = verify()
    INFO['verified_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    (OUT/'summary.json').write_text(json.dumps(INFO, ensure_ascii=False, indent=2), encoding='utf-8')
    report()
    log(str(REPORT))


if __name__ == '__main__':
    main()
