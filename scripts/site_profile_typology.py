"""DA-002: seasonal daily-shape typology, without generation forecasting."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import sysconfig
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log, valid_values

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA/01_participant_dataset'
OUT = ROOT / 'data/analysis/2026-10-21_site_typology'
REPORT = ROOT / 'reports/2026-10-21_08_발전소별_발전패턴_유형_분석.md'
FILES = ['sites.csv', 'train/generation.parquet', 'train/observed_wb.parquet']
SEED = 2026102102
BOOT_REPS = 200
NULL_REPS = 199
HOURS = np.arange(7, 18)
FIT_COUNTS = defaultdict(int)


def seasonal_profiles(cube, quarters, weights=None, condition=None):
    """Daily cube -> site/quarter/hour means and unit-sum seasonal shapes."""
    if weights is None:
        weights = np.ones(len(cube))
    if condition is None:
        condition = np.ones(cube.shape[:2], dtype=bool)
    raw, counts = [], []
    for q in range(1, 5):
        take = quarters == q
        w = weights[take, None] * condition[take]
        n = w.sum(axis=0)
        if (n <= 0).any():
            raise ValueError('quarter/site has no weighted dates')
        mean = np.einsum('ds,dsh->sh', w, cube[take], optimize=True) / n[:, None]
        if (mean.sum(axis=1) <= 1e-8).any():
            raise ValueError('quarter profile has no positive generation mass')
        raw.append(mean)
        counts.append(n)
    raw = np.stack(raw, axis=1)
    shapes = raw / raw.sum(axis=2, keepdims=True)
    features = shapes.reshape(len(shapes), -1) / 2
    return features, shapes, raw, np.stack(counts, axis=1)


def minimum_size(n):
    return max(5, int(np.ceil(.1 * n)))


def fit_partition(x, k, rng, phase):
    seed = int(rng.integers(0, 2**31 - 1))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        fit = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=300,
                     tol=1e-4, algorithm='lloyd', random_state=seed).fit(x)
    FIT_COUNTS[phase] += 1
    labels = fit.labels_.copy()
    centers = fit.cluster_centers_.copy()
    # Names C1... follow mean time center, without implying equipment orientation.
    centers_3d = centers.reshape(k, 4, len(HOURS))
    mass = centers_3d.sum(axis=(1, 2))
    clock = (centers_3d * HOURS).sum(axis=(1, 2)) / mass
    order = np.argsort(clock)
    inverse = np.empty(k, dtype=int)
    inverse[order] = np.arange(k)
    labels = inverse[labels]
    centers = centers[order]
    sizes = np.bincount(labels, minlength=k)
    unique = len(np.unique(labels))
    sil = float(silhouette_score(x, labels, metric='euclidean')) if 1 < unique < len(x) else None
    return {'k': k, 'seed': seed, 'labels': labels.tolist(), 'centers': centers.tolist(),
            'sizes': sizes.tolist(), 'silhouette': sil, 'inertia': float(fit.inertia_),
            'admissible': bool(unique == k and sizes.min() >= minimum_size(len(x))),
            'n_iter': int(fit.n_iter_), 'warnings': [str(w.message) for w in recorded]}


def choose_partition(results, admissible=True):
    possible = [r for r in results if r['silhouette'] is not None and (r['admissible'] or not admissible)]
    if not possible:
        return None
    best = max(r['silhouette'] for r in possible)
    return min((r for r in possible if best - r['silhouette'] <= 1e-12), key=lambda r: r['k'])


def align_labels(reference, candidate, k):
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (reference, candidate), 1)
    row, col = linear_sum_assignment(-counts)
    mapping = np.empty(k, dtype=int)
    mapping[col] = row
    return mapping[np.asarray(candidate)]


def gaussian_reference(x, rng):
    center = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - center, full_matrices=False)
    keep = singular > max(singular[0] * 1e-12, 1e-14)
    draw = rng.normal(size=(len(x), int(keep.sum())))
    return center + (draw * singular[keep] / np.sqrt(len(x) - 1)) @ vt[keep]


def block_weights(n, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / 7)))
    indices = ((starts[:, None] + np.arange(7)) % n).ravel()[:n]
    return np.bincount(indices, minlength=n)


def load_year(gen, sites, weather, ids, year):
    meta = sites.set_index('site_id').loc[ids]
    start = gen.ts - pd.Timedelta(hours=1)
    d = gen.loc[start.dt.year.eq(year) & start.dt.hour.between(7, 17) & gen.site_id.isin(ids)].copy()
    d['capacity'] = d.site_id.map(meta.capacity_kw)
    valid = np.isfinite(d.energy) & d.energy.ge(0) & np.isfinite(d.capacity) & d.capacity.gt(0)
    valid &= d.ts.eq(d.ts.dt.floor('h'))
    d['valid'] = valid
    d['cf'] = d.energy / d.capacity
    t = d.ts - pd.Timedelta(hours=1)
    d['date'] = t.dt.floor('D')
    d['hour'] = t.dt.hour
    daily = d.groupby('date').agg(rows=('ts', 'size'), good=('valid', 'sum'))
    dates = daily.index[daily.rows.eq(len(ids) * len(HOURS)) & daily.good.eq(len(ids) * len(HOURS))].sort_values()
    quarter_counts = pd.Series(dates.quarter).value_counts().reindex(range(1, 5), fill_value=0)
    if quarter_counts.min() < 28:
        raise ValueError('fewer than 28 balanced dates in a quarter')
    kept = d.loc[d.date.isin(dates)].sort_values(['date', 'site_id', 'hour'])
    assert not kept.duplicated(['site_id', 'ts']).any()
    assert kept.groupby(['date', 'site_id']).hour.nunique().eq(len(HOURS)).all()
    cube = kept.cf.to_numpy().reshape(len(dates), len(ids), len(HOURS))
    s = weather.ts - pd.Timedelta(hours=1)
    w = weather.loc[s.dt.year.eq(year) & s.dt.hour.between(9, 16) & weather.wb_station.isin(meta.wb_station)].copy()
    s = w.ts - pd.Timedelta(hours=1)
    w['date'] = s.dt.floor('D')
    w['valid'] = valid_values(w.wb_cloud, 'wb_cloud') & valid_values(w.wb_precip_1h, 'wb_precip_1h')
    w['valid'] &= w.ts.eq(w.ts.dt.floor('h'))
    day = w.groupby(['date', 'wb_station']).agg(n=('ts', 'size'), good=('valid', 'sum'),
        cloud=('wb_cloud', 'mean'), rain=('wb_precip_1h', 'max'))
    conditions = {}
    for threshold in [25, 50]:
        flag = day.n.eq(8) & day.good.eq(8) & day.cloud.le(threshold) & day.rain.le(.1)
        frame = flag.unstack('wb_station').reindex(index=dates, columns=meta.wb_station.tolist()).fillna(False)
        conditions[threshold] = frame.to_numpy(dtype=bool)
    audit = {'year': year, 'sites': len(ids), 'stations': int(meta.wb_station.nunique()),
             'candidate_rows': len(d), 'balanced_rows': len(kept), 'balanced_dates': len(dates),
             'excluded_dates': len(daily) - len(dates), 'dates_per_quarter': quarter_counts.tolist(),
             'zero_rows': int((cube == 0).sum()), 'above_one_rows': int((cube > 1).sum())}
    return cube, np.asarray(dates.quarter), conditions, audit


def station_diagnostics(features, labels, stations, rng):
    labels, stations = np.asarray(labels), np.asarray(stations)
    i, j = np.triu_indices(len(labels), 1)
    same_station = stations[i] == stations[j]
    same_label = labels[i] == labels[j]
    permutations = []
    for _ in range(NULL_REPS):
        permuted = rng.permutation(labels)
        permutations.append(float((permuted[i[same_station]] == permuted[j[same_station]]).mean()))
    centered = features - features.mean(axis=0)
    residual = features.copy()
    for station in np.unique(stations):
        take = stations == station
        residual[take] -= features[take].mean(axis=0)
    return {'ami': float(adjusted_mutual_info_score(stations, labels)),
            'same_station_pairs': int(same_station.sum()), 'different_station_pairs': int((~same_station).sum()),
            'same_label_given_same_station': float(same_label[same_station].mean()),
            'same_label_given_different_station': float(same_label[~same_station].mean()),
            'permutation_same_station_quantiles': np.quantile(permutations, [.025, .5, .975]).tolist(),
            'within_station_variance_fraction': float((residual ** 2).sum() / (centered ** 2).sum()),
            'stations_with_multiple_labels': int(sum(len(np.unique(labels[stations == st])) > 1 for st in np.unique(stations)))}


def run_analysis():
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Completed results exist; no automatic refitting.')
    inspection = json.loads((OUT / 'inspection.json').read_text(encoding='utf-8'))
    ids = inspection['site_ids']
    before = {f: digest(SOURCE / f) for f in FILES}
    (OUT / 'preregistered_report.md').write_text(REPORT.read_text(encoding='utf-8'), encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': before, 'code_hash': digest(__file__),
         'report_hash': digest(REPORT), 'seed': SEED, 'bootstrap_reps': BOOT_REPS, 'null_reps': NULL_REPS})
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet')
    assert not gen.duplicated(['site_id', 'ts']).any()
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig')
    weather = pd.read_parquet(SOURCE / 'train/observed_wb.parquet', columns=['wb_station', 'ts', 'wb_cloud', 'wb_precip_1h'])
    assert not weather.duplicated(['wb_station', 'ts']).any()
    meta = sites.set_index('site_id').loc[ids]
    rng = np.random.default_rng(SEED)
    summary = {'audit': {}, 'source_hashes': before, 'code_hash': digest(__file__), 'site_ids': ids,
               'python': sys.version, 'sklearn': sklearn.__version__, 'seed': SEED}
    panels, data = {}, {}
    for year in [2024, 2023]:
        cube, quarters, conditions, audit = load_year(gen, sites, weather, ids, year)
        features, shapes, raw, counts = seasonal_profiles(cube, quarters)
        panels[year] = (cube, quarters, conditions)
        data[year] = (features, shapes, raw)
        summary['audit'][str(year)] = audit
        np.savez_compressed(OUT / f'panel_{year}.npz', cf=cube, quarters=quarters,
            condition25=conditions[25], condition50=conditions[50], profiles=shapes, raw_profiles=raw,
            features=features, site_ids=np.asarray(ids), stations=meta.wb_station.to_numpy(dtype=str),
            capacities=meta.capacity_kw.to_numpy())
    x, shapes, raw = data[2024]
    with threadpool_limits(limits=1):
        log('DA-002: compare k=2..5')
        candidates = [fit_partition(x, k, rng, 'candidates') for k in range(2, 6)]
        chosen = choose_partition(candidates)
        diagnostic = chosen or choose_partition(candidates, admissible=False)
        if diagnostic is None:
            raise ValueError('no nontrivial partition even for diagnostics')
        k = diagnostic['k']
        labels = np.asarray(diagnostic['labels'])
        summary.update(candidates=candidates, selected_k=chosen['k'] if chosen else None,
                       evaluation_k=k, diagnostic_partition=diagnostic)
        null_max, null_negative = [], []
        for b in range(NULL_REPS):
            simulated = gaussian_reference(x, rng)
            null_negative.append(float((simulated < 0).mean()))
            fits = [fit_partition(simulated, q, rng, 'gaussian_reference') for q in range(2, 6)]
            best = choose_partition(fits)
            null_max.append(best['silhouette'] if best else np.nan)
            if b % 50 == 49:
                log(f'Gaussian reference {b + 1}/{NULL_REPS}')
        valid_null = np.isfinite(null_max)
        threshold = float(np.nanquantile(null_max, .95))
        summary['reference'] = {'replicates': NULL_REPS, 'valid': int(valid_null.sum()),
            'silhouette_q05_median_q95': np.nanquantile(null_max, [.05, .5, .95]).tolist(),
            'negative_feature_share_mean': float(np.mean(null_negative)),
            'separation_pass': bool(chosen and valid_null.mean() >= .95 and chosen['silhouette'] > threshold)}
        replica = fit_partition(data[2023][0], k, rng, 'replication_2023')
        replica['ari'] = float(adjusted_rand_score(labels, replica['labels']))
        replica['aligned_labels'] = align_labels(labels, replica['labels'], k).tolist()
        summary['replication_2023'] = replica
        cube, quarters, _ = panels[2024]
        ari, size_pass, all_aligned = [], [], []
        failures = defaultdict(int)
        first_weights = None
        for b in range(BOOT_REPS):
            w = block_weights(len(cube), rng)
            if b == 0:
                first_weights = w
            try:
                f, _, _, _ = seasonal_profiles(cube, quarters, weights=w)
                part = fit_partition(f, k, rng, 'bootstrap')
                ari.append(float(adjusted_rand_score(labels, part['labels'])))
                size_pass.append(part['admissible'])
                all_aligned.append(align_labels(labels, part['labels'], k))
            except ValueError as exc:
                failures[str(exc)] += 1
            if b % 50 == 49:
                log(f'Date block {b + 1}/{BOOT_REPS}')
        aligned = np.asarray(all_aligned)
        coassignment = np.mean(aligned[:, :, None] == aligned[:, None, :], axis=0)
        confidence = (aligned == labels[None, :]).mean(axis=0)
        summary['bootstrap'] = {'attempts': BOOT_REPS, 'success': len(ari), 'failures': dict(failures),
            'ari_q05_median_q95': np.quantile(ari, [.05, .5, .95]).tolist(),
            'minimum_size_pass_fraction': float(np.mean(size_pass)),
            'assignment_agreement_q05_median_q95': np.quantile(confidence, [.05, .5, .95]).tolist(),
            'sites_below_08_agreement': int((confidence < .8).sum())}
        eligible = np.ones(len(ids), dtype=bool)
        strong_eligible = np.ones(len(ids), dtype=bool)
        for year in [2024, 2023]:
            _, q, cond = panels[year]
            for quarter in range(1, 5):
                eligible &= cond[50][q == quarter].sum(axis=0) >= 10
                strong_eligible &= cond[25][q == quarter].sum(axis=0) >= 10
        weather_comparison = {'strong_condition_eligible': int(strong_eligible.sum()),
            'moderate_condition_eligible': int(eligible.sum()), 'comparison_available': False}
        saved_weather = {}
        if eligible.sum() >= max(20, 5 * k):
            baseline = labels[eligible] if eligible.all() else fit_partition(x[eligible], k, rng, 'weather_baseline_subset')['labels']
            conditioned = {}
            for year in [2024, 2023]:
                cube_y, q, cond = panels[year]
                feat, sh, _, counts = seasonal_profiles(cube_y[:, eligible], q, condition=cond[50][:, eligible])
                part = fit_partition(feat, k, rng, 'weather_condition')
                conditioned[year] = part
                saved_weather[f'weather_features_{year}'] = feat
                saved_weather[f'weather_shapes_{year}'] = sh
                saved_weather[f'weather_counts_{year}'] = counts
            ari_all = float(adjusted_rand_score(baseline, conditioned[2024]['labels']))
            ari_year = float(adjusted_rand_score(conditioned[2024]['labels'], conditioned[2023]['labels']))
            weather_comparison.update(comparison_available=True, full_vs_moderate_ari=ari_all,
                moderate_annual_ari=ari_year, moderate_2024=conditioned[2024], moderate_2023=conditioned[2023],
                sensitivity_pass=bool(ari_all >= .6 and ari_year >= .6))
        summary['weather_comparison'] = weather_comparison
        summary['station_diagnostics'] = station_diagnostics(x, labels, meta.wb_station.to_numpy(), rng)
        amplitude = fit_partition(raw.reshape(len(raw), -1) / 2, k, rng, 'amplitude_sensitivity')
        summary['amplitude_sensitivity'] = {'partition': amplitude, 'ari_vs_shape': float(adjusted_rand_score(labels, amplitude['labels']))}
    centered = x - x.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vt[:2].T
    summary['pca'] = {'explained_variance_ratio_first_two': (singular[:2] ** 2 / (singular ** 2).sum()).tolist()}
    time_centers = (shapes * HOURS).sum(axis=2).mean(axis=1)
    morning = shapes[:, :, HOURS <= 11].sum(axis=2).mean(axis=1)
    summary['continuous_descriptors'] = {'time_center_q05_median_q95': np.quantile(time_centers, [.05, .5, .95]).tolist(),
        'morning_share_q05_median_q95': np.quantile(morning, [.05, .5, .95]).tolist()}
    summary['cluster_descriptors'] = []
    for c in range(k):
        take = labels == c
        summary['cluster_descriptors'].append({'cluster': c + 1, 'sites': int(take.sum()),
            'time_center_median': float(np.median(time_centers[take])), 'morning_share_median': float(np.median(morning[take])),
            'day_cf_q25_median_q75': np.quantile(panels[2024][0][:, take].mean(axis=(0, 2)), [.25, .5, .75]).tolist(),
            'capacity_kw_median': float(np.median(meta.capacity_kw.to_numpy()[take])),
            'stations': int(meta.wb_station.iloc[np.flatnonzero(take)].nunique())})
    boot = summary['bootstrap']
    time_pass = bool(replica['ari'] >= .6 and replica['admissible'] and boot['success'] >= .9 * BOOT_REPS
                     and boot['ari_q05_median_q95'][0] >= .6 and boot['minimum_size_pass_fraction'] >= .9)
    summary['judgment'] = {'admissible_candidate': chosen is not None,
        'time_stability_pass': time_pass,
        'shape_type_candidate': bool(chosen and summary['reference']['separation_pass'] and time_pass),
        'weather_sensitivity_pass': weather_comparison.get('sensitivity_pass', False)}
    summary['judgment']['restricted_weather_type_candidate'] = bool(summary['judgment']['shape_type_candidate']
        and summary['judgment']['weather_sensitivity_pass'])
    summary['fit_calls'] = dict(FIT_COUNTS)
    summary['finished_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    assert before == {f: digest(SOURCE / f) for f in FILES}
    dump(OUT / 'summary.json', summary)
    np.savez_compressed(OUT / 'results.npz', null_max_silhouette=np.asarray(null_max),
        bootstrap_ari=np.asarray(ari), bootstrap_labels=aligned, bootstrap_size_pass=np.asarray(size_pass),
        coassignment=coassignment, assignment_agreement=confidence, first_boot_weights=first_weights,
        pca_coordinates=coordinates, pca_axes=vt[:2], pca_singular_values=singular,
        weather_eligible=eligible, **saved_weather)
    log('DA-002 calculation complete')


def inspect_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig')
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet', columns=['site_id', 'ts', 'energy'])
    assert not gen.duplicated(['site_id', 'ts']).any()
    start = gen.ts - pd.Timedelta(hours=1)
    counts = {year: gen.loc[start.dt.year.eq(year)].groupby('site_id').size() for year in [2023, 2024]}
    ids = sorted(set(counts[2023].index[counts[2023].eq(8760)]) & set(counts[2024].index[counts[2024].eq(8784)]))
    cohort = sites.loc[sites.site_id.isin(ids)]
    stations = sorted(cohort.wb_station.unique())
    weather = pd.read_parquet(SOURCE / 'train/observed_wb.parquet', columns=['wb_station', 'ts', 'wb_cloud', 'wb_precip_1h'])
    assert not weather.duplicated(['wb_station', 'ts']).any()
    s = weather.ts - pd.Timedelta(hours=1)
    weather = weather.loc[s.dt.year.isin([2023, 2024]) & s.dt.hour.between(9, 16) & weather.wb_station.isin(stations)].copy()
    s = weather.ts - pd.Timedelta(hours=1)
    weather['date'] = s.dt.floor('D')
    weather['year'] = s.dt.year
    weather['quarter'] = s.dt.quarter
    weather['valid'] = valid_values(weather.wb_cloud, 'wb_cloud') & valid_values(weather.wb_precip_1h, 'wb_precip_1h')
    day = weather.groupby(['wb_station', 'year', 'quarter', 'date']).agg(
        n=('ts', 'size'), good=('valid', 'sum'), cloud=('wb_cloud', 'mean'), rain=('wb_precip_1h', 'max'))
    day['condition'] = day.n.eq(8) & day.good.eq(8) & day.cloud.le(25) & day.rain.le(.1)
    counts_weather = day.condition.groupby(['wb_station', 'year', 'quarter']).sum().unstack(['year', 'quarter']).fillna(0)
    eligible_stations = counts_weather.index[counts_weather.min(axis=1).ge(10)]
    info = {'python': platform.python_version(), 'executable': sys.executable, 'sklearn': sklearn.__version__,
            'site_ids': ids, 'sites': len(ids), 'wb_stations': stations,
            'condition': '09-16 mean cloud <=25 and max hourly rain <=0.1, all 8 records valid',
            'weather_eligible_stations': list(eligible_stations),
            'weather_eligible_sites': cohort.loc[cohort.wb_station.isin(eligible_stations), 'site_id'].tolist(),
            'min_low_cloud_days': int(counts_weather.to_numpy().min()),
            'weather_counts': counts_weather.stack([0, 1], future_stack=True).rename('days').reset_index().to_dict('records')}
    day['moderate_condition'] = day.n.eq(8) & day.good.eq(8) & day.cloud.le(50) & day.rain.le(.1)
    moderate = day.moderate_condition.groupby(['wb_station', 'year', 'quarter']).sum().unstack(['year', 'quarter']).fillna(0)
    moderate_stations = moderate.index[moderate.min(axis=1).ge(10)]
    info['moderate_eligible_sites'] = cohort.loc[cohort.wb_station.isin(moderate_stations), 'site_id'].tolist()
    info['moderate_eligible_stations'] = list(moderate_stations)
    info['moderate_counts'] = moderate.stack([0, 1], future_stack=True).rename('days').reset_index().to_dict('records')
    dump(OUT / 'inspection.json', info)
    print(json.dumps({k: v for k, v in info.items() if k not in ['site_ids', 'wb_stations', 'weather_counts',
          'weather_eligible_sites', 'weather_eligible_stations', 'moderate_eligible_sites',
          'moderate_eligible_stations', 'moderate_counts']}, ensure_ascii=False, indent=2))
    print('Eligible weather sites:', len(info['weather_eligible_sites']), flush=True)
    print('Eligible moderate-condition sites:', len(info['moderate_eligible_sites']), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    assert sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var('Py_GIL_DISABLED')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'run'])
    args = parser.parse_args()
    if args.mode == 'inspect':
        inspect_inputs()
    elif args.mode == 'run':
        run_analysis()
