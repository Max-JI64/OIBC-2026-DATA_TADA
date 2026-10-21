"""DA-011: explanatory partial associations of A/B forecasts with generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log, valid_values
from conditional_support_date import prepare, cross, block_weights

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA/01_participant_dataset'
OUT = ROOT / 'data/analysis/2026-10-21_provider_generation_information'
REPORT = ROOT / 'reports/2026-10-21_14_AB잔여변동과_발전량의_관계_분석.md'
FILES = ['sites.csv', 'train/generation.parquet', 'train/forecast_wa.parquet', 'train/forecast_wb.parquet']
DEPENDENCIES = ['scripts/provider_generation_information.py', 'scripts/conditional_support_date.py',
                'scripts/weather_conditional_relationships.py', 'scripts/weather_latent_structure.py']
SEED = 2026102111
REPS = 100
TERMS = ['A_temp10', 'A_humidity10', 'A_cloud10', 'A_wind_ms', 'A_wet', 'A_lograin',
         'B_temp10', 'B_humidity10', 'B_cloud10', 'B_wind_ms', 'B_wet', 'B_lograin', 'B_uv']
PRIMARY = {'A_cloud': 2, 'B_cloud': 8, 'B_uv': 12}


def mapping(sites):
    pairs = sites[['wa_station', 'wb_station']].drop_duplicates().sort_values(['wa_station', 'wb_station']).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(pairs)
    site_pair = index.get_indexer(pd.MultiIndex.from_frame(sites[['wa_station', 'wb_station']]))
    counts = np.bincount(site_pair, minlength=len(pairs))
    a, b = sorted(pairs.wa_station.unique()), sorted(pairs.wb_station.unique())
    graph = np.zeros((len(a), len(b)), dtype=int)
    for r in pairs.itertuples():
        graph[a.index(r.wa_station), b.index(r.wb_station)] = 1
    match = maximum_bipartite_matching(csr_matrix(graph), perm_type='column')
    selected = np.array([match[a.index(r.wa_station)] == b.index(r.wb_station) for r in pairs.itertuples()])
    assert pairs.loc[selected].wa_station.is_unique and pairs.loc[selected].wb_station.is_unique
    return pairs, site_pair, counts, selected


def load_panel(sites, pairs, site_pair, generation, year, round_number):
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', tz='Asia/Seoul')
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for h in range(9, 17)])
    site_times = np.repeat(times, len(sites))
    site_ids = np.tile(sites.site_id, len(times))
    keys = pd.MultiIndex.from_arrays([site_ids, site_times], names=['site_id', 'ts'])
    values = generation.reindex(keys).energy.to_numpy().reshape(len(dates), 8, len(sites)).transpose(0, 2, 1)
    y = values / sites.capacity_kw.to_numpy()[None, :, None]
    good_generation = np.isfinite(y).all(axis=(1, 2)) & (y >= 0).all(axis=(1, 2))
    parts, audits, wet_flags = [], {}, []
    for provider in ['wa', 'wb']:
        key = provider + '_station'
        columns = [provider + '_temp', provider + ('_humidity' if provider == 'wa' else '_rel_hum'),
                   provider + '_cloud', provider + '_wind_speed', provider + '_rain']
        if provider == 'wb':
            columns += ['wb_uv_idx']
        frame = pd.read_parquet(SOURCE / f'train/forecast_{provider}.parquet', columns=[key, 'round', 'ts'] + columns)
        start = frame.ts - pd.Timedelta(hours=1)
        frame = frame.loc[frame['round'].eq(round_number) & start.dt.year.eq(year) & start.dt.hour.between(9, 16)
                          & frame[key].isin(pairs[key])].copy()
        nonhour = int(frame.ts.ne(frame.ts.dt.floor('h')).sum())
        frame = frame.loc[frame.ts.eq(frame.ts.dt.floor('h'))]
        if frame.duplicated([key, 'ts']).any():
            raise ValueError('duplicate forecast keys')
        bad = {}
        for col in columns:
            valid = valid_values(frame[col], col)
            bad[col] = int((~valid).sum())
            frame.loc[~valid, col] = np.nan
        frame = frame.set_index([key, 'ts'])
        idx = pd.MultiIndex.from_arrays([np.tile(pairs[key], len(times)), np.repeat(times, len(pairs))], names=frame.index.names)
        absent = int((~idx.isin(frame.index)).sum())
        raw = frame.reindex(idx)[columns].to_numpy(copy=True).reshape(len(dates), 8, len(pairs), len(columns)).transpose(0, 2, 1, 3)
        rain = raw[..., 4]
        wet = rain > .1
        transformed = [raw[..., 0] / 10, raw[..., 1] / 10, raw[..., 2] / 10,
                       raw[..., 3] / (3.6 if provider == 'wb' else 1),
                       wet.astype(float), np.where(wet, np.log1p(rain), 0.)]
        if provider == 'wb':
            transformed.append(raw[..., 5])
        z = np.stack(transformed, axis=-1)
        # A missing rain value must remain missing despite the binary transform.
        z[~np.isfinite(raw).all(axis=-1)] = np.nan
        parts.append(z)
        wet_flags.append(wet)
        audits[provider] = {'nonhour_rows': nonhour, 'missing_pair_hour_keys': absent,
                            'invalid_unique_station_cells': bad, 'unique_station_rows': len(frame)}
    x = np.concatenate(parts, axis=-1)
    keep = np.isfinite(x).all(axis=(1, 2, 3)) & good_generation
    count = np.bincount(dates[keep].month, minlength=13)[1:]
    if count.min() < 14:
        raise ValueError('fewer than 14 balanced dates in a month')
    dates, x, y = dates[keep], x[keep], y[keep]
    pair_y = np.stack([y[:, site_pair == i].mean(axis=1) for i in range(len(pairs))], axis=1)
    deviation = y - pair_y[:, site_pair]
    audit = {'year': year, 'round': round_number, 'sites': len(sites), 'pairs': len(pairs), 'dates': len(dates),
             'excluded_dates': int((~keep).sum()), 'excluded_generation_dates': int((~good_generation).sum()),
             'dates_per_month': count.tolist(), 'site_hour_rows': int(y.size),
             'valid_zero_cf': int((y == 0).sum()), 'above_one_cf': int((y > 1).sum()),
             'providers': audits, 'missing_generation_keys': int((~keys.isin(generation.index)).sum())}
    return x, y, pair_y, deviation, dates, audit


def extra_stats(deviation, months):
    return {'deviation': deviation, 'ss_day': (deviation ** 2).sum(axis=(1, 2)), 'months': months}


def extra_ss(stats, weights):
    ss = float(weights @ stats['ss_day'])
    for m in np.unique(stats['months']):
        take = stats['months'] == m
        n = weights[take].sum()
        if n > 0:
            total = np.tensordot(weights[take], stats['deviation'][take], axes=1)
            ss -= float((total * total).sum() / n)
    if ss < -1e-8:
        raise ValueError('negative extra response sum of squares')
    return max(ss, 0.)


def rain_support(x, pairs, months):
    wet_a, wet_b = x[..., 4] > .5, x[..., 10] > .5
    states = {f'A{int(a)}_B{int(b)}': {'pair_hour_rows': int(((wet_a == a) & (wet_b == b)).sum()),
               'dates': int(((wet_a == a) & (wet_b == b)).any(axis=(1, 2)).sum())}
              for a in [False, True] for b in [False, True]}
    out = {'states': states}
    for provider, target, other in [('A', wet_a, wet_b), ('B', wet_b, wet_a)]:
        disagreement = target & ~other
        active_pairs = disagreement.any(axis=(0, 2))
        cells = 0
        for month in np.unique(months):
            a, b = target[months == month], other[months == month]
            for other_state in [False, True]:
                cells += int((((a & (b == other_state)).sum(axis=0) >= 3)
                             & ((~a & (b == other_state)).sum(axis=0) >= 3)).sum())
        n = int(disagreement.sum())
        days = int(disagreement.any(axis=(1, 2)).sum())
        na = int(pairs.loc[active_pairs, 'wa_station'].nunique())
        nb = int(pairs.loc[active_pairs, 'wb_station'].nunique())
        out[provider] = {'disagreement_rows': n, 'disagreement_dates': days, 'a_stations': na,
                         'b_stations': nb, 'supported_cells': cells,
                         'eligible': bool(n >= 200 and days >= 20 and na >= 5 and nb >= 5 and cells >= 20)}
    return out


def model_columns(rain):
    all_columns = list(range(13))
    models = {'A': list(range(6)), 'B': list(range(6, 13)), 'AB': all_columns}
    for name, col in PRIMARY.items():
        models['without_' + name] = [j for j in all_columns if j != col]
        opposite = models['B'] if name.startswith('A') else models['A']
        models['opposite_plus_' + name] = sorted(opposite + [col])
    for provider, cols in [('A', [4, 5]), ('B', [10, 11])]:
        if rain[provider]['eligible']:
            models['without_' + provider + '_rain'] = [j for j in all_columns if j not in cols]
    return models


def fit(matrix, cols):
    xx, xy, yy = matrix[np.ix_(cols, cols)], matrix[cols, -1], float(matrix[-1, -1])
    diag = np.diag(xx)
    if yy <= 1e-10 or (diag <= 1e-9).any():
        return {'valid': False, 'reason': 'constant response or term'}
    sd = np.sqrt(diag)
    norm = xx / np.outer(sd, sd)
    eigen = np.linalg.eigvalsh(norm)
    condition = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else None
    if condition is None or condition >= 1e8:
        return {'valid': False, 'reason': 'rank/condition', 'condition': condition}
    beta = np.linalg.solve(norm, xy / sd) / sd
    sse = float(yy - beta @ xy)
    if sse < -1e-7 * max(yy, 1):
        raise ValueError('negative residual sum of squares')
    return {'valid': True, 'columns': cols, 'condition': condition, 'beta_pp': (100 * beta).tolist(),
            'sse': max(sse, 0.), 'within_r2': 1 - max(sse, 0.) / yy}


def evaluate(matrix, models, rows):
    fits = {name: fit(matrix, cols) for name, cols in models.items()}
    metrics = {}
    if not all(fits[k]['valid'] for k in ['A', 'B', 'AB']):
        return fits, metrics
    yy, full = float(matrix[-1, -1]), fits['AB']
    for name in ['A', 'B', 'AB']:
        metrics['R2_' + name] = fits[name]['within_r2']
    for added, reduced in [('A_given_B', 'B'), ('B_given_A', 'A')]:
        improvement = max(0., fits[reduced]['sse'] - full['sse'])
        metrics['delta_R2_' + added] = improvement / yy
        metrics['partial_R2_' + added] = improvement / fits[reduced]['sse']
    xx = matrix[:-1, :-1]
    inv = np.linalg.inv(xx)
    for name, col in PRIMARY.items():
        reduced, opposite_plus = fits['without_' + name], fits['opposite_plus_' + name]
        if not reduced['valid'] or not opposite_plus['valid']:
            continue
        improvement = max(0., reduced['sse'] - full['sse'])
        partial = improvement / reduced['sse']
        beta = full['beta_pp'][col]
        metrics[name + '_beta_pp'] = beta
        metrics[name + '_partial_correlation'] = float(np.sign(beta) * np.sqrt(partial))
        metrics[name + '_delta_R2'] = improvement / yy
        metrics[name + '_partial_R2'] = partial
        residual_ss = 1 / inv[col, col]
        metrics[name + '_remaining_weather_fraction'] = float(residual_ss / xx[col, col])
        metrics[name + '_remaining_weather_sd'] = float(np.sqrt(residual_ss / rows) * (10 if 'cloud' in name else 1))
        metrics[name + '_opposite_beta_pp'] = opposite_plus['beta_pp'][opposite_plus['columns'].index(col)]
    for provider in ['A', 'B']:
        reduced = fits.get('without_' + provider + '_rain')
        if reduced and reduced['valid']:
            improvement = max(0., reduced['sse'] - full['sse'])
            metrics[provider + '_rain_delta_R2'] = improvement / yy
            metrics[provider + '_rain_partial_R2'] = improvement / reduced['sse']
    return fits, metrics


def bootstrap_summary(array):
    ok = np.isfinite(array)
    return {'success': int(ok.sum()), 'interval': np.quantile(array[ok], [.025, .975]).tolist()
            if ok.sum() >= .9 * REPS else None}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise FileExistsError('completed results exist; no automatic rerun')
    hashes, code = {f: digest(SOURCE / f) for f in FILES}, {f: digest(ROOT / f) for f in DEPENDENCIES}
    old_hashes = json.loads((ROOT / 'data/analysis/2026-10-21_weather_conditional/summary.json').read_text(encoding='utf-8'))['source_hashes']
    for source_name in ['sites.csv', 'train/generation.parquet']:
        if hashes[source_name] != old_hashes[source_name]:
            raise ValueError('site/cohort source changed after DA-003')
    (OUT / 'preregistered_report.md').write_text(REPORT.read_text(encoding='utf-8'), encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': hashes, 'code_hashes': code, 'report_hash': digest(REPORT)})
    ids = json.loads((ROOT / 'data/analysis/2026-10-21_weather_conditional/inspection.json').read_text(encoding='utf-8'))['site_ids']
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id').loc[ids].reset_index()
    if not (np.isfinite(sites.capacity_kw) & sites.capacity_kw.gt(0)).all():
        raise ValueError('invalid capacity')
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet').set_index(['site_id', 'ts'])
    if not gen.index.is_unique:
        raise ValueError('duplicate generation keys')
    pairs, site_pair, counts, selected = mapping(sites)
    pairs.to_csv(OUT / 'pairs_local.csv', index=False, encoding='utf-8-sig')
    summary = {'python': sys.version, 'executable': sys.executable, 'source_hashes': hashes, 'code_hashes': code,
        'seed': SEED, 'reps': REPS, 'terms': TERMS, 'pairs': len(pairs), 'a_stations': int(pairs.wa_station.nunique()),
        'b_stations': int(pairs.wb_station.nunique()), 'matching_pairs': int(selected.sum()), 'cases': {}}
    rng = np.random.default_rng(SEED)
    for round_number in [10, 17]:
        for year in [2024, 2023]:
            label = f'{round_number}_{year}'
            log(label + ': reconstruct source')
            x, ysite, y, deviation, dates, audit = load_panel(sites, pairs, site_pair, gen, year, round_number)
            months = dates.month.to_numpy()
            rain = rain_support(x, pairs, months)
            models = model_columns(rain)
            stats = prepare(x, y, np.ones(y.shape, dtype=bool), counts, months)
            extra = extra_stats(deviation, months)
            site_chosen = selected[site_pair]
            subset_stats = prepare(x[:, selected], y[:, selected], np.ones(y[:, selected].shape, dtype=bool), counts[selected], months)
            subset_extra = extra_stats(deviation[:, site_chosen], months)
            np.savez_compressed(OUT / f'{label}_panel.npz', x=x, site_response=ysite, pair_response=y,
                site_pair=site_pair, pair_weights=counts, site_ids=sites.site_id.to_numpy(dtype=str),
                wa=pairs.wa_station.to_numpy(dtype=str), wb=pairs.wb_station.to_numpy(dtype=str),
                months=months, dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str), matching=selected)
            case = {'audit': audit, 'rain_support': rain, 'effects': {}}
            matrices = {}
            for date in [False, True]:
                effect = 'date_hour' if date else 'calendar'
                matrix = cross(stats, np.ones(len(y)), date)
                matrix[-1, -1] += extra_ss(extra, np.ones(len(y)))
                matrices[effect] = matrix
                fits, metrics = evaluate(matrix, models, ysite.size)
                sub = cross(subset_stats, np.ones(len(y)), date)
                sub[-1, -1] += extra_ss(subset_extra, np.ones(len(y)))
                matrices[effect + '_matching'] = sub
                sub_fits, sub_metrics = evaluate(sub, models, int(ysite[:, site_chosen].size))
                case['effects'][effect] = {'fits': fits, 'metrics': metrics,
                                          'matching': {'fits': sub_fits, 'metrics': sub_metrics}}
            np.savez_compressed(OUT / f'{label}_cross.npz', **matrices)
            names = sorted(set().union(*(v['metrics'] for v in case['effects'].values())))
            arrays = {effect: [] for effect in ['calendar', 'date_hour']}
            weights_saved = []
            for block in [7, 14]:
                for rep in range(REPS):
                    weights = block_weights(len(y), block, rng)
                    weights_saved.append(weights)
                    remainder = extra_ss(extra, weights)
                    for date in [False, True]:
                        effect = 'date_hour' if date else 'calendar'
                        matrix = cross(stats, weights, date)
                        matrix[-1, -1] += remainder
                        _, metrics = evaluate(matrix, models, int(weights.sum() * len(sites) * 8))
                        arrays[effect].append([metrics.get(name, np.nan) for name in names])
                    if (rep + 1) % 25 == 0:
                        log(f'{label} block={block}: {rep+1}/{REPS}')
            for effect in arrays:
                arrays[effect] = np.asarray(arrays[effect])
                case['effects'][effect]['bootstrap'] = {str(block): {
                    name: bootstrap_summary(arrays[effect][k * REPS:(k + 1) * REPS, j])
                    for j, name in enumerate(names)} for k, block in enumerate([7, 14])}
            np.savez_compressed(OUT / f'{label}_bootstrap.npz', weights=np.asarray(weights_saved),
                                metric_names=np.array(names, dtype=str), **arrays)
            summary['cases'][label] = case
            dump(OUT / 'progress.json', summary)
            log(label + ': ' + json.dumps({'audit': audit, 'rain': rain,
                'metrics': {k: v['metrics'] for k, v in case['effects'].items()}}, ensure_ascii=False))
    assert hashes == {f: digest(SOURCE / f) for f in FILES}
    assert code == {f: digest(ROOT / f) for f in DEPENDENCIES}
    dump(OUT / 'summary.json', summary)
    log('completed explanatory analysis')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
