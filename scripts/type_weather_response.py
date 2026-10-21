"""DA-012: fixed generation types and within-weather-station slope differences."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log
from weather_conditional_relationships import load_panel
from conditional_support_date import design, prepare, cross, solve, block_weights

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA/01_participant_dataset'
OUT = ROOT / 'data/analysis/2026-10-21_type_weather_response'
TYPE_SOURCE = ROOT / 'data/analysis/2026-10-21_site_typology/summary.json'
OLD = ROOT / 'data/analysis/2026-10-21_weather_conditional'
REPORT = ROOT / 'reports/2026-10-21_15_발전유형별_기상반응_차이_분석.md'
FILES = ['sites.csv', 'train/generation.parquet', 'train/observed_wb.parquet', 'train/forecast_wb.parquet']
DEPENDENCIES = ['scripts/type_weather_response.py', 'scripts/conditional_support_date.py',
                'scripts/weather_conditional_relationships.py', 'scripts/weather_latent_structure.py']
SEED = 2026102112
REPS = 100
TERMS = ['cloud10', 'humidity10', 'temp10', 'wind_ms', 'uv', 'wet', 'log_rain']


def group_mapping(sites, labels):
    keys = sorted(set(zip(sites.wb_station, labels)))
    mapping = np.array([keys.index((st, int(c))) for st, c in zip(sites.wb_station, labels)])
    groups = [np.flatnonzero(mapping == j) for j in range(len(keys))]
    stations = sorted(sites.wb_station.unique())
    mixed = [st for st in stations if all((st, c) in keys for c in [0, 1])]
    sturdy = [st for st in mixed if all(np.sum((sites.wb_station.to_numpy() == st) & (labels == c)) >= 2 for c in [0, 1])]
    return keys, mapping, groups, mixed, sturdy


def full_design(weather, labels):
    x = design(weather)
    types = np.asarray(labels)
    return np.concatenate([x * (types == 0)[None, :, None, None],
                           x * (types == 1)[None, :, None, None]], axis=-1)


def shared_data(weather, response, sites, labels, stations):
    x, gap, members = [], [], []
    for st in stations:
        c1 = np.flatnonzero((sites.wb_station.to_numpy() == st) & (labels == 0))
        c2 = np.flatnonzero((sites.wb_station.to_numpy() == st) & (labels == 1))
        if not len(c1) or not len(c2):
            raise ValueError('station does not contain both fixed types')
        x.append(design(weather[:, c1[0]]))
        gap.append(response[:, c2].mean(axis=1) - response[:, c1].mean(axis=1))
        members.append({'station': st, 'C1': c1.tolist(), 'C2': c2.tolist()})
    return np.stack(x, axis=1), np.stack(gap, axis=1), members


def ref_scale(generation, sites):
    dates = pd.date_range('2023-01-01', '2023-12-31', tz='Asia/Seoul')
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for _ in range(len(sites)) for h in range(9, 17)])
    keys = pd.MultiIndex.from_arrays([np.tile(np.repeat(sites.site_id, 8), len(dates)), times], names=['site_id', 'ts'])
    g = generation.set_index(['site_id', 'ts'])
    if not g.index.is_unique:
        raise ValueError('duplicate source generation key')
    cf = g.reindex(keys).energy.to_numpy().reshape(len(dates), len(sites), 8) / sites.capacity_kw.to_numpy()[None, :, None]
    if not np.isfinite(cf).all() or (cf < 0).any():
        raise ValueError('incomplete or invalid 2023 reference generation')
    scale = cf.mean(axis=(0, 2))
    if (scale <= 1e-8).any():
        raise ValueError('nonpositive frozen reference mean')
    return scale


def metrics(result, sample):
    if not result['valid']:
        return {}
    beta = np.array(result['beta_pp'], dtype=float)
    if sample.startswith('all'):
        return {**{f'C1_{v}': float(beta[j]) for j, v in enumerate(['cloud', 'humidity'])},
                **{f'C2_{v}': float(beta[7 + j]) for j, v in enumerate(['cloud', 'humidity'])},
                **{f'delta_{v}': float(beta[7 + j] - beta[j]) for j, v in enumerate(['cloud', 'humidity'])}}
    return {f'delta_{v}': float(beta[j]) for j, v in enumerate(['cloud', 'humidity'])}


def fit_data(x, y, weights, months, date):
    stats = prepare(x, y, np.ones(y.shape, dtype=bool), weights, months)
    matrix = cross(stats, np.ones(len(y)), date)
    return solve(matrix), matrix


def intervals(values):
    ok = np.isfinite(values)
    return {'success': int(ok.sum()), 'interval': np.quantile(values[ok], [.025, .975]).tolist()
            if ok.sum() >= .9 * REPS else None}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise FileExistsError('completed results exist; no automatic rerun')
    hashes, code = {f: digest(SOURCE / f) for f in FILES}, {f: digest(ROOT / f) for f in DEPENDENCIES}
    old_hashes = json.loads((OLD / 'summary.json').read_text(encoding='utf-8'))['source_hashes']
    if hashes != old_hashes:
        raise ValueError('source changed after DA-003')
    typology = json.loads(TYPE_SOURCE.read_text(encoding='utf-8'))
    type_hash = digest(TYPE_SOURCE)
    ids = typology['site_ids']
    labels = np.array(typology['diagnostic_partition']['labels'], dtype=int)
    if set(labels) != {0, 1} or list(np.bincount(labels)) != [47, 33]:
        raise ValueError('unexpected frozen typology')
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id').loc[ids].reset_index()
    inspection = json.loads((OLD / 'inspection.json').read_text(encoding='utf-8'))
    if inspection['site_ids'] != ids:
        raise ValueError('site order differs between previous analyses')
    keys, inverse, groups, mixed, sturdy = group_mapping(sites, labels)
    mixed_sites = sites.wb_station.isin(mixed).to_numpy()
    ntype = np.bincount(labels[mixed_sites], minlength=2)
    if len(mixed) < 5 or ntype.min() < 10:
        raise ValueError('insufficient same-station type support')
    for c in [0, 1]:
        if (labels == c).sum() < 10 or sites.loc[labels == c, 'wb_station'].nunique() < 5:
            raise ValueError('insufficient overall type support')
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet')
    scale = ref_scale(gen, sites)
    (OUT / 'preregistered_report.md').write_text(REPORT.read_text(encoding='utf-8'), encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': hashes, 'code_hashes': code, 'type_source_hash': type_hash,
                               'report_hash': digest(REPORT), 'seed': SEED})
    members = [{'station': st, 'type': int(c), 'site_indices': groups[j].tolist()} for j, (st, c) in enumerate(keys)]
    summary = {'python': sys.version, 'executable': sys.executable, 'source_hashes': hashes, 'code_hashes': code,
        'type_source_hash': type_hash, 'seed': SEED, 'reps': REPS, 'terms': TERMS, 'site_ids': ids, 'labels': labels.tolist(),
        'groups': members, 'mixed_stations': mixed, 'sturdy_stations': sturdy,
        'mixed_type_sizes': ntype.tolist(), 'reference_scale_cf': scale.tolist(), 'cases': {}}
    rng = np.random.default_rng(SEED)
    for kind in ['observed', 'forecast']:
        for year in [2024, 2023]:
            label = f'{kind}_{year}'
            log(label + ': reconstruct source')
            weather, response, months, audit = load_panel(kind, year, inspection, gen, sites)
            old = np.load(OLD / f'{label}_panel.npz')
            np.testing.assert_allclose(weather, old['weather'], atol=0, rtol=0)
            np.testing.assert_allclose(response, old['response'], atol=0, rtol=0)
            if len(response) < 60:
                raise ValueError('insufficient balanced dates')
            np.savez_compressed(OUT / f'{label}_panel.npz', weather=weather, response=response, months=months,
                site_ids=np.asarray(ids, dtype=str), site_stations=sites.wb_station.to_numpy(dtype=str), labels=labels,
                group_inverse=inverse, reference_scale_cf=scale)
            cache, case, matrices, point_sensitivities = {}, {'audit': audit, 'samples': {}}, {}, {}
            for mode in ['absolute', 'relative']:
                ysite = response if mode == 'absolute' else response / scale[None, :, None]
                wx = full_design(weather, labels)
                gx = np.stack([wx[:, g[0]] for g in groups], axis=1)
                gy = np.stack([ysite[:, g].mean(axis=1) for g in groups], axis=1)
                all_stats = prepare(gx, gy, np.ones(gy.shape, dtype=bool), np.array([len(g) for g in groups]), months)
                sx, sy, sm = shared_data(weather, ysite, sites, labels, mixed)
                shared_stats = prepare(sx, sy, np.ones(sy.shape, dtype=bool), np.ones(len(mixed)), months)
                for source_name, stats in [('all', all_stats), ('shared', shared_stats)]:
                    sample = source_name + '_' + mode
                    cache[sample] = stats
                    case['samples'][sample] = {}
                    for date in [False, True]:
                        effect = 'date_hour' if date else 'calendar'
                        mat = cross(stats, np.ones(len(response)), date)
                        matrices[sample + '_' + effect] = mat
                        point = solve(mat)
                        case['samples'][sample][effect] = {'fit': point, 'metrics': metrics(point, sample)}
                for date in [False, True]:
                    effect = 'date_hour' if date else 'calendar'
                    sens = {}
                    if len(sturdy) >= 3:
                        tx, ty, _ = shared_data(weather, ysite, sites, labels, sturdy)
                        point, _ = fit_data(tx, ty, np.ones(len(sturdy)), months, date)
                        sens['min_two_each'] = {'fit': point, 'metrics': metrics(point, 'shared')}
                    leave = []
                    for omitted in range(len(mixed)):
                        keep = np.arange(len(mixed)) != omitted
                        point, _ = fit_data(sx[:, keep], sy[:, keep], np.ones(keep.sum()), months, date)
                        leave.append({'omitted_station': mixed[omitted], 'fit': point, 'metrics': metrics(point, 'shared')})
                    sens['leave_one_station_out'] = leave
                    point_sensitivities[mode + '_' + effect] = sens
            case['point_sensitivities'] = point_sensitivities
            np.savez_compressed(OUT / f'{label}_cross.npz', **matrices)
            draws = {sample + '_' + effect: [] for sample in cache for effect in ['calendar', 'date_hour']}
            metric_names = {sample: list(case['samples'][sample]['calendar']['metrics']) for sample in cache}
            weights_saved = []
            for block in [7, 14]:
                for rep in range(REPS):
                    weights = block_weights(len(response), block, rng)
                    weights_saved.append(weights)
                    for sample, stats in cache.items():
                        for date in [False, True]:
                            effect = 'date_hour' if date else 'calendar'
                            point = solve(cross(stats, weights, date))
                            m = metrics(point, sample)
                            draws[sample + '_' + effect].append([m.get(name, np.nan) for name in metric_names[sample]])
                    if (rep + 1) % 25 == 0:
                        log(f'{label} joint block={block}: {rep+1}/{REPS}')
            arrays = {k: np.asarray(v) for k, v in draws.items()}
            for sample in cache:
                for effect in ['calendar', 'date_hour']:
                    case['samples'][sample][effect]['bootstrap'] = {
                        str(block): {name: intervals(arrays[sample + '_' + effect][i * REPS:(i + 1) * REPS, j])
                                     for j, name in enumerate(metric_names[sample])}
                        for i, block in enumerate([7, 14])}
            np.savez_compressed(OUT / f'{label}_bootstrap.npz', weights=np.asarray(weights_saved),
                **arrays, **{sample + '_names': np.array(names, dtype=str) for sample, names in metric_names.items()})
            summary['cases'][label] = case
            dump(OUT / 'progress.json', summary)
            log(label + ': ' + json.dumps({sample: {effect: r['metrics'] for effect, r in values.items()}
                                             for sample, values in case['samples'].items()}))
    assert hashes == {f: digest(SOURCE / f) for f in FILES}
    assert code == {f: digest(ROOT / f) for f in DEPENDENCIES}
    assert type_hash == digest(TYPE_SOURCE)
    dump(OUT / 'summary.json', summary)
    log('completed without new clusters or generation predictions')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
