"""DA-003 explanatory within-site/calendar weather associations, no forecasting."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import sysconfig
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log, valid_values

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA' / '01_participant_dataset'
OUT = ROOT / 'data/analysis/2026-10-21_weather_conditional'
REPORT = ROOT / 'reports/2026-10-21_07_기상조건과_발전량의_조건부관계_분석.md'
FILES = ['sites.csv', 'train/generation.parquet', 'train/observed_wb.parquet', 'train/forecast_wb.parquet']
SEED = 2026102103
BOOT_REPS = 200
VARIABLES = ['cloud', 'humidity', 'temp', 'wind', 'uv', 'rain']
KNOTS = {'cloud': [0, 25, 50, 75, 100], 'humidity': [20, 40, 60, 80, 100],
         'temp': [-10, 0, 10, 20, 30, 40], 'wind': [0, 1, 3, 5, 10], 'uv': [0, 1, 3, 6, 10]}
REFERENCE = {'cloud': 50., 'humidity': 60., 'temp': 20., 'wind': 3., 'uv': 3., 'rain': 0.}
CONTRASTS = [
    {'id': 'cloud_dry', 'variable': 'cloud', 'a': {'cloud': 25., 'rain': 0.}, 'b': {'cloud': 75., 'rain': 0.}},
    {'id': 'cloud_wet', 'variable': 'cloud', 'a': {'cloud': 25., 'rain': 1.}, 'b': {'cloud': 75., 'rain': 1.}},
    {'id': 'rain_cloud25', 'variable': 'rain', 'a': {'cloud': 25., 'rain': 0.}, 'b': {'cloud': 25., 'rain': 1.}},
    {'id': 'rain_cloud75', 'variable': 'rain', 'a': {'cloud': 75., 'rain': 0.}, 'b': {'cloud': 75., 'rain': 1.}},
    {'id': 'wind_temp10', 'variable': 'wind', 'a': {'wind': 1., 'temp': 10.}, 'b': {'wind': 5., 'temp': 10.}},
    {'id': 'wind_temp30', 'variable': 'wind', 'a': {'wind': 1., 'temp': 30.}, 'b': {'wind': 5., 'temp': 30.}},
    {'id': 'temp_wind1', 'variable': 'temp', 'a': {'temp': 10., 'wind': 1.}, 'b': {'temp': 30., 'wind': 1.}},
    {'id': 'temp_wind5', 'variable': 'temp', 'a': {'temp': 10., 'wind': 5.}, 'b': {'temp': 30., 'wind': 5.}},
    {'id': 'humidity', 'variable': 'humidity', 'a': {'humidity': 40.}, 'b': {'humidity': 80.}},
    {'id': 'uv', 'variable': 'uv', 'a': {'uv': 1.}, 'b': {'uv': 5.}},
]


def rcs_basis(x, knots):
    """Restricted cubic truncated-power basis, normalized to dimensionless terms."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    span = t[-1] - t[0]
    terms = [(x - t[0]) / span]
    for knot in t[:-2]:
        term = (np.maximum(x - knot, 0) ** 3
                - np.maximum(x - t[-2], 0) ** 3 * (t[-1] - knot) / (t[-1] - t[-2])
                + np.maximum(x - t[-1], 0) ** 3 * (t[-2] - knot) / (t[-1] - t[-2])) / span ** 3
        terms.append(term)
    return np.stack(terms, axis=-1)


def design(weather):
    parts, names, groups = [], [], {}
    for i, var in enumerate(VARIABLES):
        v = weather[..., i]
        if var == 'rain':
            wet = v > .1
            b = np.stack([wet.astype(float), np.where(wet, np.log1p(v), 0)], axis=-1)
        else:
            b = rcs_basis(v, KNOTS[var])
        start = len(names)
        names.extend([f'{var}_{j}' for j in range(b.shape[-1])])
        groups[var] = list(range(start, len(names)))
        parts.append(b)
    cloud, _, temp, wind, _, rain = np.moveaxis(weather, -1, 0)
    parts.extend([((rain > .1) * (cloud - 50) / 50)[..., None],
                  (((temp - 20) / 10) * ((wind - 3) / 3))[..., None]])
    names.extend(['wet_cloud', 'temp_wind'])
    return np.concatenate(parts, axis=-1), names, groups


def model_columns(names, groups):
    return {'full': list(range(len(names))),
            'no_uv': [i for i in range(len(names)) if i not in groups['uv']],
            **{'single_' + var: indices for var, indices in groups.items()}}


def contrast_vector(c):
    a = np.array([c['a'].get(v, REFERENCE[v]) for v in VARIABLES])
    b = np.array([c['b'].get(v, REFERENCE[v]) for v in VARIABLES])
    xa, _, _ = design(a)
    xb, _, _ = design(b)
    return 100 * (xb - xa)


def support_mask(weather, point):
    ok = np.ones(weather.shape[:-1], dtype=bool)
    tolerance = {'cloud': 10, 'humidity': 10, 'temp': 5, 'wind': 1, 'uv': 1}
    for var, value in point.items():
        values = weather[..., VARIABLES.index(var)]
        if var == 'rain':
            ok &= values <= .1 if value == 0 else (values >= .5) & (values <= 2)
        else:
            ok &= np.abs(values - value) <= tolerance[var]
    return ok


def support_stats(mask):
    rows = int(mask.sum())
    dates = int(mask.any(axis=(1, 2)).sum())
    sites = int(mask.any(axis=(0, 2)).sum())
    return {'rows': rows, 'dates': dates, 'sites': sites,
            'supported': rows >= 200 and dates >= 20 and sites >= 10}


def absorb(data, months, weights=None):
    if weights is None:
        weights = np.ones(len(data))
    out = data.copy()
    for m in np.unique(months):
        take = months == m
        total = weights[take].sum()
        if total <= 0:
            out[take] = 0
            continue
        mean = np.tensordot(weights[take], data[take], axes=(0, 0)) / total
        out[take] -= mean
    return out


def prepare_stats(x, y, months):
    flatx = x.reshape(len(x), -1, x.shape[-1])
    flaty = y.reshape(len(y), -1)
    return {'x': flatx, 'y': flaty, 'months': months,
            'xx_day': np.einsum('dcp,dcq->dpq', flatx, flatx, optimize=True),
            'xy_day': np.einsum('dcp,dc->dp', flatx, flaty, optimize=True),
            'yy_day': (flaty ** 2).sum(axis=1)}


def within_cross(stats, weights):
    xx = np.tensordot(weights, stats['xx_day'], axes=(0, 0))
    xy = weights @ stats['xy_day']
    yy = float(weights @ stats['yy_day'])
    for m in np.unique(stats['months']):
        take = stats['months'] == m
        w = weights[take]
        n = w.sum()
        if n <= 0:
            continue
        sx = np.tensordot(w, stats['x'][take], axes=(0, 0))
        sy = w @ stats['y'][take]
        xx -= sx.T @ sx / n
        xy -= sx.T @ sy / n
        yy -= sy @ sy / n
    return (xx + xx.T) / 2, xy, yy


def solve_cross(xx, xy, yy):
    if yy <= 1e-10:
        return {'valid': False, 'reason': 'constant within-cell response'}
    sd = np.sqrt(np.maximum(np.diag(xx), 0))
    if (sd < 1e-10).any():
        return {'valid': False, 'reason': 'constant within-cell basis'}
    normalized = xx / np.outer(sd, sd)
    eigen = np.linalg.eigvalsh(normalized)
    condition = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else None
    if eigen[0] <= 0 or condition >= 1e10:
        return {'valid': False, 'reason': 'rank or condition limit', 'condition': condition}
    beta = np.linalg.solve(normalized, xy / sd) / sd
    residual = float(yy - 2 * beta @ xy + beta @ xx @ beta)
    assert residual >= -1e-7 * max(yy, 1)
    normal_error = float(np.linalg.norm(xx @ beta - xy) / max(np.linalg.norm(xy), 1))
    assert normal_error < 1e-7
    return {'valid': True, 'beta': beta.tolist(), 'condition': condition,
            'sse': max(0., residual), 'within_r2': 1 - residual / yy,
            'normal_error': normal_error}


def block_weights(n, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / 7)))
    idx = ((starts[:, None] + np.arange(7)) % n).ravel()[:n]
    return np.bincount(idx, minlength=n)


def load_panel(kind, year, inspection, gen, sites):
    ids = inspection['site_ids']
    cohort = sites.loc[sites.site_id.isin(ids)].copy()
    start = gen.ts - pd.Timedelta(hours=1)
    g = gen.loc[start.dt.year.eq(year) & start.dt.hour.between(9, 16) & gen.site_id.isin(ids)].copy()
    g = g.merge(cohort[['site_id', 'capacity_kw', 'wb_station']], on='site_id', how='left', validate='many_to_one')
    rain = 'wb_precip_1h' if kind == 'observed' else 'wb_rain'
    raw_columns = ['wb_cloud', 'wb_rel_hum', 'wb_temp', 'wb_wind_speed', 'wb_uv_idx', rain]
    w = pd.read_parquet(SOURCE / f'train/{kind}_wb.parquet')
    s = w.ts - pd.Timedelta(hours=1)
    mask = s.dt.year.eq(year) & s.dt.hour.between(9, 16) & w.wb_station.isin(inspection['wb_stations'])
    if kind == 'forecast':
        mask &= w['round'].eq(17)
    w = w.loc[mask, ['wb_station', 'ts'] + raw_columns].copy()
    assert not w.duplicated(['wb_station', 'ts']).any()
    before = len(g)
    d = g.merge(w, on=['wb_station', 'ts'], how='left', validate='many_to_one', indicator=True)
    assert len(d) == before and not d.duplicated(['site_id', 'ts']).any()
    valid = np.isfinite(d.energy) & d.energy.ge(0) & np.isfinite(d.capacity_kw) & d.capacity_kw.gt(0)
    invalid_weather = {}
    for c in raw_columns:
        good = valid_values(d[c], c)
        invalid_weather[c] = int((~good).sum())
        valid &= good
    valid &= d.ts.eq(d.ts.dt.floor('h'))
    d['valid'] = valid
    d['cf'] = d.energy / d.capacity_kw
    d['date'] = (d.ts - pd.Timedelta(hours=1)).dt.floor('D')
    d['hour'] = (d.ts - pd.Timedelta(hours=1)).dt.hour
    audit_dates = d.groupby('date').agg(rows=('ts', 'size'), valid=('valid', 'sum'))
    dates = audit_dates.index[audit_dates.rows.eq(len(ids) * 8) & audit_dates.valid.eq(len(ids) * 8)].sort_values()
    monthly = pd.Series(dates.month).value_counts().reindex(range(1, 13), fill_value=0)
    if monthly.min() < 14:
        raise ValueError('fewer than 14 balanced dates in a month')
    d = d.loc[d.date.isin(dates)].sort_values(['date', 'site_id', 'hour'])
    assert d.groupby(['date', 'site_id']).hour.nunique().eq(8).all()
    shape = (len(dates), len(ids), 8)
    weather = d[raw_columns].to_numpy(copy=True).reshape(*shape, 6)
    weather[..., 3] /= 3.6
    y = d.cf.to_numpy().reshape(shape)
    assert np.isfinite(weather).all() and np.isfinite(y).all()
    audit = {'kind': kind, 'year': year, 'sites': len(ids), 'stations': len(inspection['wb_stations']),
             'candidate_rows': before, 'balanced_rows': len(d), 'balanced_dates': len(dates),
             'excluded_dates': len(audit_dates) - len(dates), 'dates_per_month': monthly.tolist(),
             'invalid_weather_cells': invalid_weather,
             'zero_cf_rows': int((y == 0).sum()), 'above_one_cf_rows': int((y > 1).sum())}
    return weather, y, np.asarray(dates.month), audit


def analyze_panel(weather, y, months, label, rng):
    x, names, groups = design(weather)
    models = model_columns(names, groups)
    stats = prepare_stats(x, y, months)
    xx, xy, yy = within_cross(stats, np.ones(len(y)))
    vectors = {c['id']: contrast_vector(c) for c in CONTRASTS}
    case = {'basis_names': names, 'models': {}, 'support': {}, 'ranges': {}}
    for c in CONTRASTS:
        ends = [support_stats(support_mask(weather, c[k])) for k in ['a', 'b']]
        case['support'][c['id']] = {'endpoints': ends, 'supported': all(e['supported'] for e in ends)}
    for i, var in enumerate(VARIABLES):
        case['ranges'][var] = np.quantile(weather[..., i], [.02, .5, .98]).tolist()
    np.savez_compressed(OUT / f'{label}_panel.npz', weather=weather, response=y, months=months)
    saved = {'xx': xx, 'xy': xy, 'yy': np.asarray(yy)}
    betas = {model: [] for model in models}
    failures = {model: {} for model in models}
    for model, cols in models.items():
        fit = solve_cross(xx[np.ix_(cols, cols)], xy[cols], yy)
        fit['columns'] = cols
        case['models'][model] = fit
    for b in range(BOOT_REPS):
        weights = block_weights(len(y), rng)
        if b == 0:
            saved['first_boot_weights'] = weights
        bxx, bxy, byy = within_cross(stats, weights)
        for model, cols in models.items():
            fit = solve_cross(bxx[np.ix_(cols, cols)], bxy[cols], byy)
            if fit['valid']:
                betas[model].append(fit['beta'])
            else:
                betas[model].append([np.nan] * len(cols))
                reason = fit['reason']
                failures[model][reason] = failures[model].get(reason, 0) + 1
        if b % 50 == 49:
            log(f'{label}: block bootstrap {b + 1}/{BOOT_REPS}')
    for model, cols in models.items():
        fit = case['models'][model]
        matrix = np.asarray(betas[model])
        saved['boot_' + model] = matrix
        good = np.isfinite(matrix).all(axis=1)
        fit.update(bootstrap_success=int(good.sum()), bootstrap_attempts=BOOT_REPS,
                   bootstrap_failures=failures[model], contrasts={})
        if not fit['valid']:
            continue
        for c in CONTRASTS:
            if model.startswith('single_') and model != 'single_' + c['variable']:
                continue
            if model == 'no_uv' and c['variable'] == 'uv':
                continue
            delta = vectors[c['id']][cols]
            samples = matrix[good] @ delta
            fit['contrasts'][c['id']] = {'pp': float(np.asarray(fit['beta']) @ delta),
                'interval': np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                'supported': case['support'][c['id']]['supported']}
    # Paired change when UV is included, using exactly the same bootstrap dates.
    case['uv_adjustment'] = {}
    for c in CONTRASTS:
        if c['variable'] == 'uv' or not all(case['models'][m]['valid'] for m in ['full', 'no_uv']):
            continue
        full = saved['boot_full'] @ vectors[c['id']][models['full']]
        reduced = saved['boot_no_uv'] @ vectors[c['id']][models['no_uv']]
        diff = full - reduced
        good = np.isfinite(diff)
        case['uv_adjustment'][c['id']] = {'pp_full_minus_no_uv':
            case['models']['full']['contrasts'][c['id']]['pp'] - case['models']['no_uv']['contrasts'][c['id']]['pp'],
            'paired_success': int(good.sum()),
            'interval': np.quantile(diff[good], [.025, .975]).tolist() if good.any() else None}
    # Site-level descriptive fits, with the same cell means and basis.
    cx, cy = absorb(x, months), absorb(y, months)
    heterogeneity = {c['id']: [] for c in CONTRASTS}
    eligible_sites = 0
    for site in range(y.shape[1]):
        sx = cx[:, site].reshape(-1, x.shape[-1])
        sy = cy[:, site].ravel()
        fit = solve_cross(sx.T @ sx, sx.T @ sy, float(sy @ sy))
        if not fit['valid']:
            continue
        eligible_sites += 1
        for c in CONTRASTS:
            supported = True
            for end in ['a', 'b']:
                m = support_mask(weather[:, site:site+1], c[end])
                supported &= int(m.sum()) >= 20 and int(m.any(axis=(1, 2)).sum()) >= 10
            if supported:
                heterogeneity[c['id']].append(float(np.asarray(fit['beta']) @ vectors[c['id']]))
    case['heterogeneity'] = {'numerically_eligible_sites': eligible_sites, 'contrasts': {}}
    for key, values in heterogeneity.items():
        a = np.asarray(values)
        case['heterogeneity']['contrasts'][key] = {'sites': len(a), 'positive': int((a > 0).sum()),
            'negative': int((a < 0).sum()), 'q25_median_q75': np.quantile(a, [.25, .5, .75]).tolist() if len(a) else None}
    saved.update({f'site_values_{key}': np.asarray(value) for key, value in heterogeneity.items()})
    np.savez_compressed(OUT / f'{label}_results.npz', **saved)
    return case


def run_analysis():
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Completed results exist; no automatic refitting.')
    inspection = json.loads((OUT / 'inspection.json').read_text(encoding='utf-8'))
    source_hashes = {f: digest(SOURCE / f) for f in FILES}
    (OUT / 'preregistered_report.md').write_text(REPORT.read_text(encoding='utf-8'), encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': source_hashes, 'code_hash': digest(__file__),
                               'report_hash': digest(REPORT), 'seed': SEED, 'bootstrap_reps': BOOT_REPS})
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet')
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig')
    rng = np.random.default_rng(SEED)
    summary = {'cases': {}, 'source_hashes': source_hashes, 'code_hash': digest(__file__),
               'seed': SEED, 'bootstrap_reps': BOOT_REPS, 'python': sys.version, 'executable': sys.executable}
    with threadpool_limits(limits=1):
        for kind in ['observed', 'forecast']:
            for year in [2024, 2023]:
                label = kind + '_' + str(year)
                log(label + ': prepare panel')
                weather, y, months, audit = load_panel(kind, year, inspection, gen, sites)
                case = analyze_panel(weather, y, months, label, rng)
                case['audit'] = audit
                summary['cases'][label] = case
                dump(OUT / 'progress.json', summary)
    summary['replication'] = {}
    for kind in ['observed', 'forecast']:
        summary['replication'][kind] = {}
        for c in CONTRASTS:
            key = c['id']
            a = summary['cases'][kind + '_2024']['models']['full']
            b = summary['cases'][kind + '_2023']['models']['full']
            aa, bb = a.get('contrasts', {}).get(key), b.get('contrasts', {}).get(key)
            if not aa or not bb or a['bootstrap_success'] < 180 or b['bootstrap_success'] < 180:
                status = 'fit or bootstrap failure'
            elif not (aa['supported'] and bb['supported']):
                status = 'insufficient support'
            else:
                signs = [1 if x['interval'][0] > 0 else -1 if x['interval'][1] < 0 else 0 for x in [aa, bb]]
                status = 'same direction' if signs[0] == signs[1] != 0 else 'opposite direction' if signs[0] * signs[1] < 0 else 'uncertain'
            summary['replication'][kind][key] = status
    assert source_hashes == {f: digest(SOURCE / f) for f in FILES}
    summary['finished_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    dump(OUT / 'summary.json', summary)
    log('DA-003 calculation complete')


def inspect_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet', columns=['site_id', 'ts', 'energy'])
    assert not gen.duplicated(['site_id', 'ts']).any()
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig')
    start = gen.ts - pd.Timedelta(hours=1)
    good = {}
    for year, expected in [(2023, 8760), (2024, 8784)]:
        d = gen.loc[start.dt.year.eq(year)].copy()
        c = d.groupby('site_id').size()
        good[year] = set(c.index[c.eq(expected)])
    ids = sorted(good[2023] & good[2024])
    cohort = sites.loc[sites.site_id.isin(ids)]
    info = {'python': platform.python_version(), 'executable': sys.executable,
            'complete_2024': len(good[2024]), 'complete_2023': len(good[2023]),
            'common_sites': len(ids), 'site_ids': ids, 'wb_stations': sorted(cohort.wb_station.unique()),
            'capacities_positive': bool(cohort.capacity_kw.gt(0).all()), 'weather': {}}
    for kind in ['observed', 'forecast']:
        d = pd.read_parquet(SOURCE / f'train/{kind}_wb.parquet')
        start = d.ts - pd.Timedelta(hours=1)
        d = d.loc[start.dt.year.isin([2023, 2024]) & d.wb_station.isin(info['wb_stations'])]
        if kind == 'forecast':
            d = d.loc[d['round'].eq(17)]
        d = d.copy()
        d['year'] = (d.ts - pd.Timedelta(hours=1)).dt.year
        info['weather'][kind] = {'columns': d.columns.tolist(),
             'availability': d.groupby('year').agg(rows=('ts', 'size'), stations=('wb_station', 'nunique')).reset_index().to_dict('records')}
    dump(OUT / 'inspection.json', info)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


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
