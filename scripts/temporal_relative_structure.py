"""DA-004 descriptive structural stability of peer-relative daily generation."""
from __future__ import annotations

import json
import sys
import sysconfig
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA/01_participant_dataset'
PRIOR = ROOT / 'data/analysis/2026-10-21_weather_conditional'
OUT = ROOT / 'data/analysis/2026-10-21_temporal_relative'
REPORT = ROOT / 'reports/2026-10-21_09_발전소_상대수준의_시간구조_분석.md'
FILES = ['sites.csv', 'train/generation.parquet']
SEED, REPS, TRIM = 2026102104, 499, 90


def seasonal_design(dates, degree):
    dates = pd.DatetimeIndex(dates)
    lengths = np.where(dates.is_leap_year, 366., 365.)
    phase = (dates.dayofyear.to_numpy() - 1) / lengths
    cols = [np.ones(len(dates))]
    for j in range(1, degree + 1):
        cols.extend([np.sin(2 * np.pi * j * phase), np.cos(2 * np.pi * j * phase)])
    return np.column_stack(cols)


def relative_to_peers(daily, stations):
    result = np.zeros_like(daily, dtype=float)
    eligible = np.zeros(daily.shape[1], dtype=bool)
    for station in np.unique(stations):
        ix = np.flatnonzero(stations == station)
        if len(ix) < 2:
            continue
        eligible[ix] = True
        values = daily[:, ix]
        peers = (values.sum(axis=1, keepdims=True) - values) / (len(ix) - 1)
        result[:, ix] = values - peers
    return result[:, eligible], eligible


def scan(residual, trim=TRIM):
    n = len(residual)
    if n < 2 * trim:
        raise ValueError('insufficient dates for two trimmed segments')
    cuts = np.arange(trim, n - trim + 1)
    cs = np.vstack([np.zeros(residual.shape[1]), np.cumsum(residual, axis=0)])
    delta = (cs[-1] - cs[cuts]) / (n - cuts[:, None]) - cs[cuts] / cuts[:, None]
    sd = residual.std(axis=0, ddof=1)
    scale = np.sqrt(cuts * (n - cuts) / n)
    values = np.divide(np.abs(delta) * scale[:, None], sd[None],
                       out=np.zeros_like(delta), where=sd[None] > 1e-12)
    best = values.argmax(axis=0)
    return cuts, values, cuts[best], values[best, np.arange(residual.shape[1])], delta[best, np.arange(residual.shape[1])]


def circular_indices(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)) % n).ravel()[:n]


def global_pvalues(statistics, maxima):
    return (1 + (maxima[:, None] >= statistics[None]).sum(axis=0)) / (len(maxima) + 1)


def persistence(residual, cuts, whole_delta):
    d90, thirds = [], []
    for j, k in enumerate(cuts):
        pre, post = residual[k-90:k, j], residual[k:k+90, j]
        d90.append(post.mean() - pre.mean())
        thirds.append([post[t:t+30].mean() - pre[60-t:90-t].mean() for t in (0, 30, 60)])
    d90, thirds = np.asarray(d90), np.asarray(thirds)
    passes = (np.abs(d90) >= .02) & ((thirds * np.sign(whole_delta[:, None])) > 0).all(axis=1)
    return d90, thirds, passes


def load_daily():
    old = json.loads((PRIOR / 'summary.json').read_text(encoding='utf-8'))
    assert old['source_hashes'] == {f: digest(SOURCE / f) for f in old['source_hashes']}
    info = json.loads((PRIOR / 'inspection.json').read_text(encoding='utf-8'))
    ids = info['site_ids']
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id').loc[ids]
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet', columns=['site_id', 'ts', 'energy'])
    assert not gen.duplicated(['site_id', 'ts']).any()
    start = gen.ts - pd.Timedelta(hours=1)
    good = {}
    for year, count in [(2023, 8760), (2024, 8784)]:
        counts = gen.loc[start.dt.year.eq(year)].groupby('site_id').size()
        good[year] = set(counts.index[counts.eq(count)])
    assert sorted(good[2023] & good[2024]) == ids
    take = gen.site_id.isin(ids) & start.dt.year.isin([2023, 2024]) & start.dt.hour.between(9, 16)
    g = gen.loc[take].copy()
    g['date'] = start[take].dt.floor('D')
    g['capacity'] = g.site_id.map(sites.capacity_kw)
    assert np.isfinite(g.energy).all() and g.energy.ge(0).all()
    assert np.isfinite(g.capacity).all() and g.capacity.gt(0).all()
    g['cf'] = g.energy / g.capacity
    assert g.groupby(['date', 'site_id']).size().eq(8).all()
    daily = g.groupby(['date', 'site_id']).cf.mean().unstack().loc[:, ids]
    expected = pd.date_range('2023-01-01', '2024-12-31', tz='Asia/Seoul')
    assert daily.index.tolist() == expected.tolist()
    assert daily.notna().all().all()
    dependencies = {}
    for year in [2023, 2024]:
        path = PRIOR / f'observed_{year}_panel.npz'
        dependencies[path.name] = digest(path)
        with np.load(path) as p:
            part = daily.loc[daily.index.year == year].to_numpy()
            assert p['response'].shape == (len(part), len(ids), 8)
            np.testing.assert_allclose(part, p['response'].mean(axis=2), atol=1e-12, rtol=0)
    # Derived inputs do not replace source verification, and no old fit is reused.
    return daily.to_numpy(), expected, np.asarray(ids), sites.wb_station.to_numpy(), {
        'cohort_sites': len(ids), 'dates': len(expected), 'hourly_rows': len(g),
        'zero_hourly_rows': int(g.cf.eq(0).sum()), 'above_one_rows': int(g.cf.gt(1).sum()),
        'derived_panel_hashes': dependencies, 'prior_code_hash': old['code_hash']}


def run():
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Completed DA-004 results already exist; no automatic rerun.')
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {f: digest(SOURCE / f) for f in FILES}
    codehash = digest(__file__)
    (OUT / 'preregistered_report.md').write_text(REPORT.read_text(encoding='utf-8'), encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': hashes, 'code_hash': codehash,
                               'report_hash': digest(REPORT), 'seed': SEED, 'reps': REPS})
    daily, dates, all_ids, stations, audit = load_daily()
    relative, eligible = relative_to_peers(daily, stations)
    ids = all_ids[eligible]
    st = stations[eligible]
    audit.update(eligible_sites=len(ids), eligible_stations=len(np.unique(st)),
                 excluded_singleton_sites=all_ids[~eligible].tolist())
    np.savez_compressed(OUT / 'panel.npz', daily=daily, relative=relative, site_ids=all_ids,
                        stations=stations, eligible=eligible, dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'))
    s = {'source_hashes': hashes, 'code_hash': codehash, 'seed': SEED, 'bootstrap_reps': REPS,
         'python': sys.version, 'executable': sys.executable, 'audit': audit, 'cases': {}}
    saved = {}
    with threadpool_limits(limits=1):
        for degree in [3, 6]:
            basis = seasonal_design(dates, degree)
            q = np.linalg.qr(basis, mode='reduced')[0]
            residual = relative - q @ (q.T @ relative)
            cuts, scores, best, peak, delta = scan(residual)
            d90, thirds, persistent = persistence(residual, best, delta)
            case = {'degree': degree, 'cut_indices': best.tolist(), 'dates': dates[best].strftime('%Y-%m-%d').tolist(),
                    'statistics': peak.tolist(), 'delta_pp': (delta * 100).tolist(),
                    'delta90_pp': (d90 * 100).tolist(), 'thirds_pp': (thirds * 100).tolist(),
                    'persistent': persistent.tolist(), 'blocks': {}}
            saved.update({f'residual_{degree}': residual, f'scores_{degree}': scores})
            for block in [7, 14]:
                rng = np.random.default_rng(SEED + 100 * degree + block)
                maxima = []
                for b in range(REPS):
                    ix = circular_indices(len(dates), block, rng)
                    sampled = residual[ix]
                    # Projection annihilates the fitted seasonal signal exactly.
                    resampled = sampled - q @ (q.T @ sampled)
                    maxima.append(float(scan(resampled)[3].max()))
                    if b == 0:
                        saved[f'first_indices_{degree}_{block}'] = ix
                    if b % 100 == 99:
                        log(f'degree={degree}, block={block}: {b+1}/{REPS}')
                maxima = np.asarray(maxima)
                pv = global_pvalues(peak, maxima)
                case['blocks'][str(block)] = {'p_adjusted': pv.tolist(),
                    'significant_sites': int((pv <= .05).sum()),
                    'null_q50_q95_q99': np.quantile(maxima, [.5, .95, .99]).tolist()}
                saved[f'null_max_{degree}_{block}'] = maxima
            s['cases'][str(degree)] = case
            dump(OUT / 'progress.json', s)
    a, b = s['cases']['3'], s['cases']['6']
    sig = np.ones(len(ids), dtype=bool)
    for degree in ['3', '6']:
        for block in ['7', '14']:
            sig &= np.asarray(s['cases'][degree]['blocks'][block]['p_adjusted']) <= .05
    close = np.abs(np.asarray(a['cut_indices']) - np.asarray(b['cut_indices'])) <= 60
    same = np.asarray(a['delta_pp']) * np.asarray(b['delta_pp']) > 0
    robust = sig & close & same & np.asarray(a['persistent'])
    s['sites'] = []
    for j, site in enumerate(ids):
        s['sites'].append({'site_id': str(site), 'station': str(st[j]),
            'peers': int((stations == st[j]).sum() - 1), 'all_four_significant': bool(sig[j]),
            'dates_within_60d': bool(close[j]), 'same_direction': bool(same[j]),
            'robust_candidate': bool(robust[j])})
    s['counts'] = {'all_four_significant': int(sig.sum()), 'date_and_direction_pass': int((sig & close & same).sum()),
                   'robust_candidates': int(robust.sum()), 'candidate_stations': len(np.unique(st[robust]))}
    saved['cuts'] = cuts
    np.savez_compressed(OUT / 'results.npz', **saved)
    assert hashes == {f: digest(SOURCE / f) for f in FILES}
    assert codehash == digest(__file__)
    s['finished_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    dump(OUT / 'summary.json', s)
    print(json.dumps({'audit': audit, 'counts': s['counts'], 'finished_at': s['finished_at']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    assert sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var('Py_GIL_DISABLED')
    run()
