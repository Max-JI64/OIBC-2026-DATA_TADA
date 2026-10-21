"""DA-007: within-calendar joint support and date/hour fixed-effect sensitivity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log
from weather_conditional_relationships import load_panel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA/01_participant_dataset'
OLD = ROOT / 'data/analysis/2026-10-21_weather_conditional'
OUT = ROOT / 'data/analysis/2026-10-21_conditional_support_date'
REPORT = ROOT / 'reports/2026-10-21_13_공동지지와_날짜요인의_조건부관계_분석.md'
FILES = ['sites.csv', 'train/generation.parquet', 'train/observed_wb.parquet', 'train/forecast_wb.parquet']
DEPENDENCIES = ['scripts/conditional_support_date.py', 'scripts/weather_conditional_relationships.py',
                'scripts/weather_latent_structure.py']
TERMS = ['cloud10', 'humidity10', 'temp10', 'wind_ms', 'uv', 'wet', 'log_rain']
SEED = 2026102107
REPS = 100
SPECS = {'cloud_wide': (0, 25., 75., 10.), 'humidity_wide': (1, 40., 80., 10.),
         'cloud_local': (0, 40., 60., 5.), 'humidity_local': (1, 50., 70., 5.)}


def design(w):
    return np.stack([w[..., 0] / 10, w[..., 1] / 10, w[..., 2] / 10, w[..., 3], w[..., 4],
                     (w[..., 5] > .1).astype(float), np.where(w[..., 5] > .1, np.log1p(w[..., 5]), 0.)], axis=-1)


def prune_edges(edges):
    """Bipartite 2-core: every retained date has >=2 counterpart dates."""
    keep = edges.copy()
    while True:
        new = keep & (keep.sum(axis=1) >= 2)[:, None] & (keep.sum(axis=0) >= 2)[None, :]
        if np.array_equal(new, keep):
            return keep
        keep = new


def joint_mask(weather, months, spec):
    variable, low, high, width = spec
    mask = np.zeros(weather.shape[:-1], dtype=bool)
    cells = []
    limits = [10., 10., 3., 1., 1.]
    for month in np.unique(months):
        ds = np.flatnonzero(months == month)
        for station in range(weather.shape[1]):
            for hour in range(weather.shape[2]):
                z = weather[ds, station, hour]
                lo = np.flatnonzero(abs(z[:, variable] - low) <= width)
                hi = np.flatnonzero(abs(z[:, variable] - high) <= width)
                if len(lo) < 3 or len(hi) < 3:
                    continue
                a, b = z[lo], z[hi]
                edge = np.ones((len(a), len(b)), dtype=bool)
                for col, limit in enumerate(limits):
                    if col != variable:
                        edge &= abs(a[:, None, col] - b[None, :, col]) <= limit
                edge &= (a[:, None, 5] > .1) == (b[None, :, 5] > .1)
                edge &= abs(np.log1p(a[:, None, 5]) - np.log1p(b[None, :, 5])) <= .4
                edge &= lo[:, None] != hi[None, :]
                edge = prune_edges(edge)
                lkeep, hkeep = edge.any(axis=1), edge.any(axis=0)
                if lkeep.sum() < 3 or hkeep.sum() < 3 or edge.sum() < 5:
                    continue
                left, right = ds[lo[lkeep]], ds[hi[hkeep]]
                mask[np.r_[left, right], station, hour] = True
                cells.append({'month': int(month), 'station_index': station, 'hour': hour + 9,
                              'low_dates': left.tolist(), 'high_dates': right.tolist(), 'edges': int(edge.sum())})
    return mask, cells


def prepare(x, y, mask, station_weights, months):
    z = np.concatenate([x, y[..., None]], axis=-1)
    v = mask * np.asarray(station_weights, dtype=float)[None, :, None]
    total = v.sum(axis=1)
    daymean = np.divide(np.einsum('dsh,dshp->dhp', v, z), total[..., None],
                        out=np.zeros((len(x), x.shape[2], z.shape[-1])), where=total[..., None] > 0)
    centered = z - daymean[:, None]
    offdiag = np.einsum('dsh,dth->dhst', v, v)
    offdiag = np.divide(offdiag, total[..., None, None], out=np.zeros_like(offdiag), where=total[..., None, None] > 0)
    graph = -offdiag
    index = np.arange(x.shape[1])
    graph[..., index, index] += v.transpose(0, 2, 1)
    return {'months': months, 'v': v.transpose(0, 2, 1),
        'raw_b': (v[..., None] * z).transpose(0, 2, 1, 3),
        'date_b': (v[..., None] * centered).transpose(0, 2, 1, 3),
        'raw_zz': np.einsum('dsh,dshp,dshq->dpq', v, z, z, optimize=True),
        'date_zz': np.einsum('dsh,dshp,dshq->dpq', v, centered, centered, optimize=True), 'graph': graph}


def cross(stats, weights, date_effect):
    total = np.tensordot(weights, stats['date_zz' if date_effect else 'raw_zz'], axes=1)
    for month in np.unique(stats['months']):
        take = stats['months'] == month
        wt = weights[take]
        b = np.tensordot(wt, stats['date_b' if date_effect else 'raw_b'][take], axes=1)
        if date_effect:
            g = np.tensordot(wt, stats['graph'][take], axes=1)
            eig, vectors = np.linalg.eigh((g + g.transpose(0, 2, 1)) / 2)
            keep = eig > np.maximum(eig[:, -1:] * 1e-10, 1e-10)
            inv = np.divide(1., eig, out=np.zeros_like(eig), where=keep)
            projected = np.einsum('hsk,hsp->hkp', vectors, b)
            total -= np.einsum('hkp,hk,hkq->pq', projected, inv, projected, optimize=True)
        else:
            n = np.tensordot(wt, stats['v'][take], axes=1)
            normalized = np.divide(b, n[..., None], out=np.zeros_like(b), where=n[..., None] > 0)
            total -= np.einsum('hsp,hsq->pq', b, normalized)
    return (total + total.T) / 2


def solve(matrix, target=None):
    xx, xy = matrix[:-1, :-1], matrix[:-1, -1]
    diagonal = np.diag(xx)
    active = diagonal > 1e-9
    beta = np.full(len(diagonal), np.nan)
    base = {'active': active.tolist(), 'within_ss': np.maximum(diagonal, 0).tolist()}
    if not active.any() or (target is not None and not active[target]):
        return {**base, 'valid': False, 'reason': 'no remaining target variation'}
    sd = np.sqrt(diagonal[active])
    cov = xx[np.ix_(active, active)] / np.outer(sd, sd)
    eigen = np.linalg.eigvalsh(cov)
    condition = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else None
    if condition is None or condition >= 1e8:
        return {**base, 'valid': False, 'reason': 'rank/condition', 'condition': condition}
    beta[active] = np.linalg.solve(cov, xy[active] / sd) / sd * 100
    return {**base, 'valid': True, 'condition': condition,
            'beta_pp': [float(v) if np.isfinite(v) else None for v in beta]}


def support_audit(mask, station_weights, cells=None):
    nstation = int(mask.any(axis=(0, 2)).sum())
    ndates = int(mask.any(axis=(1, 2)).sum())
    nrows = int((mask * station_weights[None, :, None]).sum())
    ncells = len(cells) if cells is not None else 12 * mask.shape[1] * mask.shape[2]
    return {'stations': nstation, 'dates': ndates, 'station_hour_rows': int(mask.sum()),
            'site_hour_rows': nrows, 'calendar_cells': ncells,
            'eligible': bool(nstation >= 5 and ndates >= 60 and nrows >= 1000 and ncells >= 20)}


def block_weights(n, block, rng):
    starts = rng.integers(n, size=int(np.ceil(n / block)))
    indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
    return np.bincount(indices, minlength=n)


def intervals(draws):
    out, successes = [], []
    for col in range(draws.shape[1]):
        ok = np.isfinite(draws[:, col])
        successes.append(int(ok.sum()))
        out.append(np.quantile(draws[ok, col], [.025, .975]).tolist() if ok.sum() >= .9 * REPS else None)
    return {'success_by_term': successes, 'intervals_pp': out}


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        raise FileExistsError('completed results exist; no automatic rerun')
    hashes = {f: digest(SOURCE / f) for f in FILES}
    code = {f: digest(ROOT / f) for f in DEPENDENCIES}
    text = REPORT.read_text(encoding='utf-8')
    (OUT / 'preregistered_report.md').write_text(text, encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': hashes, 'code_hashes': code, 'report_hash': digest(REPORT)})
    inspection = json.loads((OLD / 'inspection.json').read_text(encoding='utf-8'))
    old_summary = json.loads((OLD / 'summary.json').read_text(encoding='utf-8'))
    assert hashes == old_summary['source_hashes']
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet')
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id').loc[inspection['site_ids']].reset_index()
    counts = sites.groupby('wb_station').size().reindex(inspection['wb_stations']).to_numpy()
    station_names = inspection['wb_stations']
    summary = {'python': sys.version, 'executable': sys.executable, 'source_hashes': hashes, 'code_hashes': code,
               'seed': SEED, 'reps': REPS, 'terms': TERMS, 'specs': SPECS, 'cases': {}}
    rng = np.random.default_rng(SEED)
    for kind in ['observed', 'forecast']:
        for year in [2024, 2023]:
            label = f'{kind}_{year}'
            log(label + ': reconstruct source')
            weather, response, months, audit = load_panel(kind, year, inspection, gen, sites)
            old = np.load(OLD / f'{label}_panel.npz')
            np.testing.assert_allclose(weather, old['weather'], atol=0, rtol=0)
            np.testing.assert_allclose(response, old['response'], atol=0, rtol=0)
            np.testing.assert_array_equal(months, old['months'])
            w = np.stack([weather[:, np.flatnonzero(sites.wb_station.eq(st))[0]] for st in station_names], axis=1)
            y = np.stack([response[:, sites.wb_station.eq(st).to_numpy()].mean(axis=1) for st in station_names], axis=1)
            x = design(w)
            masks, cell_records = {'all': np.ones(y.shape, dtype=bool)}, {}
            for name, spec in SPECS.items():
                masks[name], cell_records[name] = joint_mask(w, months, spec)
            np.savez_compressed(OUT / f'{label}_panel.npz', weather=w, response=y, site_response=response,
                months=months, station_weights=counts, station_names=np.array(station_names, dtype=str),
                site_stations=sites.wb_station.to_numpy(dtype=str), site_ids=sites.site_id.to_numpy(dtype=str),
                **{'mask_' + k: v for k, v in masks.items()})
            dump(OUT / f'{label}_support_cells.json', cell_records)
            case = {'audit': audit, 'samples': {}}
            cache = {}
            matrices = {}
            for sample, mask in masks.items():
                a = support_audit(mask, counts, cell_records.get(sample))
                result = {'support': a, 'fits': {}}
                case['samples'][sample] = result
                log(label + ' ' + sample + ': ' + json.dumps(a))
                if not a['eligible']:
                    continue
                stats = prepare(x, y, mask, counts, months)
                cache[sample] = stats
                for date in [False, True]:
                    name = 'date_hour' if date else 'calendar'
                    mat = cross(stats, np.ones(len(y)), date)
                    matrices[sample + '_' + name] = mat
                    result['fits'][name] = solve(mat, SPECS[sample][0] if sample != 'all' else None)
                r1, r2 = result['fits']['calendar'], result['fits']['date_hour']
                result['remaining_variance_ratio'] = np.divide(r2['within_ss'], r1['within_ss'],
                    out=np.zeros(len(TERMS)), where=np.array(r1['within_ss']) > 1e-9).tolist()
            np.savez_compressed(OUT / f'{label}_cross.npz', **matrices)
            draws = {sample + '_' + effect: [] for sample in cache for effect in ['calendar', 'date_hour']}
            weight_draws = []
            for block in [7, 14]:
                for rep in range(REPS):
                    weight = block_weights(len(y), block, rng)
                    weight_draws.append(weight)
                    for sample, stats in cache.items():
                        target = SPECS[sample][0] if sample != 'all' else None
                        for date in [False, True]:
                            effect = 'date_hour' if date else 'calendar'
                            r = solve(cross(stats, weight, date), target)
                            values = [np.nan if v is None else v for v in r['beta_pp']] if r['valid'] else [np.nan] * len(TERMS)
                            draws[sample + '_' + effect].append(values)
                    if (rep + 1) % 25 == 0:
                        log(f'{label} joint block={block}: {rep+1}/{REPS}')
            arrays = {key: np.asarray(value) for key, value in draws.items()}
            np.savez_compressed(OUT / f'{label}_bootstrap.npz', weights=np.asarray(weight_draws), **arrays)
            for sample in cache:
                result = case['samples'][sample]
                for effect in ['calendar', 'date_hour']:
                    a = arrays[sample + '_' + effect]
                    result['fits'][effect]['bootstrap'] = {str(b): intervals(a[k * REPS:(k + 1) * REPS])
                                                          for k, b in enumerate([7, 14])}
                diff = arrays[sample + '_date_hour'] - arrays[sample + '_calendar']
                result['date_difference'] = {str(b): intervals(diff[k * REPS:(k + 1) * REPS]) for k, b in enumerate([7, 14])}
            summary['cases'][label] = case
            dump(OUT / 'progress.json', summary)
    assert hashes == {f: digest(SOURCE / f) for f in FILES}
    assert code == {f: digest(ROOT / f) for f in DEPENDENCIES}
    dump(OUT / 'summary.json', summary)
    log('completed without forecasting')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        analyze()
