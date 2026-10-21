"""Independent source-key, support-graph and direct dummy least-squares checks."""
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from conditional_support_date import ROOT, SOURCE, OLD, OUT, FILES, DEPENDENCIES, SPECS, SEED, REPS
from weather_latent_structure import digest, dump, valid_values


def direct_cross(x, y, mask, sizes, months, weights, date_effect):
    """Explicit nuisance dummy projection per month/hour, without Schur formulas."""
    total = np.zeros((x.shape[-1] + 1, x.shape[-1] + 1))
    for month in np.unique(months):
        take = months == month
        xm, ym, mm, wm = x[take], y[take], mask[take], weights[take]
        n, ns, nh = mm.shape
        for h in range(nh):
            z = np.column_stack([xm[:, :, h].reshape(-1, x.shape[-1]), ym[:, :, h].ravel()])
            nuisance = np.tile(np.eye(ns), (n, 1))
            if date_effect:
                nuisance = np.column_stack([nuisance, np.repeat(np.eye(n), ns, axis=0)])
            weight = (wm[:, None] * sizes[None, :] * mm[:, :, h]).ravel()
            active = weight > 0
            if not active.any():
                continue
            a = nuisance[active] * np.sqrt(weight[active, None])
            b = z[active] * np.sqrt(weight[active, None])
            residual = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
            total += residual.T @ residual
    return total


def independent_support(w, months, spec):
    var, low, high, width = spec
    mask = np.zeros(w.shape[:-1], dtype=bool)
    cells = {}
    for month in sorted(set(months)):
        ds = np.flatnonzero(months == month)
        for station in range(w.shape[1]):
            for h in range(w.shape[2]):
                left = [int(d) for d in ds if abs(w[d, station, h, var] - low) <= width]
                right = [int(d) for d in ds if abs(w[d, station, h, var] - high) <= width]
                if len(left) < 3 or len(right) < 3:
                    continue
                edges = set()
                for a in left:
                    for b in right:
                        aa, bb = w[a, station, h], w[b, station, h]
                        if a == b or (aa[5] > .1) != (bb[5] > .1) or abs(np.log1p(aa[5]) - np.log1p(bb[5])) > .4:
                            continue
                        if any(abs(aa[k] - bb[k]) > limit for k, limit in enumerate([10., 10., 3., 1., 1.]) if k != var):
                            continue
                        edges.add((a, b))
                while edges:
                    lcounts = {a: sum(e[0] == a for e in edges) for a, _ in edges}
                    rcounts = {b: sum(e[1] == b for e in edges) for _, b in edges}
                    reduced = {(a, b) for a, b in edges if lcounts[a] >= 2 and rcounts[b] >= 2}
                    if reduced == edges:
                        break
                    edges = reduced
                lset = {a for a, _ in edges}
                rset = {b for _, b in edges}
                if len(lset) >= 3 and len(rset) >= 3 and len(edges) >= 5:
                    mask[list(lset | rset), station, h] = True
                    cells[(int(month), station, h + 9)] = (sorted(lset), sorted(rset), len(edges))
    return mask, cells


def verify():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert pre['source_hashes'] == s['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert pre['code_hashes'] == s['code_hashes'] == {f: digest(ROOT / f) for f in DEPENDENCIES}
    snap = (OUT / 'preregistered_report.md').read_bytes().replace(b'\r\n', b'\n')
    assert pre['report_hash'] in [hashlib.sha256(snap).hexdigest(), hashlib.sha256(snap.replace(b'\n', b'\r\n')).hexdigest()]
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet').set_index(['site_id', 'ts'])
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id')
    rng = np.random.default_rng(SEED)
    counts = {'generation_cells': 0, 'weather_cells': 0, 'support_masks': 0, 'direct_point_crosses': 0,
              'direct_bootstrap_crosses': 0, 'original_site_dummy_checks': 0, 'weight_draws': 0}
    errors = []
    for label, case in s['cases'].items():
        kind, year = label.split('_')
        year = int(year)
        p = np.load(OUT / f'{label}_panel.npz')
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', tz='Asia/Seoul')
        assert len(dates) == len(p['months'])
        np.testing.assert_array_equal(dates.month, p['months'])
        ids = p['site_ids']
        ns = len(ids)
        times = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for _ in ids for h in range(9, 17)])
        keys = pd.MultiIndex.from_arrays([np.tile(np.repeat(ids, 8), len(dates)), times], names=gen.index.names)
        ysite = gen.reindex(keys).energy.to_numpy().reshape(len(dates), ns, 8) / sites.loc[ids].capacity_kw.to_numpy()[None, :, None]
        np.testing.assert_allclose(ysite, p['site_response'], atol=0, rtol=0)
        stations = p['station_names']
        site_stations = sites.loc[ids].wb_station.to_numpy()
        groups = [np.flatnonzero(site_stations == st) for st in stations]
        sizes = np.array([len(v) for v in groups])
        np.testing.assert_array_equal(sizes, p['station_weights'])
        mean = np.stack([ysite[:, g].mean(axis=1) for g in groups], axis=1)
        np.testing.assert_allclose(mean, p['response'], atol=1e-15)
        rain = 'wb_precip_1h' if kind == 'observed' else 'wb_rain'
        cols = ['wb_cloud', 'wb_rel_hum', 'wb_temp', 'wb_wind_speed', 'wb_uv_idx', rain]
        wsource = pd.read_parquet(SOURCE / f'train/{kind}_wb.parquet')
        if kind == 'forecast':
            wsource = wsource.loc[wsource['round'].eq(17)]
        for col in cols:
            wsource.loc[~valid_values(wsource[col], col), col] = np.nan
        wsource = wsource.set_index(['wb_station', 'ts'])
        wtimes = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for _ in stations for h in range(9, 17)])
        wkeys = pd.MultiIndex.from_arrays([np.tile(np.repeat(stations, 8), len(dates)), wtimes], names=wsource.index.names)
        weather = wsource.reindex(wkeys)[cols].to_numpy(copy=True).reshape(len(dates), len(stations), 8, 6)
        weather[..., 3] /= 3.6
        np.testing.assert_allclose(weather, p['weather'], atol=0, rtol=0)
        counts['generation_cells'] += ysite.size
        counts['weather_cells'] += weather.size
        saved_cells = json.loads((OUT / f'{label}_support_cells.json').read_text(encoding='utf-8'))
        for sample, spec in SPECS.items():
            mask, cells = independent_support(weather, p['months'], spec)
            np.testing.assert_array_equal(mask, p['mask_' + sample])
            expected = {(r['month'], r['station_index'], r['hour']): (r['low_dates'], r['high_dates'], r['edges'])
                        for r in saved_cells[sample]}
            assert cells == expected
            a = case['samples'][sample]['support']
            n = int((mask * sizes[None, :, None]).sum())
            assert n == a['site_hour_rows'] and len(cells) == a['calendar_cells']
            assert int(mask.any(axis=(1, 2)).sum()) == a['dates']
            assert int(mask.any(axis=(0, 2)).sum()) == a['stations']
            assert a['eligible'] == bool(a['stations'] >= 5 and a['dates'] >= 60 and n >= 1000 and len(cells) >= 20)
            counts['support_masks'] += 1
        x = np.stack([weather[..., 0] / 10, weather[..., 1] / 10, weather[..., 2] / 10, weather[..., 3],
                      weather[..., 4], (weather[..., 5] > .1).astype(float),
                      np.where(weather[..., 5] > .1, np.log1p(weather[..., 5]), 0)], axis=-1)
        saved = np.load(OUT / f'{label}_cross.npz')
        boot = np.load(OUT / f'{label}_bootstrap.npz')
        for k, block in enumerate([7, 14]):
            for rep in range(REPS):
                starts = rng.integers(len(dates), size=int(np.ceil(len(dates) / block)))
                ix = np.concatenate([np.arange(v, v + block) % len(dates) for v in starts])[:len(dates)]
                np.testing.assert_array_equal(np.bincount(ix, minlength=len(dates)), boot['weights'][k * REPS + rep])
                counts['weight_draws'] += 1
        for sample, result in case['samples'].items():
            if not result['support']['eligible']:
                assert not result['fits']
                continue
            mask = p['mask_' + sample]
            for effect in ['calendar', 'date_hour']:
                name = sample + '_' + effect
                matrix = direct_cross(x, mean, mask, sizes, p['months'], np.ones(len(dates)), effect == 'date_hour')
                np.testing.assert_allclose(matrix, saved[name], atol=2e-7, rtol=1e-8)
                fit = result['fits'][effect]
                active = np.array(fit['active'])
                beta = np.linalg.lstsq(matrix[:-1, :-1][np.ix_(active, active)], matrix[:-1, -1][active], rcond=None)[0] * 100
                np.testing.assert_allclose(beta, np.array(fit['beta_pp'], dtype=float)[active], atol=1e-7)
                errors.append(float(np.max(abs(beta - np.array(fit['beta_pp'], dtype=float)[active]))))
                counts['direct_point_crosses'] += 1
                draws = boot[name]
                for k, block in enumerate([7, 14]):
                    number = k * REPS
                    mat = direct_cross(x, mean, mask, sizes, p['months'], boot['weights'][number], effect == 'date_hour')
                    beta = np.linalg.lstsq(mat[:-1, :-1][np.ix_(active, active)], mat[:-1, -1][active], rcond=None)[0] * 100
                    np.testing.assert_allclose(beta, draws[number, active], atol=1e-7)
                    counts['direct_bootstrap_crosses'] += 1
                    values = draws[k * REPS:(k + 1) * REPS]
                    for col in range(values.shape[1]):
                        good = np.isfinite(values[:, col])
                        check = fit['bootstrap'][str(block)]
                        assert good.sum() == check['success_by_term'][col]
                        if good.sum() >= .9 * REPS:
                            np.testing.assert_allclose(np.quantile(values[good, col], [.025, .975]), check['intervals_pp'][col], atol=1e-12)
            for k, block in enumerate([7, 14]):
                values = boot[sample + '_date_hour'] - boot[sample + '_calendar']
                for col in range(7):
                    a = values[k * REPS:(k + 1) * REPS, col]
                    if np.isfinite(a).sum() >= .9 * REPS:
                        np.testing.assert_allclose(np.quantile(a[np.isfinite(a)], [.025, .975]),
                                                   result['date_difference'][str(block)]['intervals_pp'][col], atol=1e-12)
            # Original 80-site nuisance dummies vs station-mean sufficient statistics in July/hour 12.
            pick = p['months'] == 7
            mapping = np.array([list(stations).index(st) for st in site_stations])
            for date in [False, True]:
                a = direct_cross(x[pick, :, 3:4], mean[pick, :, 3:4], mask[pick, :, 3:4], sizes,
                                 np.ones(pick.sum(), dtype=int), np.ones(pick.sum()), date)
                b = direct_cross(x[pick][:, mapping, 3:4], ysite[pick, :, 3:4], mask[pick][:, mapping, 3:4],
                                 np.ones(ns), np.ones(pick.sum(), dtype=int), np.ones(pick.sum()), date)
                np.testing.assert_allclose(a[:-1], b[:-1], atol=1e-8, rtol=1e-9)
                counts['original_site_dummy_checks'] += 1
    counts['max_point_beta_difference_pp'] = max(errors)
    counts['source_hashes_unchanged'] = True
    counts['all_saved_intervals_checked'] = True
    dump(OUT / 'verification.json', counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
