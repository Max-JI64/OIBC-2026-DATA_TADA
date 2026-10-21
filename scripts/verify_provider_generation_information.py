"""DA-011 independent source reconstruction and original-site dummy OLS checks."""
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from provider_generation_information import ROOT, SOURCE, OUT, FILES, DEPENDENCIES, PRIMARY, SEED, REPS
from weather_latent_structure import digest, dump, valid_values
from verify_conditional_support_date import direct_cross


def check_metrics(matrix, result, rows):
    yy = float(matrix[-1, -1])
    independently = {}
    for name, fit in result['fits'].items():
        if not fit['valid']:
            continue
        cols = fit['columns']
        xx, xy = matrix[np.ix_(cols, cols)], matrix[cols, -1]
        beta = np.linalg.lstsq(xx, xy, rcond=None)[0]
        sse = float(yy - beta @ xy)
        np.testing.assert_allclose(100 * beta, fit['beta_pp'], atol=1e-7)
        np.testing.assert_allclose(sse, fit['sse'], atol=1e-7)
        np.testing.assert_allclose(1 - sse / yy, fit['within_r2'], atol=1e-10)
        independently[name] = (beta, sse)
    if not all(k in independently for k in ['A', 'B', 'AB']):
        assert not result['metrics']
        return 0
    m, full_sse = result['metrics'], independently['AB'][1]
    for name in ['A', 'B', 'AB']:
        np.testing.assert_allclose(m['R2_' + name], 1 - independently[name][1] / yy, atol=1e-10)
    for label, reduced in [('A_given_B', 'B'), ('B_given_A', 'A')]:
        improvement = independently[reduced][1] - full_sse
        np.testing.assert_allclose(m['delta_R2_' + label], improvement / yy, atol=1e-10)
        np.testing.assert_allclose(m['partial_R2_' + label], improvement / independently[reduced][1], atol=1e-10)
    for name, col in PRIMARY.items():
        other = [j for j in range(13) if j != col]
        xx = matrix[np.ix_(other, other)]
        ax = np.linalg.lstsq(xx, matrix[other, col], rcond=None)[0]
        ay = np.linalg.lstsq(xx, matrix[other, -1], rcond=None)[0]
        rx = float(matrix[col, col] - matrix[col, other] @ ax)
        ry = float(yy - matrix[-1, other] @ ay)
        rxy = float(matrix[col, -1] - matrix[col, other] @ ay)
        beta = 100 * rxy / rx
        expected = {'beta_pp': beta, 'partial_correlation': rxy / np.sqrt(rx * ry),
                    'delta_R2': rxy ** 2 / rx / yy, 'partial_R2': rxy ** 2 / rx / ry,
                    'remaining_weather_fraction': rx / matrix[col, col],
                    'remaining_weather_sd': np.sqrt(rx / rows) * (10 if 'cloud' in name else 1)}
        for metric, value in expected.items():
            np.testing.assert_allclose(m[name + '_' + metric], value, atol=1e-8)
        op = result['fits']['opposite_plus_' + name]['columns']
        beta_op = independently['opposite_plus_' + name][0][op.index(col)]
        np.testing.assert_allclose(m[name + '_opposite_beta_pp'], 100 * beta_op, atol=1e-7)
    for provider in ['A', 'B']:
        reduced = independently.get('without_' + provider + '_rain')
        if reduced:
            improvement = reduced[1] - full_sse
            np.testing.assert_allclose(m[provider + '_rain_delta_R2'], improvement / yy, atol=1e-10)
            np.testing.assert_allclose(m[provider + '_rain_partial_R2'], improvement / reduced[1], atol=1e-10)
    return len(independently)


def verify():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert pre['source_hashes'] == s['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert pre['code_hashes'] == s['code_hashes'] == {f: digest(ROOT / f) for f in DEPENDENCIES}
    snap = (OUT / 'preregistered_report.md').read_bytes().replace(b'\r\n', b'\n')
    assert pre['report_hash'] in [hashlib.sha256(snap).hexdigest(), hashlib.sha256(snap.replace(b'\n', b'\r\n')).hexdigest()]
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id')
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet').set_index(['site_id', 'ts'])
    sources = {provider: pd.read_parquet(SOURCE / f'train/forecast_{provider}.parquet') for provider in ['wa', 'wb']}
    audit = {'source_generation_cells': 0, 'transformed_weather_cells': 0, 'point_original_site_crosses': 0,
             'bootstrap_original_site_crosses': 0, 'nested_point_fits_checked': 0, 'date_weight_draws': 0,
             'rain_support_cases': 0, 'direct_partial_checks': 0}
    rng = np.random.default_rng(SEED)
    maximum_cross_error = 0.
    for label, case in s['cases'].items():
        round_number, year = map(int, label.split('_'))
        p = np.load(OUT / f'{label}_panel.npz')
        ids, pair_a, pair_b = p['site_ids'], p['wa'], p['wb']
        meta = sites.loc[ids]
        pair_lookup = list(zip(pair_a, pair_b))
        expected_map = np.array([pair_lookup.index(tuple(row)) for row in meta[['wa_station', 'wb_station']].to_numpy()])
        np.testing.assert_array_equal(expected_map, p['site_pair'])
        np.testing.assert_array_equal(np.bincount(expected_map, minlength=len(pair_a)), p['pair_weights'])
        # Prove maximum matching cardinality with an independent augmenting-path algorithm.
        adjacency = {a: [b for aa, b in pair_lookup if aa == a] for a in sorted(set(pair_a))}
        owned = {}
        def augment(a, visited):
            for b in adjacency[a]:
                if b in visited:
                    continue
                visited.add(b)
                if b not in owned or augment(owned[b], visited):
                    owned[b] = a
                    return True
            return False
        for a in adjacency:
            augment(a, set())
        chosen = p['matching']
        assert len(set(pair_a[chosen])) == len(set(pair_b[chosen])) == chosen.sum() == len(owned) == s['matching_pairs']
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', tz='Asia/Seoul')
        nt, npairs = len(dates), len(pair_a)
        blocks = []
        for provider, stations in [('wa', pair_a), ('wb', pair_b)]:
            frame = sources[provider].loc[sources[provider]['round'].eq(round_number)].copy()
            columns = [provider + '_temp', provider + ('_humidity' if provider == 'wa' else '_rel_hum'),
                       provider + '_cloud', provider + '_wind_speed', provider + '_rain']
            if provider == 'wb':
                columns.append('wb_uv_idx')
            for col in columns:
                frame.loc[~valid_values(frame[col], col), col] = np.nan
            frame = frame.set_index([provider + '_station', 'ts'])
            time = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for _ in stations for h in range(9, 17)])
            idx = pd.MultiIndex.from_arrays([np.tile(np.repeat(stations, 8), nt), time], names=frame.index.names)
            raw = frame.reindex(idx)[columns].to_numpy(copy=True).reshape(nt, npairs, 8, len(columns))
            fields = [raw[..., 0] / 10, raw[..., 1] / 10, raw[..., 2] / 10, raw[..., 3] / (3.6 if provider == 'wb' else 1),
                      (raw[..., 4] > .1).astype(float), np.where(raw[..., 4] > .1, np.log1p(raw[..., 4]), 0.)]
            if provider == 'wb':
                fields.append(raw[..., 5])
            block = np.stack(fields, axis=-1)
            block[~np.isfinite(raw).all(axis=-1)] = np.nan
            blocks.append(block)
        x = np.concatenate(blocks, axis=-1)
        times = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for _ in ids for h in range(9, 17)])
        keys = pd.MultiIndex.from_arrays([np.tile(np.repeat(ids, 8), nt), times], names=gen.index.names)
        ysite = gen.reindex(keys).energy.to_numpy().reshape(nt, len(ids), 8) / meta.capacity_kw.to_numpy()[None, :, None]
        good = np.isfinite(x).all(axis=(1, 2, 3)) & np.isfinite(ysite).all(axis=(1, 2)) & (ysite >= 0).all(axis=(1, 2))
        np.testing.assert_array_equal(dates[good].strftime('%Y-%m-%d').to_numpy(dtype=str), p['dates'])
        assert (~good).sum() == case['audit']['excluded_dates']
        x, ysite = x[good], ysite[good]
        months = dates[good].month.to_numpy()
        np.testing.assert_allclose(x, p['x'], atol=0, rtol=0)
        np.testing.assert_allclose(ysite, p['site_response'], atol=0, rtol=0)
        mean = np.stack([ysite[:, expected_map == j].mean(axis=1) for j in range(npairs)], axis=1)
        np.testing.assert_allclose(mean, p['pair_response'], atol=1e-15)
        audit['source_generation_cells'] += ysite.size
        audit['transformed_weather_cells'] += x.size
        wa, wb = x[..., 4] > .5, x[..., 10] > .5
        for a in [0, 1]:
            for b in [0, 1]:
                mask = (wa == a) & (wb == b)
                r = case['rain_support']['states'][f'A{a}_B{b}']
                assert mask.sum() == r['pair_hour_rows'] and mask.any(axis=(1, 2)).sum() == r['dates']
        for provider, target, other in [('A', wa, wb), ('B', wb, wa)]:
            frame = pd.DataFrame({'month': np.repeat(months, npairs * 8), 'pair': np.tile(np.repeat(np.arange(npairs), 8), len(months)),
                'hour': np.tile(np.arange(8), len(months) * npairs), 'other': other.ravel(), 'wet': target.ravel()})
            grouped = frame.groupby(['month', 'pair', 'hour', 'other']).wet.agg(['sum', 'size'])
            ncells = int(((grouped['sum'] >= 3) & (grouped['size'] - grouped['sum'] >= 3)).sum())
            disagree = target & ~other
            active = disagree.any(axis=(0, 2))
            n, days = int(disagree.sum()), int(disagree.any(axis=(1, 2)).sum())
            na, nb = len(set(pair_a[active])), len(set(pair_b[active]))
            r = case['rain_support'][provider]
            assert [n, days, na, nb, ncells] == [r[k] for k in ['disagreement_rows', 'disagreement_dates', 'a_stations', 'b_stations', 'supported_cells']]
            assert r['eligible'] == bool(n >= 200 and days >= 20 and na >= 5 and nb >= 5 and ncells >= 20)
            audit['rain_support_cases'] += 1
        boot = np.load(OUT / f'{label}_bootstrap.npz')
        for k, block in enumerate([7, 14]):
            for rep in range(REPS):
                starts = rng.integers(len(months), size=int(np.ceil(len(months) / block)))
                ix = np.concatenate([np.arange(v, v + block) % len(months) for v in starts])[:len(months)]
                np.testing.assert_array_equal(np.bincount(ix, minlength=len(months)), boot['weights'][k * REPS + rep])
                audit['date_weight_draws'] += 1
        original_x = x[:, expected_map]
        saved_crosses = np.load(OUT / f'{label}_cross.npz')
        selected_sites = chosen[expected_map]
        for effect, result in case['effects'].items():
            date_effect = effect == 'date_hour'
            for matched in [False, True]:
                xx = original_x[:, selected_sites] if matched else original_x
                yy = ysite[:, selected_sites] if matched else ysite
                mat = direct_cross(xx, yy, np.ones(yy.shape, dtype=bool), np.ones(yy.shape[1]),
                                   months, np.ones(len(months)), date_effect)
                key = effect + ('_matching' if matched else '')
                error = np.max(abs(mat - saved_crosses[key]))
                maximum_cross_error = max(maximum_cross_error, float(error))
                np.testing.assert_allclose(mat, saved_crosses[key], atol=3e-7, rtol=1e-8)
                audit['nested_point_fits_checked'] += check_metrics(mat, result['matching'] if matched else result, yy.size)
                audit['point_original_site_crosses'] += 1
                audit['direct_partial_checks'] += 3
            metric_names = list(boot['metric_names'])
            for k, block in enumerate([7, 14]):
                weights = boot['weights'][k * REPS]
                mat = direct_cross(original_x, ysite, np.ones(ysite.shape, dtype=bool), np.ones(len(ids)), months, weights, date_effect)
                # Independently compare primary coefficients for the first draw of each block.
                beta = np.linalg.lstsq(mat[:-1, :-1], mat[:-1, -1], rcond=None)[0] * 100
                for name, col in PRIMARY.items():
                    np.testing.assert_allclose(beta[col], boot[effect][k * REPS, metric_names.index(name + '_beta_pp')], atol=1e-7)
                audit['bootstrap_original_site_crosses'] += 1
                values = boot[effect][k * REPS:(k + 1) * REPS]
                for j, name in enumerate(metric_names):
                    good = np.isfinite(values[:, j])
                    r = result['bootstrap'][str(block)][name]
                    assert good.sum() == r['success']
                    if good.sum() >= .9 * REPS:
                        np.testing.assert_allclose(np.quantile(values[good, j], [.025, .975]), r['interval'], atol=1e-12)
    audit['max_original_site_cross_difference'] = maximum_cross_error
    audit['source_hashes_unchanged'] = True
    audit['all_saved_intervals_checked'] = True
    dump(OUT / 'verification.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
