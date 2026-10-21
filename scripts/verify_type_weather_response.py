"""DA-012 source reconstruction, balanced projections and explicit dummy checks."""
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from type_weather_response import ROOT, SOURCE, OUT, TYPE_SOURCE, FILES, DEPENDENCIES, SEED, REPS
from weather_latent_structure import digest, dump, valid_values
from verify_conditional_support_date import direct_cross


def project_cross(x, y, months, weights, date):
    """Direct original-observation weighted demeaning for a balanced rectangle."""
    z = np.concatenate([x, y[..., None]], axis=-1).copy()
    for month in np.unique(months):
        take = months == month
        n = weights[take].sum()
        if n:
            z[take] -= np.tensordot(weights[take], z[take], axes=1) / n
        else:
            z[take] = 0.
    if date:
        z -= z.mean(axis=1, keepdims=True)
    return np.einsum('d,dshp,dshq->pq', weights, z, z, optimize=True)


def linear_slope(mat):
    return 100 * np.linalg.lstsq(mat[:-1, :-1], mat[:-1, -1], rcond=None)[0]


def compare_fit(mat, record, source):
    if not record['fit']['valid']:
        return
    beta = linear_slope(mat)
    np.testing.assert_allclose(beta, record['fit']['beta_pp'], atol=1e-7)
    expected = {}
    for j, variable in enumerate(['cloud', 'humidity']):
        if source == 'all':
            expected['C1_' + variable] = beta[j]
            expected['C2_' + variable] = beta[j + 7]
            expected['delta_' + variable] = beta[j + 7] - beta[j]
        else:
            expected['delta_' + variable] = beta[j]
    for key, value in expected.items():
        np.testing.assert_allclose(value, record['metrics'][key], atol=1e-7)


def gap_data(weather_x, response, labels, site_stations, mixed):
    indexes, gaps = [], []
    for st in mixed:
        c1 = np.flatnonzero((site_stations == st) & (labels == 0))
        c2 = np.flatnonzero((site_stations == st) & (labels == 1))
        indexes.append(int(c1[0]))
        gaps.append(response[:, c2].mean(axis=1) - response[:, c1].mean(axis=1))
    return weather_x[:, indexes], np.stack(gaps, axis=1)


def verify():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert pre['source_hashes'] == s['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert pre['code_hashes'] == s['code_hashes'] == {f: digest(ROOT / f) for f in DEPENDENCIES}
    assert pre['type_source_hash'] == s['type_source_hash'] == digest(TYPE_SOURCE)
    snap = (OUT / 'preregistered_report.md').read_bytes().replace(b'\r\n', b'\n')
    assert pre['report_hash'] in [hashlib.sha256(snap).hexdigest(), hashlib.sha256(snap.replace(b'\n', b'\r\n')).hexdigest()]
    typology = json.loads(TYPE_SOURCE.read_text(encoding='utf-8'))
    np.testing.assert_array_equal(s['labels'], typology['diagnostic_partition']['labels'])
    assert s['site_ids'] == typology['site_ids']
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id').loc[s['site_ids']]
    labels = np.array(s['labels'])
    stations = sites.wb_station.to_numpy()
    mixed = [st for st in sorted(set(stations)) if set(labels[stations == st]) == {0, 1}]
    sturdy = [st for st in mixed if min(np.bincount(labels[stations == st], minlength=2)) >= 2]
    assert mixed == s['mixed_stations'] and sturdy == s['sturdy_stations']
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet').set_index(['site_id', 'ts'])
    if not gen.index.is_unique:
        raise ValueError('duplicate original generation key')
    reference_days = pd.date_range('2023-01-01', '2023-12-31', tz='Asia/Seoul')
    def source_y(dates):
        times = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for _ in range(len(sites)) for h in range(9, 17)])
        keys = pd.MultiIndex.from_arrays([np.tile(np.repeat(sites.index, 8), len(dates)), times], names=gen.index.names)
        return gen.reindex(keys).energy.to_numpy().reshape(len(dates), len(sites), 8) / sites.capacity_kw.to_numpy()[None, :, None]
    reference = source_y(reference_days).mean(axis=(0, 2))
    np.testing.assert_allclose(reference, s['reference_scale_cf'], atol=1e-14)
    rng = np.random.default_rng(SEED)
    audit = {'source_generation_cells': 0, 'source_weather_cells': 0, 'point_crosses': 0,
             'sensitivity_point_fits': 0, 'bootstrap_crosses': 0, 'explicit_dummy_month_hour_checks': 0,
             'date_weight_draws': 0, 'max_point_cross_error': 0.}
    for case_name, case in s['cases'].items():
        kind, year = case_name.split('_')
        year = int(year)
        p = np.load(OUT / f'{case_name}_panel.npz')
        days = pd.date_range(f'{year}-01-01', f'{year}-12-31', tz='Asia/Seoul')
        months = days.month.to_numpy()
        response = source_y(days)
        np.testing.assert_allclose(response, p['response'], atol=0, rtol=0)
        np.testing.assert_array_equal(months, p['months'])
        np.testing.assert_array_equal(labels, p['labels'])
        rain = 'wb_precip_1h' if kind == 'observed' else 'wb_rain'
        cols = ['wb_cloud', 'wb_rel_hum', 'wb_temp', 'wb_wind_speed', 'wb_uv_idx', rain]
        frame = pd.read_parquet(SOURCE / f'train/{kind}_wb.parquet')
        if kind == 'forecast':
            frame = frame.loc[frame['round'].eq(17)].copy()
        for col in cols:
            frame.loc[~valid_values(frame[col], col), col] = np.nan
        frame = frame.set_index(['wb_station', 'ts'])
        time = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in days for _ in stations for h in range(9, 17)])
        idx = pd.MultiIndex.from_arrays([np.tile(np.repeat(stations, 8), len(days)), time], names=frame.index.names)
        weather = frame.reindex(idx)[cols].to_numpy(copy=True).reshape(len(days), len(sites), 8, 6)
        weather[..., 3] /= 3.6
        np.testing.assert_allclose(weather, p['weather'], atol=0, rtol=0)
        assert np.isfinite(weather).all()
        wx = np.stack([weather[..., 0] / 10, weather[..., 1] / 10, weather[..., 2] / 10,
                       weather[..., 3], weather[..., 4], (weather[..., 5] > .1).astype(float),
                       np.where(weather[..., 5] > .1, np.log1p(weather[..., 5]), 0)], axis=-1)
        fullx = np.zeros((*wx.shape[:-1], 14))
        fullx[:, labels == 0, :, :7] = wx[:, labels == 0]
        fullx[:, labels == 1, :, 7:] = wx[:, labels == 1]
        audit['source_generation_cells'] += response.size
        audit['source_weather_cells'] += weather.size
        boot = np.load(OUT / f'{case_name}_bootstrap.npz')
        for k, block in enumerate([7, 14]):
            for rep in range(REPS):
                starts = rng.integers(len(days), size=int(np.ceil(len(days) / block)))
                ix = np.concatenate([np.arange(start, start + block) % len(days) for start in starts])[:len(days)]
                np.testing.assert_array_equal(np.bincount(ix, minlength=len(days)), boot['weights'][k * REPS + rep])
                audit['date_weight_draws'] += 1
        saved = np.load(OUT / f'{case_name}_cross.npz')
        for mode in ['absolute', 'relative']:
            y = response if mode == 'absolute' else response / reference[None, :, None]
            sx, sy = gap_data(wx, y, labels, stations, mixed)
            for source, x, yr in [('all', fullx, y), ('shared', sx, sy)]:
                sample = source + '_' + mode
                names = list(boot[sample + '_names'])
                for effect in ['calendar', 'date_hour']:
                    date = effect == 'date_hour'
                    mat = project_cross(x, yr, months, np.ones(len(days)), date)
                    expected = saved[sample + '_' + effect]
                    # Group means preserve the X/X and X/Y blocks; Y/Y differs for all-site compression.
                    got, want = (mat[:-1], expected[:-1]) if source == 'all' else (mat, expected)
                    error = float(np.max(abs(got - want)))
                    audit['max_point_cross_error'] = max(audit['max_point_cross_error'], error)
                    np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-8)
                    record = case['samples'][sample][effect]
                    compare_fit(mat, record, source)
                    audit['point_crosses'] += 1
                    take = months == 7
                    shortx, shorty = x[take, :, 3:4], yr[take, :, 3:4]
                    m = np.ones(take.sum(), dtype=int)
                    explicit = direct_cross(shortx, shorty, np.ones(shorty.shape, dtype=bool), np.ones(shorty.shape[1]),
                                            m, np.ones(take.sum()), date)
                    direct = project_cross(shortx, shorty, m, np.ones(take.sum()), date)
                    np.testing.assert_allclose(direct, explicit, atol=1e-8, rtol=1e-9)
                    audit['explicit_dummy_month_hour_checks'] += 1
                    for k, block in enumerate([7, 14]):
                        number = k * REPS
                        mat = project_cross(x, yr, months, boot['weights'][number], date)
                        beta = linear_slope(mat)
                        direct_values = {'delta_cloud': beta[7] - beta[0], 'delta_humidity': beta[8] - beta[1],
                                         'C1_cloud': beta[0], 'C1_humidity': beta[1], 'C2_cloud': beta[7], 'C2_humidity': beta[8]} if source == 'all' else {
                                             'delta_cloud': beta[0], 'delta_humidity': beta[1]}
                        np.testing.assert_allclose([direct_values[n] for n in names], boot[sample + '_' + effect][number], atol=1e-7)
                        audit['bootstrap_crosses'] += 1
                        values = boot[sample + '_' + effect][k * REPS:(k + 1) * REPS]
                        for col, name in enumerate(names):
                            ok = np.isfinite(values[:, col])
                            r = record['bootstrap'][str(block)][name]
                            assert r['success'] == ok.sum()
                            if ok.sum() >= .9 * REPS:
                                np.testing.assert_allclose(np.quantile(values[ok, col], [.025, .975]), r['interval'], atol=1e-12)
            for effect in ['calendar', 'date_hour']:
                sens = case['point_sensitivities'][mode + '_' + effect]
                date = effect == 'date_hour'
                if 'min_two_each' in sens:
                    tx, ty = gap_data(wx, y, labels, stations, sturdy)
                    mat = project_cross(tx, ty, months, np.ones(len(days)), date)
                    compare_fit(mat, sens['min_two_each'], 'shared')
                    audit['sensitivity_point_fits'] += 1
                for rec in sens['leave_one_station_out']:
                    keep = [st for st in mixed if st != rec['omitted_station']]
                    tx, ty = gap_data(wx, y, labels, stations, keep)
                    mat = project_cross(tx, ty, months, np.ones(len(days)), date)
                    compare_fit(mat, rec, 'shared')
                    audit['sensitivity_point_fits'] += 1
    audit['source_hashes_unchanged'] = True
    audit['all_saved_intervals_checked'] = True
    audit['frozen_typology_and_reference_checked'] = True
    dump(OUT / 'verification.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
