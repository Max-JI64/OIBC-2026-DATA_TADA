"""Rebuild source keys, support and profile comparisons; verify fits by direct SVD."""
import json
import hashlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cohort_applicability import ROOT, SOURCE, OUT, FILES, WEATHER, HOURS, SEED, BOOT_REPS
from weather_latent_structure import digest, dump, valid_values


def direct_fit(x, y, mask, months, weights):
    rx, ry = x.copy(), y.copy()
    for month in np.unique(months):
        take = months == month
        wm = mask[take] * weights[take, None, None]
        n = wm.sum(axis=0)
        mx = np.divide((x[take] * wm[..., None]).sum(axis=0), n[..., None],
                       out=np.zeros(x.shape[1:]), where=n[..., None] > 0)
        my = np.divide((y[take] * wm).sum(axis=0), n, out=np.zeros(y.shape[1:]), where=n > 0)
        rx[take] -= mx
        ry[take] -= my
    active = mask & (weights[:, None, None] > 0)
    sw = np.broadcast_to(np.sqrt(weights)[:, None, None], mask.shape)[active]
    a = rx[active] * sw[:, None]
    b = ry[active] * sw
    beta = np.linalg.lstsq(a, b, rcond=None)[0]
    r2 = 1 - np.sum((b - a @ beta) ** 2) / np.sum(b ** 2)
    return beta * 100, r2


def verify():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert pre['source_hashes'] == s['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert pre['code_hash'] == s['code_hash'] == digest(ROOT / 'scripts/cohort_applicability.py')
    snapshot = (OUT / 'preregistered_report.md').read_bytes().replace(b'\r\n', b'\n')
    assert pre['report_hash'] in [hashlib.sha256(snapshot).hexdigest(),
                                  hashlib.sha256(snapshot.replace(b'\n', b'\r\n')).hexdigest()]
    p = np.load(OUT / 'panel.npz')
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id').loc[p['site_ids']]
    dates = pd.to_datetime(p['dates']).tz_localize('Asia/Seoul')
    months = dates.month.to_numpy()
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet').set_index(['site_id', 'ts'])
    assert gen.index.is_unique
    local_start = gen.index.get_level_values('ts') - pd.Timedelta(hours=1)
    counts = pd.Series(1, index=gen.index).groupby([gen.index.get_level_values('site_id'), local_start.year]).sum().unstack(fill_value=0).reindex(sites.index, fill_value=0)
    expected_cohort = np.where((counts[2023] == 8760) & (counts[2024] == 8784), 'L80',
                               np.where(counts[2024] == 8784, 'M68', 'S105'))
    np.testing.assert_array_equal(expected_cohort, p['cohorts'])
    ns = len(sites)
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=int(h + 1)) for d in dates for _ in range(ns) for h in HOURS])
    ids = np.tile(np.repeat(sites.index.to_numpy(), len(HOURS)), len(dates))
    idx = pd.MultiIndex.from_arrays([ids, times], names=gen.index.names)
    energy = gen.reindex(idx).energy.to_numpy().reshape(len(dates), ns, len(HOURS))
    cf = energy / sites.capacity_kw.to_numpy()[None, :, None]
    np.testing.assert_allclose(cf, p['cf'], atol=0, rtol=0, equal_nan=True)
    complete = np.isfinite(cf).all(axis=2) & (cf >= 0).all(axis=2)
    np.testing.assert_array_equal(complete, p['complete'])
    eligible = np.zeros_like(complete)
    for month in range(1, 13):
        take = months == month
        eligible[take] = complete[take].sum(axis=0) >= 14
    np.testing.assert_array_equal(eligible, p['eligible'])
    present = idx.isin(gen.index)
    assert (~present).sum() == s['audit']['missing_source_keys']
    assert (cf == 0).sum() == s['audit']['valid_zero_cells']
    assert (cf > 1).sum() == s['audit']['above_one_cells']
    # Independently align weather site/station keys in date/site/hour order.
    weather = pd.read_parquet(SOURCE / 'train/observed_wb.parquet', columns=['wb_station', 'ts'] + WEATHER)
    for col in WEATHER:
        weather.loc[~valid_values(weather[col], col), col] = np.nan
    weather = weather.set_index(['wb_station', 'ts'])
    daytime = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in dates for _ in range(ns) for h in range(9, 17)])
    keys = np.tile(np.repeat(sites.wb_station.to_numpy(), 8), len(dates))
    index = pd.MultiIndex.from_arrays([keys, daytime], names=weather.index.names)
    w = weather.reindex(index)[WEATHER].to_numpy(copy=True).reshape(len(dates), ns, 8, 6)
    w[..., 3] /= 3.6
    np.testing.assert_allclose(w, p['weather'], atol=0, rtol=0, equal_nan=True)
    source = np.flatnonzero(expected_cohort == 'L80')
    unique_stations = sorted(sites.wb_station.iloc[source].unique())
    members = [source[sites.wb_station.iloc[source].to_numpy() == st] for st in unique_stations]
    refs = np.stack([w[:, indexes[0]] for indexes in members], axis=2)
    refcf = np.stack([cf[:, indexes, 2:10].mean(axis=1) for indexes in members], axis=-1)
    support = np.zeros_like(p['support'])
    refcount = np.zeros_like(p['reference_count'])
    benchmark = np.full(support.shape, np.nan)
    limits = [15., 15., 5., 2., 1.5]
    for i in range(ns):
        valid = np.isfinite(w[:, i]).all(axis=-1)[..., None] & np.isfinite(refs).all(axis=-1)
        for k, limit in enumerate(limits):
            valid &= abs(w[:, i, :, None, k] - refs[..., k]) <= limit
        valid &= (w[:, i, :, None, 5] > .1) == (refs[..., 5] > .1)
        valid &= np.array(unique_stations) != sites.wb_station.iloc[i]
        valid &= complete[:, i, None, None] & eligible[:, i, None, None]
        n = valid.sum(axis=-1)
        refcount[:, i] = n
        support[:, i] = n >= 2
        for h in range(8):
            values = np.sum(np.where(valid[:, h], refcf[:, h], 0), axis=1)
            benchmark[:, i, h] = np.divide(values, n[:, h], out=np.full(len(dates), np.nan), where=n[:, h] >= 2)
    np.testing.assert_array_equal(refcount, p['reference_count'])
    np.testing.assert_array_equal(support, p['support'])
    np.testing.assert_allclose(benchmark, p['benchmark'], atol=1e-14, rtol=0, equal_nan=True)
    passed = np.zeros_like(support)
    for r in s['support']:
        i = sites.index.get_loc(r['site_id'])
        take = (months == r['month']) & eligible[:, i]
        good = support[take, i]
        total = 8 * int(complete[take, i].sum())
        ok = good.sum() >= total / 2 and good.any(axis=1).sum() >= 14 and (good.sum(axis=0) >= 7).all()
        assert total == r['candidate_rows'] and good.sum() == r['supported_rows'] and bool(ok) == r['passed']
        if ok:
            passed[take, i] = good
            np.testing.assert_allclose(100 * np.nanmean(cf[take, i, 2:10] - benchmark[take, i]), r['cf_diff_pp'])
    np.testing.assert_array_equal(passed, p['passed'])
    for r in s['profiles']:
        i = sites.index.get_loc(r['site_id'])
        take = (months == r['month']) & eligible[:, i] & complete[:, i]
        peers = source if r['reference'] == 'all_L' else source[sites.wb_station.iloc[source].to_numpy() == sites.wb_station.iloc[i]]
        a = np.nanmean(cf[take, i], axis=0)
        b = np.nanmean(np.nanmean(cf[take][:, peers], axis=1), axis=0)
        ashape, bshape = a / a.sum(), b / b.sum()
        np.testing.assert_allclose(ashape, r['target_shape'], atol=1e-14)
        np.testing.assert_allclose(bshape, r['reference_shape'], atol=1e-14)
        diff = [100 * (ashape[:5].sum() - bshape[:5].sum()),
                60 * ((ashape - bshape) @ HOURS), 100 * (a[2:10].mean() - b[2:10].mean())]
        np.testing.assert_allclose(diff, [r['morning_diff_pp'], r['center_diff_minutes'], r['cf_diff_pp']], atol=1e-11)
    boot = np.load(OUT / 'bootstrap.npz')
    rng = np.random.default_rng(SEED)
    n = int((months >= 7).sum())
    counter = 0
    for length in [7, 14]:
        for _ in range(BOOT_REPS):
            starts = rng.integers(n, size=int(np.ceil(n / length)))
            indices = np.concatenate([(np.arange(v, v + length) % n) for v in starts])[:n]
            np.testing.assert_array_equal(np.bincount(indices, minlength=n), boot['weights'][counter])
            counter += 1
    errors = []
    fits = 0
    for cohort, result in s['regression'].items():
        if not result['valid']:
            continue
        f = np.load(OUT / f'fit_{cohort}.npz')
        pick = f['site_indices']
        raw = w[months >= 7][:, pick]
        x = np.stack([raw[..., 0] / 10, raw[..., 1] / 10, raw[..., 2] / 10,
                      raw[..., 3], raw[..., 4], (raw[..., 5] > .1).astype(float),
                      np.where(raw[..., 5] > .1, np.log1p(raw[..., 5]), 0)], axis=-1)
        x = np.nan_to_num(x, nan=0.)
        np.testing.assert_allclose(x, f['x'], atol=0)
        np.testing.assert_allclose(cf[months >= 7][:, pick, 2:10], f['y'], atol=0, equal_nan=True)
        np.testing.assert_array_equal(passed[months >= 7][:, pick], f['mask'])
        # Missing responses are only outside the mask. Use zeros there for residualization.
        y = np.where(f['mask'], f['y'], 0.)
        x = np.where(f['mask'][..., None], f['x'], 0.)
        beta, r2 = direct_fit(x, y, f['mask'], f['months'], np.ones(n))
        errors.append(float(np.max(abs(beta - result['beta_pp']))))
        np.testing.assert_allclose(beta, result['beta_pp'], atol=1e-9)
        np.testing.assert_allclose(r2, result['within_r2'], atol=1e-11)
        fits += 1
        saved = boot['beta_' + cohort]
        for number in [0, BOOT_REPS]:
            beta, _ = direct_fit(x, y, f['mask'], f['months'], boot['weights'][number])
            np.testing.assert_allclose(beta, saved[number], atol=1e-9)
            fits += 1
        for k, length in enumerate([7, 14]):
            v = saved[k * BOOT_REPS:(k + 1) * BOOT_REPS]
            take = np.isfinite(v).all(axis=1)
            assert take.sum() == result['bootstrap'][str(length)]['success']
            np.testing.assert_allclose(np.quantile(v[take], [.025, .975], axis=0).T,
                                       result['bootstrap'][str(length)]['intervals_pp'], atol=1e-12)
    for cohort, result in s['contrasts'].items():
        np.testing.assert_allclose(np.array(s['regression'][cohort]['beta_pp']) - s['regression']['L80']['beta_pp'],
                                   result['difference_pp'])
        diff = boot['beta_' + cohort] - boot['beta_L80']
        for k, length in enumerate([7, 14]):
            v = diff[k * BOOT_REPS:(k + 1) * BOOT_REPS]
            np.testing.assert_allclose(np.quantile(v, [.025, .975], axis=0).T,
                                       result['bootstrap'][str(length)]['intervals_pp'], atol=1e-12)
    out = {'source_generation_cells': int(cf.size), 'source_weather_cells': int(w.size),
           'profile_comparisons': len(s['profiles']), 'weather_site_months': len(s['support']),
           'direct_svd_fits': fits, 'all_bootstrap_weight_draws': counter,
           'max_point_beta_difference_pp': max(errors), 'source_hashes_unchanged': True,
           'all_bootstrap_intervals_checked': True}
    dump(OUT / 'verification.json', out)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
