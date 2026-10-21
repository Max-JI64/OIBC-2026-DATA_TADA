"""DA-010: observation coverage and matched-cohort descriptive transport limits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log, valid_values

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA/01_participant_dataset'
OUT = ROOT / 'data/analysis/2026-10-21_cohort_applicability'
REPORT = ROOT / 'reports/2026-10-21_12_장기관측_표본의_적용범위_분석.md'
FILES = ['sites.csv', 'train/generation.parquet', 'train/observed_wb.parquet']
HOURS = np.arange(7, 18)
WEATHER = ['wb_cloud', 'wb_rel_hum', 'wb_temp', 'wb_wind_speed', 'wb_uv_idx', 'wb_precip_1h']
SEED = 2026102110
BOOT_REPS = 200
COHORTS = ['L80', 'M68', 'S105']
TERMS = ['cloud10', 'humidity10', 'temp10', 'wind_ms', 'uv', 'wet', 'log_rain']


def load():
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').sort_values('site_id').reset_index(drop=True)
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet', columns=['site_id', 'ts', 'energy'])
    if gen.duplicated(['site_id', 'ts']).any():
        raise ValueError('duplicate generation site/time')
    if not gen.ts.eq(gen.ts.dt.floor('h')).all():
        raise ValueError('nonhour generation time')
    if not sites.site_id.is_unique or not set(gen.site_id).issubset(set(sites.site_id)):
        raise ValueError('invalid site keys')
    start = gen.ts - pd.Timedelta(hours=1)
    gen = gen.loc[start.dt.year.isin([2023, 2024])].copy()
    gen['start'] = gen.ts - pd.Timedelta(hours=1)
    counts = gen.groupby([gen.site_id, gen.start.dt.year]).size().unstack(fill_value=0).reindex(sites.site_id, fill_value=0)
    full24 = counts[2024].eq(8784)
    long = full24 & counts[2023].eq(8760)
    sites['cohort'] = np.where(long, 'L80', np.where(full24, 'M68', 'S105'))
    dates = pd.date_range('2023-01-01', '2024-12-31', tz='Asia/Seoul', freq='D')
    grid = pd.DatetimeIndex([d + pd.Timedelta(hours=int(h)) for d in dates for h in HOURS])
    raw = gen.set_index(['start', 'site_id']).energy.unstack('site_id').reindex(index=grid, columns=sites.site_id)
    energy = raw.to_numpy(copy=True).reshape(len(dates), len(HOURS), len(sites)).transpose(0, 2, 1)
    cf = energy / sites.capacity_kw.to_numpy()[None, :, None]
    valid = np.isfinite(cf) & (cf >= 0)
    if not (np.isfinite(sites.capacity_kw) & sites.capacity_kw.gt(0)).all():
        raise ValueError('invalid capacity')
    complete = valid.all(axis=2)
    w = pd.read_parquet(SOURCE / 'train/observed_wb.parquet', columns=['wb_station', 'ts'] + WEATHER)
    if w.duplicated(['wb_station', 'ts']).any():
        raise ValueError('duplicate B observed station/time')
    w['start'] = w.ts - pd.Timedelta(hours=1)
    w = w.loc[w.start.dt.year.isin([2023, 2024]) & w.start.dt.hour.isin(HOURS) & w.ts.eq(w.ts.dt.floor('h'))].copy()
    for col in WEATHER:
        w.loc[~valid_values(w[col], col), col] = np.nan
    stations = sorted(sites.wb_station.unique())
    arrays = []
    for col in WEATHER:
        arr = w.set_index(['start', 'wb_station'])[col].unstack('wb_station').reindex(index=grid, columns=stations)
        arrays.append(arr.to_numpy(copy=True).reshape(len(dates), len(HOURS), len(stations)).transpose(0, 2, 1))
    weather_stations = np.stack(arrays, axis=-1)
    station_index = np.array([stations.index(st) for st in sites.wb_station])
    weather_sites = weather_stations[:, station_index]
    return sites, gen, dates, cf, complete, weather_sites, weather_stations, stations


def inspect():
    OUT.mkdir(parents=True, exist_ok=True)
    sites, gen, dates, cf, complete, weather, station_weather, stations = load()
    # Availability only: no response means, shapes or association estimates.
    months = dates.strftime('%Y-%m').to_numpy()
    source_stations = set(sites.loc[sites.cohort.eq('L80'), 'wb_station'])
    source_n = sites.loc[sites.cohort.eq('L80')].groupby('wb_station').size()
    table = []
    for i, row in sites.iterrows():
        for month in sorted(set(months)):
            take = months == month
            n = int(complete[take, i].sum())
            daylight = np.isfinite(weather[take, i, 2:10]).all(axis=(1, 2))
            table.append({'site_id': row.site_id, 'cohort': row.cohort, 'month': month,
                          'days': n, 'joint_weather_days': int((complete[take, i] & daylight).sum())})
    availability = pd.DataFrame(table)
    availability.to_csv(OUT / 'monthly_availability.csv', index=False, encoding='utf-8-sig')
    info = {'python': sys.version, 'executable': sys.executable, 'source_hashes': {f: digest(SOURCE / f) for f in FILES},
            'cohorts': {}, 'months': {}, 'valid_zero_retained': True,
            'site_ids': sites.site_id.tolist(), 'labels': sites.cohort.tolist(),
            'unavailable_or_invalid_daylight_cells': int((~np.isfinite(cf) | (cf < 0)).sum())}
    for cohort in ['L80', 'M68', 'S105']:
        sub = sites.loc[sites.cohort.eq(cohort)]
        a = availability.loc[availability.cohort.eq(cohort)]
        gen_sub = gen.loc[gen.site_id.isin(sub.site_id)]
        info['cohorts'][cohort] = {'sites': len(sub), 'current_groups': sub.current_group.value_counts().to_dict(),
            'wb_stations': int(sub.wb_station.nunique()), 'wa_stations': int(sub.wa_station.nunique()),
            'stations_shared_with_L': len(set(sub.wb_station) & source_stations),
            'sites_with_L_peer': int(sub.wb_station.isin(source_stations).sum()),
            'sites_with_two_L_peers': int(sub.wb_station.map(source_n).fillna(0).ge(2).sum()),
            'first_start': str(gen_sub.start.min()), 'last_start': str(gen_sub.start.max())}
        info['months'][cohort] = []
        for month, am in a.groupby('month', sort=True):
            info['months'][cohort].append({'month': month, 'sites_ge14days': int(am.days.ge(14).sum()),
                'sites_ge14joint': int(am.joint_weather_days.ge(14).sum()), 'min_days': int(am.days.min()),
                'median_days': float(am.days.median()), 'max_days': int(am.days.max())})
    for year in [2023, 2024]:
        t = dates.year == year
        full = complete[t].all(axis=0)
        info[f'complete_day_profiles_{year}'] = int(full.sum())
    dump(OUT / 'inspection.json', info)
    small = {k: v for k, v in info.items() if k not in ['source_hashes', 'site_ids', 'labels']}
    print(json.dumps(small, ensure_ascii=False, indent=2), flush=True)


def profile_metrics(profile):
    profile = np.asarray(profile)
    total = profile.sum()
    if not np.isfinite(profile).all() or total <= 1e-10:
        return None
    shape = profile / total
    return np.array([100 * shape[:5].sum(), float(shape @ HOURS), 100 * profile[2:10].mean()])


def monthly_eligibility(complete, months):
    out = np.zeros_like(complete, dtype=bool)
    for m in np.unique(months):
        take = months == m
        out[take] = complete[take].sum(axis=0) >= 14
    return out


def weather_caliper(target, references):
    """Target [...,6], references [...,n,6], wind already in m/s."""
    tolerances = np.array([15., 15., 5., 2., 1.5])
    ok = np.isfinite(target).all(axis=-1)[..., None] & np.isfinite(references).all(axis=-1)
    ok &= (np.abs(target[..., None, :5] - references[..., :5]) <= tolerances).all(axis=-1)
    ok &= (target[..., None, 5] > .1) == (references[..., 5] > .1)
    return ok


def profile_comparisons(sites, dates, cf, complete, eligible):
    months = dates.month.to_numpy()
    source = np.flatnonzero(sites.cohort.eq('L80'))
    records = []
    for i, site in sites.iterrows():
        if site.cohort == 'L80':
            continue
        peers = source[sites.wb_station.iloc[source].to_numpy() == site.wb_station]
        for m in range(1, 13):
            take = (months == m) & eligible[:, i] & complete[:, i]
            if not take.any():
                continue
            target = cf[take, i].mean(axis=0)
            tm = profile_metrics(target)
            for label, refs in [('all_L', source), ('same_B_L', peers)]:
                if label == 'same_B_L' and len(refs) < 2:
                    continue
                reference = cf[take][:, refs].mean(axis=(0, 1))
                rm = profile_metrics(reference)
                if tm is None or rm is None:
                    continue
                diff = tm - rm
                records.append({'site_id': site.site_id, 'cohort': site.cohort, 'month': m,
                    'reference': label, 'dates': int(take.sum()), 'reference_sites': len(refs),
                    'target_morning': float(tm[0]), 'target_center': float(tm[1]), 'target_cf': float(tm[2]),
                    'reference_morning': float(rm[0]), 'reference_center': float(rm[1]), 'reference_cf': float(rm[2]),
                    'morning_diff_pp': float(diff[0]), 'center_diff_minutes': float(60 * diff[1]),
                    'cf_diff_pp': float(diff[2]), 'target_shape': (target / target.sum()).tolist(),
                    'reference_shape': (reference / reference.sum()).tolist()})
    return records


def matched_support(sites, dates, cf, complete, weather, station_weather, stations, eligible):
    source = np.flatnonzero(sites.cohort.eq('L80'))
    source_stations = sorted(sites.wb_station.iloc[source].unique())
    ref_indices = np.array([stations.index(st) for st in source_stations])
    refw = station_weather[:, ref_indices, 2:10].transpose(0, 2, 1, 3).copy()
    refw[..., 3] /= 3.6
    w = weather[:, :, 2:10].copy()
    w[..., 3] /= 3.6
    refcf = np.stack([cf[:, source[sites.wb_station.iloc[source].to_numpy() == st], 2:10].mean(axis=1)
                      for st in source_stations], axis=-1)
    mask = np.zeros(cf[:, :, 2:10].shape, dtype=bool)
    count = np.zeros(mask.shape, dtype=np.int16)
    benchmark = np.full(mask.shape, np.nan)
    for i, site in sites.iterrows():
        ok = weather_caliper(w[:, i], refw)
        ok[..., np.array(source_stations) == site.wb_station] = False
        ok &= complete[:, i, None, None] & eligible[:, i, None, None]
        n = ok.sum(axis=-1)
        count[:, i] = n
        mask[:, i] = n >= 2
        values = np.divide((ok * refcf).sum(axis=-1), n, out=np.full(n.shape, np.nan), where=n > 0)
        benchmark[:, i] = np.where(n >= 2, values, np.nan)
    passed = np.zeros_like(mask)
    records = []
    for i, site in sites.iterrows():
        for m in range(1, 13):
            take = (dates.month == m) & eligible[:, i]
            candidate = complete[take, i]
            total = int(candidate.sum()) * 8
            if total == 0:
                continue
            supported = mask[take, i]
            good = bool(supported.sum() >= .5 * total and supported.any(axis=1).sum() >= 14
                        and (supported.sum(axis=0) >= 7).all())
            if good:
                passed[take, i] = supported
            diff = cf[take, i, 2:10] - benchmark[take, i]
            records.append({'site_id': site.site_id, 'cohort': site.cohort, 'month': m,
                'candidate_rows': total, 'supported_rows': int(supported.sum()),
                'supported_dates': int(supported.any(axis=1).sum()), 'support_fraction': float(supported.sum() / total),
                'passed': good, 'cf_diff_pp': float(np.nanmean(diff) * 100) if good else None,
                'mean_reference_stations': float(count[take, i][supported].mean()) if supported.any() else None})
    return w, mask, passed, count, benchmark, records


def linear_design(w):
    raw = np.nan_to_num(w, nan=0.)
    return np.stack([raw[..., 0] / 10, raw[..., 1] / 10, raw[..., 2] / 10,
                     raw[..., 3], raw[..., 4], (raw[..., 5] > .1).astype(float),
                     np.where(raw[..., 5] > .1, np.log1p(raw[..., 5]), 0.)], axis=-1)


def masked_stats(x, y, mask, months):
    x = np.where(mask[..., None], x, 0.).reshape(len(x), -1, x.shape[-1])
    y = np.where(mask, y, 0.).reshape(len(y), -1)
    z = mask.reshape(len(mask), -1).astype(float)
    return {'x': x, 'y': y, 'z': z, 'months': months,
            'xx_day': np.einsum('dcp,dcq->dpq', x, x, optimize=True),
            'xy_day': np.einsum('dcp,dc->dp', x, y, optimize=True), 'yy_day': (y * y).sum(axis=1)}


def masked_cross(stats, weights):
    xx = np.tensordot(weights, stats['xx_day'], axes=1)
    xy = weights @ stats['xy_day']
    yy = float(weights @ stats['yy_day'])
    for month in np.unique(stats['months']):
        take = stats['months'] == month
        wt = weights[take]
        n = wt @ stats['z'][take]
        active = n > 0
        if not active.any():
            continue
        sx = np.tensordot(wt, stats['x'][take], axes=1)[active]
        sy = (wt @ stats['y'][take])[active]
        n = n[active]
        xx -= sx.T @ (sx / n[:, None])
        xy -= sx.T @ (sy / n)
        yy -= float((sy * sy / n).sum())
    return (xx + xx.T) / 2, xy, yy


def solve(xx, xy, yy):
    sd = np.sqrt(np.maximum(np.diag(xx), 0))
    if (sd <= 1e-10).any() or yy <= 1e-10:
        return {'valid': False, 'reason': 'constant within-cell term/response'}
    norm = xx / np.outer(sd, sd)
    eigen = np.linalg.eigvalsh(norm)
    cond = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else None
    if cond is None or cond >= 1e8:
        return {'valid': False, 'reason': 'rank/condition', 'condition': cond}
    beta = np.linalg.solve(norm, xy / sd) / sd
    sse = float(yy - beta @ xy)
    if sse < -1e-7 * max(yy, 1):
        raise ValueError('negative residual sum of squares')
    return {'valid': True, 'beta_pp': (beta * 100).tolist(), 'within_r2': 1 - max(sse, 0) / yy,
            'condition': cond, 'normal_error': float(np.linalg.norm(xx @ beta - xy) / max(np.linalg.norm(xy), 1))}


def block_weights(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    index = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
    return np.bincount(index, minlength=n)


def analyze():
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Completed results exist; no automatic rerun')
    inspection = json.loads((OUT / 'inspection.json').read_text(encoding='utf-8'))
    before = {f: digest(SOURCE / f) for f in FILES}
    if before != inspection['source_hashes']:
        raise ValueError('source changed after availability inspection')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'preregistered_report.md').write_text(REPORT.read_text(encoding='utf-8'), encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': before, 'code_hash': digest(__file__),
                               'report_hash': digest(REPORT), 'seed': SEED, 'bootstrap_reps': BOOT_REPS})
    sites, gen, dates, cf, complete, weather, station_weather, stations = load()
    if sites.site_id.tolist() != inspection['site_ids'] or sites.cohort.tolist() != inspection['labels']:
        raise ValueError('cohort changed after inspection')
    take = dates.year == 2024
    dates, cf, complete = dates[take], cf[take], complete[take]
    weather, station_weather = weather[take], station_weather[take]
    months = dates.month.to_numpy()
    eligible = monthly_eligibility(complete, months)
    profiles = profile_comparisons(sites, dates, cf, complete, eligible)
    log(f'calendar and same-B comparisons: {len(profiles)}')
    w, support, passed, counts, bench, support_records = matched_support(
        sites, dates, cf, complete, weather, station_weather, stations, eligible)
    log('joint weather support complete')
    sites.to_csv(OUT / 'sites_local.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{k: v for k, v in r.items() if not k.endswith('_shape')} for r in profiles]).to_csv(
        OUT / 'matched_profiles.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(support_records).to_csv(OUT / 'weather_support.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(OUT / 'panel.npz', cf=cf, complete=complete, eligible=eligible, weather=w,
        station_weather=station_weather, support=support, passed=passed, reference_count=counts, benchmark=bench,
        site_ids=sites.site_id.to_numpy(dtype=str), cohorts=sites.cohort.to_numpy(dtype=str),
        stations=sites.wb_station.to_numpy(dtype=str), dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str))
    summary = {'source_hashes': before, 'code_hash': digest(__file__), 'python': sys.version,
        'executable': sys.executable, 'seed': SEED, 'terms': TERMS, 'inspection': inspection['cohorts'],
        'profiles': profiles, 'support': support_records, 'regression': {}, 'contrasts': {}}
    daylight_gen = gen.loc[gen.start.dt.year.eq(2024) & gen.start.dt.hour.isin(HOURS)]
    summary['audit'] = {'grid_cells_2024': int(cf.size), 'present_source_rows': len(daylight_gen),
        'missing_source_keys': int(cf.size - len(daylight_gen)),
        'invalid_present_energy': int((~np.isfinite(daylight_gen.energy) | daylight_gen.energy.lt(0)).sum()),
        'valid_zero_cells': int((cf == 0).sum()), 'above_one_cells': int((cf > 1).sum())}
    second = months >= 7
    fixed = np.ones(len(sites), dtype=bool)
    for month in range(7, 13):
        fixed &= complete[months == month].sum(axis=0) >= 14
    summary['fixed_h2_sites'] = {c: int((fixed & sites.cohort.eq(c).to_numpy()).sum()) for c in COHORTS}
    hmonths = months[second]
    cases = {}
    xall = linear_design(w[second])
    for cohort in COHORTS:
        pick = np.flatnonzero(fixed & sites.cohort.eq(cohort).to_numpy())
        mask = passed[second][:, pick]
        present_sites = mask.any(axis=(0, 2))
        distinct = sites.wb_station.iloc[pick[present_sites]].nunique()
        ndates = int(mask.any(axis=(1, 2)).sum())
        nmonths = len(np.unique(hmonths[mask.any(axis=(1, 2))]))
        audit = {'fixed_sites': len(pick), 'supported_sites': int(present_sites.sum()),
                 'stations': int(distinct), 'dates': ndates, 'rows': int(mask.sum()), 'months': nmonths}
        if present_sites.sum() < 10 or distinct < 5 or ndates < 60 or mask.sum() < 1000 or nmonths < 6:
            summary['regression'][cohort] = {'audit': audit, 'valid': False, 'reason': 'support threshold'}
            continue
        stats = masked_stats(xall[:, pick], cf[second][:, pick, 2:10], mask, hmonths)
        point = solve(*masked_cross(stats, np.ones(second.sum())))
        summary['regression'][cohort] = {'audit': audit, **point}
        if point['valid']:
            cases[cohort] = stats
        np.savez_compressed(OUT / f'fit_{cohort}.npz', x=xall[:, pick], y=cf[second][:, pick, 2:10],
                            mask=mask, months=hmonths, site_indices=pick)
    rng = np.random.default_rng(SEED)
    draws, fitted = [], {c: [] for c in cases}
    for block in [7, 14]:
        for rep in range(BOOT_REPS):
            weights = block_weights(int(second.sum()), block, rng)
            draws.append(weights)
            for cohort, stats in cases.items():
                result = solve(*masked_cross(stats, weights))
                fitted[cohort].append(result['beta_pp'] if result['valid'] else [np.nan] * len(TERMS))
            if (rep + 1) % 50 == 0:
                log(f'joint bootstrap block={block}: {rep+1}/{BOOT_REPS}')
    arrays = {f'beta_{c}': np.asarray(v) for c, v in fitted.items()}
    arrays['weights'] = np.asarray(draws)
    np.savez_compressed(OUT / 'bootstrap.npz', **arrays)
    for cohort, values in fitted.items():
        values = np.asarray(values)
        summary['regression'][cohort]['bootstrap'] = {}
        for k, block in enumerate([7, 14]):
            a = values[k * BOOT_REPS:(k + 1) * BOOT_REPS]
            valid = np.isfinite(a).all(axis=1)
            summary['regression'][cohort]['bootstrap'][str(block)] = {
                'success': int(valid.sum()), 'intervals_pp': np.quantile(a[valid], [.025, .975], axis=0).T.tolist()
                if valid.sum() >= .9 * BOOT_REPS else None}
    if 'L80' in fitted:
        for cohort in ['M68', 'S105']:
            if cohort not in fitted:
                continue
            values = np.asarray(fitted[cohort]) - np.asarray(fitted['L80'])
            diff = np.array(summary['regression'][cohort]['beta_pp']) - summary['regression']['L80']['beta_pp']
            record = {'difference_pp': diff.tolist(), 'bootstrap': {}}
            for k, block in enumerate([7, 14]):
                a = values[k * BOOT_REPS:(k + 1) * BOOT_REPS]
                valid = np.isfinite(a).all(axis=1)
                record['bootstrap'][str(block)] = {'success': int(valid.sum()),
                    'intervals_pp': np.quantile(a[valid], [.025, .975], axis=0).T.tolist()
                    if valid.sum() >= .9 * BOOT_REPS else None}
            summary['contrasts'][cohort] = record
    assert before == {f: digest(SOURCE / f) for f in FILES}
    assert digest(__file__) == summary['code_hash']
    dump(OUT / 'summary.json', summary)
    log(json.dumps({'audit': summary['audit'], 'fixed_h2_sites': summary['fixed_h2_sites'],
                    'regression': summary['regression'], 'contrasts': summary['contrasts']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['inspect', 'analyze'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        inspect() if args.mode == 'inspect' else analyze()
