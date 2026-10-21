"""DA-001: supplied weather latent structure, without generation prediction."""
from __future__ import annotations

import importlib
import argparse
import hashlib
import json
import platform
import sys
import sysconfig
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from statsmodels.multivariate.factor import Factor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA' / '01_participant_dataset'
OUT = ROOT / 'data' / 'analysis' / '2026-10-21_weather_latent'
REPORT = ROOT / 'reports' / '2026-10-21_06_기상변수_잠재구조_분석.md'
FILES = ['sites.csv', 'train/generation.parquet'] + [
    f'train/{kind}_{p}.parquet' for p in ['wa', 'wb'] for kind in ['observed', 'forecast']]
SEED = 20261021
NULL_REPS = 199
BOOT_REPS = 100


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def log(message):
    print(datetime.now(ZoneInfo('Asia/Seoul')).strftime('%H:%M:%S') + ' ' + message, flush=True)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def correlation_from_stats(total, cross, n):
    mean = total / n
    cov = cross / n - np.outer(mean, mean)
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    if not np.isfinite(sd).all() or (sd < 1e-10).any():
        raise ValueError('constant or nonfinite variable')
    r = cov / np.outer(sd, sd)
    return (r + r.T) / 2


def diagnostics(r):
    ev = np.linalg.eigvalsh(r)
    condition = float(ev[-1] / ev[0]) if ev[0] > 0 else None
    good = bool(ev[0] > 1e-8 and condition < 1e8 and np.isfinite(r).all())
    out = {'min_eigenvalue': float(ev[0]), 'condition': condition, 'eligible': good}
    if not good:
        return out
    inv = np.linalg.inv(r)
    partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    mask = ~np.eye(len(r), dtype=bool)
    a = np.where(mask, r * r, 0).sum(axis=0)
    b = np.where(mask, partial * partial, 0).sum(axis=0)
    out['kmo'] = float(a.sum() / (a + b).sum())
    out['kmo_by_variable'] = (a / (a + b)).tolist()
    out['eigenvalues'] = ev[::-1].tolist()
    out['pca_2'] = float(ev[-2:].sum() / len(r))
    out['pca_3'] = float(ev[-3:].sum() / len(r))
    return out


def max_factors(p):
    return max([k for k in range(1, min(3, p - 1) + 1)
                if ((p - k) ** 2 - p - k) / 2 > 0], default=0)


def fit_factor(r, k):
    if k == 0:
        return {'valid': False, 'reason': 'no factor selected'}
    diag = diagnostics(r)
    if not diag['eligible']:
        return {'valid': False, 'reason': 'singular, ill-conditioned or constant matrix'}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            fit = Factor(corr=r, n_factor=k, method='pa', smc=True).fit(maxiter=500, tol=1e-8)
            original = np.asarray(fit.loadings).copy()
            # statsmodels PA has no convergence flag. Check its next communality update.
            reduced = r.copy()
            np.fill_diagonal(reduced, fit.communality)
            val, vec = np.linalg.eigh(reduced)
            next_load = vec[:, -k:] * np.sqrt(np.maximum(val[-k:], 0))
            fixed_error = float(np.linalg.norm((next_load ** 2).sum(axis=1) - fit.communality))
            fit.rotate('varimax')
            load = np.asarray(fit.loadings).copy()
            rotation_error = float(np.max(np.abs(load @ load.T - original @ original.T)))
            # Give each factor a deterministic order and sign; no scientific meaning to order/sign.
            load = load[:, np.argsort(-(load ** 2).sum(axis=0))]
            for j in range(load.shape[1]):
                if load[np.argmax(np.abs(load[:, j])), j] < 0:
                    load[:, j] *= -1
            h = (load ** 2).sum(axis=1)
            residual = r - load @ load.T
            off = residual[~np.eye(len(r), dtype=bool)]
            proper = bool(load.shape[1] == k and np.isfinite(load).all()
                          and h.min() >= -1e-8 and h.max() <= 1 + 1e-8
                          and fixed_error < 1e-6 and rotation_error < 1e-7)
            return {'valid': proper, 'reason': None if proper else 'Heywood, dimension, convergence or rotation check',
                    'loadings': load.tolist(), 'communality': h.tolist(), 'uniqueness': (1 - h).tolist(),
                    'rms_residual': float(np.sqrt(np.mean(off ** 2))), 'max_residual': float(np.max(np.abs(off))),
                    'factor_ss': (load ** 2).sum(axis=0).tolist(),
                    'support_per_factor': (np.abs(load) >= .4).sum(axis=0).tolist(),
                    'fixed_point_error': fixed_error, 'rotation_error': rotation_error,
                    'warnings': [str(w.message) for w in recorded]}
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return {'valid': False, 'reason': str(exc), 'warnings': [str(w.message) for w in recorded]}


def align_loadings(reference, candidate):
    a = np.asarray(reference)
    b = np.asarray(candidate)
    cos = a.T @ b / np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    row, col = linear_sum_assignment(-np.abs(cos))
    order = col[np.argsort(row)]
    signs = np.sign(cos[np.arange(a.shape[1]), order])
    signs[signs == 0] = 1
    aligned = b[:, order] * signs
    congruence = np.sum(a * aligned, axis=0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(aligned, axis=0))
    return aligned, congruence


def circular_cross_products(x):
    """Sum x[d, cells, i] * x[d+lag, cells, j] for every lag/i/j."""
    n = len(x)
    flat = x.reshape(n, -1, x.shape[-1])
    f = np.fft.rfft(flat, axis=0)
    spectral = np.einsum('fci,fcj->fij', f.conj(), f, optimize=True)
    return np.fft.irfft(spectral, n=n, axis=0)


def null_eigenvalues(z, months, rng, reps=NULL_REPS):
    p = z.shape[-1]
    sums = np.zeros((reps, p, p))
    for month in range(1, 13):
        part = z[months == month]
        cross = circular_cross_products(part)
        offsets = rng.integers(0, len(part), size=(reps, p))
        lags = (offsets[:, :, None] - offsets[:, None, :]) % len(part)
        sums += cross[lags, np.arange(p)[None, :, None], np.arange(p)[None, None, :]]
    corr = sums / np.prod(z.shape[:-1])
    assert np.allclose(corr, corr.transpose(0, 2, 1), atol=1e-10)
    assert np.allclose(np.diagonal(corr, axis1=1, axis2=2), 1, atol=1e-9)
    assert np.linalg.eigvalsh(corr).min() > -1e-8
    return np.linalg.eigvalsh(corr)[:, ::-1]


def daily_stats(z):
    flat = z.reshape(len(z), -1, z.shape[-1])
    sums = flat.sum(axis=1)
    crosses = np.einsum('dci,dcj->dij', flat, flat, optimize=True)
    return sums, crosses, flat.shape[1]


def block_indices(n, rng, block=7):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)) % n).ravel()[:n]


def bootstrap(sums, crosses, ncell, reference, rng):
    n = len(sums)
    congruences, loadings, reasons = [], [], {}
    for _ in range(BOOT_REPS):
        idx = block_indices(n, rng)
        try:
            r = correlation_from_stats(sums[idx].sum(axis=0), crosses[idx].sum(axis=0), n * ncell)
            fit = fit_factor(r, len(reference[0]))
        except ValueError as exc:
            fit = {'valid': False, 'reason': str(exc)}
        if fit['valid']:
            aligned, congruence = align_loadings(reference, fit['loadings'])
            loadings.append(aligned)
            congruences.append(congruence)
        else:
            reason = fit['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
    out = {'attempts': BOOT_REPS, 'success': len(loadings), 'failure_reasons': reasons}
    if loadings:
        out.update(min_congruence_q05=float(np.quantile(np.min(congruences, axis=1), .05)),
                   congruence_quantiles=np.quantile(congruences, [.025, .5, .975], axis=0).tolist(),
                   loading_intervals=np.quantile(loadings, [.025, .5, .975], axis=0).tolist())
    return out


def core_columns(provider, kind):
    if provider == 'wa':
        return ['wa_temp', 'wa_humidity', 'wa_cloud', 'wa_ground_press', 'wa_wind_speed', 'wa_rain', 'wa_snow']
    rain = 'wb_precip_1h' if kind == 'observed' else 'wb_rain'
    return ['wb_temp', 'wb_rel_hum', 'wb_cloud', 'wb_wind_speed', 'wb_uv_idx', rain, 'wb_vis', 'wb_ceiling']


def valid_values(s, col):
    ok = np.isfinite(s)
    if col.endswith(('humidity', 'rel_hum', 'cloud')):
        ok &= s.between(0, 100)
    elif col.endswith('wind_speed'):
        ok &= s.between(0, 50 if col.startswith('wa') else 200)
    elif col.endswith('press'):
        ok &= s.between(900, 1100)
    elif col == 'wb_uv_idx':
        ok &= s.between(0, 15)
    elif col == 'wb_vis':
        ok &= s.between(0, 50)
    elif col.endswith(('rain', 'snow', 'ceiling', 'precip_1h')):
        ok &= s >= 0
    return ok


def balanced_cube(data, provider, kind, stations, year):
    station = provider + '_station'
    columns = core_columns(provider, kind)
    start = data.ts - pd.Timedelta(hours=1)
    d = data.loc[start.dt.year.eq(year) & data[station].isin(stations), [station, 'ts'] + columns].copy()
    raw_n = len(d)
    onhour = d.ts.eq(d.ts.dt.floor('h'))
    nonhour = int((~onhour).sum())
    d = d.loc[onhour]
    duplicates = int(d.duplicated([station, 'ts'], keep=False).sum())
    conflicts = 0
    if duplicates:
        if kind != 'observed':
            raise ValueError('duplicate forecast key')
        grouped = d.groupby([station, 'ts'], observed=True)[columns]
        distinct = grouped.nunique(dropna=False)
        conflicts = int((distinct > 1).sum().sum())
        d = grouped.first().mask(distinct > 1).reset_index()
    assert not d.duplicated([station, 'ts']).any()
    invalid = {}
    for c in columns:
        good = valid_values(d[c], c)
        invalid[c] = int((~good).sum())
        d.loc[~good, c] = np.nan
        if c.endswith(('rain', 'snow', 'ceiling', 'precip_1h')):
            d[c] = np.log1p(d[c])
    start = d.ts - pd.Timedelta(hours=1)
    d['date'] = start.dt.floor('D')
    d['hour'] = start.dt.hour
    d['bad'] = ~np.isfinite(d[columns]).all(axis=1)
    daily = d.groupby('date', observed=True).agg(rows=('ts', 'size'), bad=('bad', 'sum'))
    good_dates = daily.index[daily.rows.eq(len(stations) * 24) & daily.bad.eq(0)]
    good_dates = good_dates.sort_values()
    monthly = pd.Series(good_dates.month).value_counts().reindex(range(1, 13), fill_value=0)
    if monthly.min() < 14:
        raise ValueError(f'{provider}/{kind}/{year}: fewer than 14 balanced dates in a month')
    keep = d.loc[d.date.isin(good_dates)].sort_values(['date', station, 'hour'])
    x = keep[columns].to_numpy().reshape(len(good_dates), len(stations), 24, len(columns))
    assert np.isfinite(x).all() and len(keep) == len(good_dates) * len(stations) * 24
    assert keep.groupby(['date', station], observed=True).hour.nunique().eq(24).all()
    audit = {'year': year, 'raw_rows': raw_n, 'nonhour_rows': nonhour, 'duplicate_key_rows': duplicates,
             'conflicting_cells': conflicts, 'invalid_by_column': invalid, 'available_dates': len(daily),
             'balanced_dates': len(good_dates), 'balanced_rows': len(keep), 'stations': len(stations),
             'excluded_dates': len(daily) - len(good_dates), 'dates_per_month': monthly.tolist()}
    return x, np.asarray(good_dates.month), columns, audit


def make_view(x, months, view):
    y = x.copy()
    if view == 'within_day':
        y = y[:, :, 9:17, :]
        for m in range(1, 13):
            y[months == m] -= y[months == m].mean(axis=0, keepdims=True)
    mean = y.mean(axis=(0, 1, 2), keepdims=True)
    sd = y.std(axis=(0, 1, 2), keepdims=True)
    if (sd < 1e-10).any():
        raise ValueError('constant annual variable')
    return (y - mean) / sd


def analyze_case(x24, m24, x23, m23, columns, label, rng, sensitivity=False):
    results = {}
    for view in ['all', 'within_day']:
        log(label + ' / ' + view + (' / sensitivity' if sensitivity else ''))
        z = make_view(x24, m24, view)
        sums, crosses, ncell = daily_stats(z)
        n = len(z) * ncell
        r = correlation_from_stats(sums.sum(axis=0), crosses.sum(axis=0), n)
        diag = diagnostics(r)
        case = {'columns': columns, 'diagnostics': diag, 'n': n, 'correlation': r.tolist()}
        null = null_eigenvalues(z, m24, rng)
        threshold = np.quantile(null, .95, axis=0)
        ev = np.linalg.eigvalsh(r)[::-1]
        exceed = ev > threshold
        selected_raw = int(exceed.sum())
        k = min(selected_raw, max_factors(len(columns)))
        case.update(null_threshold=threshold.tolist(), null_exceeds=exceed.tolist(), selected_raw=selected_raw,
                    selected=k, selected_cap=max_factors(len(columns)))
        fit = fit_factor(r, k)
        case['factor'] = fit
        z23 = make_view(x23, m23, view)
        s23, c23, ncell23 = daily_stats(z23)
        r23 = correlation_from_stats(s23.sum(axis=0), c23.sum(axis=0), len(z23) * ncell23)
        fit23 = fit_factor(r23, k)
        case['replication_2023'] = {'diagnostics': diagnostics(r23), 'factor': fit23}
        if fit['valid'] and fit23['valid']:
            aligned, cong = align_loadings(fit['loadings'], fit23['loadings'])
            case['replication_2023'].update(aligned_loadings=aligned.tolist(), congruence=cong.tolist())
        if not sensitivity:
            for q in range(1, 5):
                take = ((m24 - 1) // 3 + 1) == q
                try:
                    rq = correlation_from_stats(sums[take].sum(axis=0), crosses[take].sum(axis=0), int(take.sum()) * ncell)
                    fq = fit_factor(rq, k)
                except ValueError as exc:
                    fq = {'valid': False, 'reason': str(exc)}
                if fit['valid'] and fq['valid']:
                    _, congruence = align_loadings(fit['loadings'], fq['loadings'])
                    fq['congruence'] = congruence.tolist()
                case[f'quarter_{q}'] = fq
            if fit['valid']:
                case['bootstrap'] = bootstrap(sums, crosses, ncell, fit['loadings'], rng)
            # Sufficient statistics only, stored locally. Includes no individual weather rows.
            np.savez_compressed(OUT / f'{label}_{view}_stats.npz', sums=sums, crosses=crosses,
                                ncell=ncell, months=m24, sums23=s23, crosses23=c23, ncell23=ncell23)
        warnings_list = []
        if diag.get('kmo', 0) < .6:
            warnings_list.append('KMO below 0.6')
        if selected_raw > k:
            warnings_list.append('factor count capped by prespecified limit')
        if not fit['valid']:
            warnings_list.append('invalid or unselected primary factor solution')
        if fit.get('support_per_factor') and min(fit['support_per_factor']) < 2:
            warnings_list.append('factor with fewer than two loadings >= 0.4')
        boot = case.get('bootstrap', {})
        if boot and boot['success'] < .9 * BOOT_REPS:
            warnings_list.append('bootstrap success below 90%')
        cong = case['replication_2023'].get('congruence', [])
        case['stable_summary_candidate'] = bool(fit['valid'] and cong and min(cong) >= .85
            and boot.get('min_congruence_q05', 0) >= .85 and fit['rms_residual'] <= .1)
        case['cautions'] = warnings_list
        results[view] = case
    return results


def make_figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eda_plot_utils import save_plot, write_gallery
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 9})
    gallery = []
    for label, data in summary['primary'].items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for col, view in enumerate(['all', 'within_day']):
            case = data[view]
            ev = case['diagnostics']['eigenvalues']
            ax = axes[0, col]
            ax.plot(np.arange(1, len(ev) + 1), ev, 'o-', label='관측')
            ax.plot(np.arange(1, len(ev) + 1), case['null_threshold'], 's--', label='월내 순환이동 95%')
            ax.set(xlabel='차원 순서', ylabel='상관행렬 고유값', title=view + f" / 선택 {case['selected']}개")
            ax.legend()
            ax = axes[1, col]
            fit = case['factor']
            if 'loadings' in fit:
                a = np.asarray(fit['loadings'])
                im = ax.imshow(a, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax.set_yticks(range(len(case['columns'])), case['columns'])
                ax.set_xticks(range(a.shape[1]), [f'F{i+1}' for i in range(a.shape[1])])
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        ax.text(j, i, f'{a[i,j]:.2f}', ha='center', va='center', fontsize=9)
                ax.set_title('회전 적재량' + ('' if fit['valid'] else ' / 부적절해 또는 수렴 미확인'))
                fig.colorbar(im, ax=ax, shrink=.7)
            else:
                ax.text(.1, .5, fit['reason'], transform=ax.transAxes)
        save_plot(fig, OUT, gallery, label, label + ' 잠재 구조',
                  'all: 전체 수준. within_day: 지점/월/시간 평균을 뺀 낮 편차. 요인 부호와 순서는 임의적이다.')
    fig, ax = plt.subplots(figsize=(14, 6))
    names, repeat, block = [], [], []
    for label, data in summary['primary'].items():
        for view, case in data.items():
            names.append(label + '/' + view)
            repeat.append(min(case['replication_2023'].get('congruence', [np.nan])))
            block.append(case.get('bootstrap', {}).get('min_congruence_q05', np.nan))
    xx = np.arange(len(names))
    ax.bar(xx - .18, repeat, width=.36, label='2023 재현 최저 일치도')
    ax.bar(xx + .18, block, width=.36, label='7일 블록 최저 일치도 5백분위')
    ax.axhline(.85, color='black', linestyle='--', label='사전 기술 기준 0.85')
    ax.set(ylim=(0, 1.05), ylabel='Tucker 일치도')
    ax.set_xticks(xx, names, rotation=50, ha='right')
    ax.legend()
    save_plot(fig, OUT, gallery, 'stability', '기간과 날짜 블록의 구조 안정성',
              '값이 없는 막대는 기준 적합 또는 반복 적합 실패다. 예측 성능 검증이 아니다.')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, view in zip(axes, ['all', 'within_day']):
        labels = list(summary['sensitivity'])
        full = [summary['primary'][k][view]['diagnostics']['kmo'] for k in labels]
        reduced = [summary['sensitivity'][k][view]['diagnostics']['kmo'] for k in labels]
        xx = np.arange(len(labels))
        ax.bar(xx - .18, full, width=.36, label='시정/운고 포함 8변수')
        ax.bar(xx + .18, reduced, width=.36, label='같은 행의 6변수')
        ax.axhline(.6, color='black', linestyle='--')
        ax.set_xticks(xx, labels, rotation=25, ha='right')
        ax.set(title=view, ylim=(0, 1), ylabel='KMO')
        ax.legend(fontsize=8)
    save_plot(fig, OUT, gallery, 'sensitivity', 'B 시정/운고 변수군의 민감도',
              '같은 행을 유지하고 두 열만 제외했다. 집중값 행 제거 효과를 측정한 것이 아니다.')
    write_gallery(OUT, gallery, 'DA-001 기상 변수 잠재 구조')


def run_analysis():
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Completed results exist; do not automatically refit.')
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {rel: digest(SOURCE / rel) for rel in FILES}
    (OUT / 'preregistered_report.md').write_text(REPORT.read_text(encoding='utf-8'), encoding='utf-8')
    dump(OUT / 'precheck.json', {'source_hashes': source_hashes, 'code_hash': digest(__file__),
                               'report_hash': digest(REPORT), 'null_reps': NULL_REPS, 'bootstrap_reps': BOOT_REPS,
                               'seed': SEED, 'python': sys.version, 'executable': sys.executable})
    inspection = json.loads((OUT / 'inspection.json').read_text(encoding='utf-8'))
    summary = {'primary': {}, 'sensitivity': {}, 'audit': {}, 'source_hashes': source_hashes,
               'runtime': inspection['packages'], 'seed': SEED, 'null_reps': NULL_REPS, 'bootstrap_reps': BOOT_REPS}
    rng = np.random.default_rng(SEED)
    with threadpool_limits(limits=1):
        for provider in ['wa', 'wb']:
            for kind in ['observed', 'forecast']:
                rel = f'train/{kind}_{provider}.parquet'
                stations = inspection['files'][rel]['stations']
                data = pd.read_parquet(SOURCE / rel)
                start = data.ts - pd.Timedelta(hours=1)
                data = data.loc[start.dt.year.isin([2023, 2024]) & data[provider + '_station'].isin(stations)]
                for round_ in ([None] if kind == 'observed' else [10, 17]):
                    label = provider + ('_observed' if round_ is None else f'_{round_}')
                    part = data if round_ is None else data.loc[data['round'].eq(round_)]
                    x24, m24, columns, a24 = balanced_cube(part, provider, kind, stations, 2024)
                    x23, m23, _, a23 = balanced_cube(part, provider, kind, stations, 2023)
                    summary['audit'][label] = [a24, a23]
                    summary['primary'][label] = analyze_case(x24, m24, x23, m23, columns, label, rng)
                    if provider == 'wb':
                        summary['sensitivity'][label] = analyze_case(x24[..., :6], m24, x23[..., :6], m23,
                                                                    columns[:6], label, rng, sensitivity=True)
                    dump(OUT / 'progress.json', summary)
    assert source_hashes == {rel: digest(SOURCE / rel) for rel in FILES}
    summary['finished_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    summary['code_hash'] = digest(__file__)
    dump(OUT / 'summary.json', summary)
    make_figures(summary)
    log('Analysis completed: ' + str(OUT))


def inspect_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'pyarrow', 'PIL', 'threadpoolctl']:
        try:
            mod = importlib.import_module(name)
            packages[name] = getattr(mod, '__version__', 'available')
        except ImportError as exc:
            packages[name] = {'error': str(exc)}
    import pandas as pd
    import pyarrow.parquet as pq
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig')
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet', columns=['site_id', 'ts'])
    start = gen.ts - pd.Timedelta(hours=1)
    counts = gen.loc[start.dt.year.eq(2024)].groupby('site_id').size()
    ids = counts.index[counts.eq(8784)]
    cohort = sites.loc[sites.site_id.isin(ids)]
    info = {'python': platform.python_version(), 'executable': sys.executable,
            'packages': packages, 'cohort_sites': len(cohort), 'files': {}}
    for provider in ['wa', 'wb']:
        station = provider + '_station'
        stations = sorted(cohort[station].unique().tolist())
        for kind in ['observed', 'forecast']:
            rel = f'train/{kind}_{provider}.parquet'
            schema = pq.read_schema(SOURCE / rel)
            columns = [station, 'ts'] + (['round'] if kind == 'forecast' else [])
            d = pd.read_parquet(SOURCE / rel, columns=columns)
            s = d.ts - pd.Timedelta(hours=1)
            keep = d[station].isin(stations) & s.dt.year.isin([2023, 2024])
            d = d.loc[keep].copy()
            d['year'] = (d.ts - pd.Timedelta(hours=1)).dt.year
            keys = ['year'] + (['round'] if kind == 'forecast' else [])
            rows = d.groupby(keys, observed=True).agg(rows=('ts', 'size'), stations=(station, 'nunique'))
            info['files'][rel] = {'schema': str(schema), 'stations': stations,
                                  'availability': rows.reset_index().to_dict('records')}
    (OUT / 'inspection.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    assert sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var('Py_GIL_DISABLED')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['run', 'figures'])
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        inspect_inputs()
    elif args.mode == 'run':
        run_analysis()
    elif args.mode == 'figures':
        make_figures(json.loads((OUT / 'summary.json').read_text(encoding='utf-8')))
    else:
        parser.error('choose --inspect, run or figures')
