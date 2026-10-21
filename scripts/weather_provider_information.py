"""DA-005 canonical relationships and unexplained linear variance in A/B forecasts."""
from __future__ import annotations

import argparse
import json
import sys
import sysconfig
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from threadpoolctl import threadpool_limits

from weather_latent_structure import digest, dump, log

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'OIBC_2026_DATA/01_participant_dataset'
OUT = ROOT / 'data/analysis/2026-10-21_provider_information'
REPORT = ROOT / 'reports/2026-10-21_10_AB예보의_공통정보와_잔여변동_분석.md'
FILES = ['sites.csv', 'train/generation.parquet', 'train/forecast_wa.parquet', 'train/forecast_wb.parquet']
SEED, BOOT, NULL = 2026102105, 200, 199
A_NAMES = ['temp', 'humidity', 'cloud', 'wind', 'wet', 'lograin']
B_NAMES = A_NAMES + ['uv']


def mapping():
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet', columns=['site_id', 'ts'])
    start = gen.ts - pd.Timedelta(hours=1)
    counts = gen.loc[start.dt.year.eq(2024)].groupby('site_id').size()
    ids = sorted(counts.index[counts.eq(8784)])
    assert len(ids) == 148 and not gen.duplicated(['site_id', 'ts']).any()
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig')
    pairs = sites.loc[sites.site_id.isin(ids), ['wa_station', 'wb_station']].drop_duplicates().sort_values(['wa_station', 'wb_station']).reset_index(drop=True)
    a = sorted(pairs.wa_station.unique())
    b = sorted(pairs.wb_station.unique())
    graph = np.zeros((len(a), len(b)), dtype=int)
    for row in pairs.itertuples():
        graph[a.index(row.wa_station), b.index(row.wb_station)] = 1
    match = maximum_bipartite_matching(csr_matrix(graph), perm_type='column')
    chosen = np.array([match[a.index(row.wa_station)] == b.index(row.wb_station) for row in pairs.itertuples()])
    assert pairs.loc[chosen].wa_station.is_unique and pairs.loc[chosen].wb_station.is_unique
    return pairs, chosen


def load_panel(pairs, round_number, year):
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', tz='Asia/Seoul')
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=h+1) for d in dates for h in range(9, 17)])
    ids = np.arange(len(pairs))
    grid = pd.MultiIndex.from_product([times, ids], names=['ts', 'pair_id']).to_frame(index=False)
    grid = grid.merge(pairs.reset_index(names='pair_id'), on='pair_id', how='left', validate='many_to_one')
    audits = {}
    columns = {'wa': ['wa_temp', 'wa_humidity', 'wa_cloud', 'wa_wind_speed', 'wa_rain'],
               'wb': ['wb_temp', 'wb_rel_hum', 'wb_cloud', 'wb_wind_speed', 'wb_rain', 'wb_uv_idx']}
    for provider in ['wa', 'wb']:
        key = provider + '_station'
        cols = columns[provider]
        w = pd.read_parquet(SOURCE / f'train/forecast_{provider}.parquet', columns=[key, 'ts', 'round']+cols)
        start = w.ts - pd.Timedelta(hours=1)
        w = w.loc[w['round'].eq(round_number) & start.dt.year.eq(year) & start.dt.hour.between(9,16) & w[key].isin(pairs[key])].copy()
        nonhour = int(w.ts.ne(w.ts.dt.floor('h')).sum())
        w = w.loc[w.ts.eq(w.ts.dt.floor('h'))]
        duplicated = int(w.duplicated([key, 'ts'], keep=False).sum())
        if duplicated:
            raise ValueError('duplicate forecast station/time key')
        invalid = {}
        for c in cols:
            good = np.isfinite(w[c])
            if c.endswith(('humidity', 'rel_hum', 'cloud')):
                good &= w[c].between(0,100)
            elif c.endswith('wind_speed'):
                good &= w[c].between(0,50 if provider == 'wa' else 200)
            elif c.endswith('rain'):
                good &= w[c].ge(0)
            elif c.endswith('uv_idx'):
                good &= w[c].between(0,15)
            invalid[c] = int((~good).sum())
            w.loc[~good, c] = np.nan
        audits[provider] = {'station_rows': len(w), 'nonhour_rows': nonhour, 'duplicate_rows': duplicated, 'invalid_cells': invalid}
        grid = grid.merge(w.drop(columns='round'), on=[key,'ts'], how='left', validate='many_to_one')
    grid = grid.sort_values(['ts','pair_id'])
    raw = grid[columns['wa']+columns['wb']].to_numpy(copy=True).reshape(len(dates),8,len(pairs),11).transpose(0,2,1,3)
    good_dates = np.isfinite(raw).all(axis=(1,2,3))
    dates = dates[good_dates]
    raw = raw[good_dates]
    nmonth = np.bincount(dates.month, minlength=13)[1:]
    if nmonth.min() < 14:
        raise ValueError('fewer than 14 balanced dates in a month')
    a, b = raw[...,:5], raw[...,5:]
    xa = np.stack([a[...,0],a[...,1],a[...,2],a[...,3],(a[...,4]>.1).astype(float),np.log1p(a[...,4])],axis=-1)
    xb = np.stack([b[...,0],b[...,1],b[...,2],b[...,3]/3.6,(b[...,4]>.1).astype(float),np.log1p(b[...,4]),b[...,5]],axis=-1)
    z = np.concatenate([xa,xb],axis=-1)
    audit = {'round': round_number,'year':year,'pairs':len(pairs),'dates':len(dates),
             'excluded_dates': int((~good_dates).sum()),'rows':int(np.prod(z.shape[:-1])),
             'dates_per_month':nmonth.tolist(),'providers':audits}
    return z, dates, audit


def inspect():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, chosen = mapping()
    info = {'pairs':len(pairs),'a_stations':pairs.wa_station.nunique(),'b_stations':pairs.wb_station.nunique(),
            'matching_pairs':int(chosen.sum()),'panels':{}}
    for round_number in [10,17]:
        for year in [2024,2023]:
            z,dates,audit = load_panel(pairs, round_number,year)
            info['panels'][f'{round_number}_{year}'] = audit
    dump(OUT/'inspection.json',info)
    print(json.dumps(info,ensure_ascii=False,indent=2))


def centered(z, months, weights=None):
    w = np.ones(len(z)) if weights is None else weights
    out = z.copy()
    for month in np.unique(months):
        take = months == month
        total = w[take].sum()
        out[take] = z[take] - np.tensordot(w[take], z[take], axes=(0,0))/total if total > 0 else 0
    return out


def covariance(z, months, weights=None, within=True, day_cross=None):
    x = z.reshape(len(z),-1,z.shape[-1])
    w = np.ones(len(z)) if weights is None else weights
    n = w.sum()*x.shape[1]
    if day_cross is None:
        day_cross = np.einsum('dci,dcj->dij',x,x,optimize=True)
    gram = np.tensordot(w,day_cross,axes=(0,0))
    if within:
        for month in np.unique(months):
            take = months == month
            total = w[take].sum()
            if total > 0:
                sx = np.tensordot(w[take],x[take],axes=(0,0))
                gram -= sx.T@sx/total
    else:
        sx = np.tensordot(w,x.sum(axis=1),axes=(0,0))
        gram -= np.outer(sx,sx)/n
    return (gram+gram.T)/(2*n)


def canonical(cov):
    sd = np.sqrt(np.maximum(np.diag(cov),0))
    if not np.isfinite(sd).all() or (sd <= 1e-10).any():
        return {'valid':False,'reason':'constant or nonfinite column'}
    r = cov/np.outer(sd,sd)
    xx,yy,xy = r[:6,:6],r[6:,6:],r[:6,6:]
    whiten,conditions = [],[]
    for block in [xx,yy]:
        vals,vecs = np.linalg.eigh(block)
        condition = float(vals[-1]/vals[0]) if vals[0] > 0 else None
        conditions.append(condition)
        if vals[0] <= 1e-8 or condition >= 1e8:
            return {'valid':False,'reason':'rank/condition limit','conditions':conditions}
        whiten.append((vecs/np.sqrt(vals))@vecs.T)
    u,roots,vt = np.linalg.svd(whiten[0]@xy@whiten[1],full_matrices=False)
    ax,by = whiten[0]@u,whiten[1]@vt.T
    loading_a,loading_b = xx@ax,yy@by
    for j in range(6):
        if loading_a[np.argmax(np.abs(loading_a[:,j])),j] < 0:
            ax[:,j] *= -1
            by[:,j] *= -1
    loading_a,loading_b = xx@ax,yy@by
    remaining_a = 1-np.diag(xy@np.linalg.solve(yy,xy.T))
    remaining_b = 1-np.diag(xy.T@np.linalg.solve(xx,xy))
    assert roots.max() <= 1+1e-7 and min(remaining_a.min(),remaining_b.min()) >= -1e-7
    np.testing.assert_allclose(ax.T@xx@ax,np.eye(6),atol=1e-7)
    np.testing.assert_allclose(by.T@yy@by,np.eye(6),atol=1e-7)
    cross_a,cross_b = xy@by,xy.T@ax
    return {'valid':True,'roots':roots.tolist(),'a_weights':ax.tolist(),'b_weights':by.tolist(),
            'a_loadings':loading_a.tolist(),'b_loadings':loading_b.tolist(),
            'remaining_a':np.clip(remaining_a,0,1).tolist(),'remaining_b':np.clip(remaining_b,0,1).tolist(),
            'redundancy_a_first2':float((cross_a[:,:2]**2).sum(axis=1).mean()),
            'redundancy_b_first2':float((cross_b[:,:2]**2).sum(axis=1).mean()),
            'sd':sd.tolist(),'conditions':conditions}


def frozen_correlations(cov,fit):
    wa = np.asarray(fit['a_weights'])/np.asarray(fit['sd'])[:6,None]
    wb = np.asarray(fit['b_weights'])/np.asarray(fit['sd'])[6:,None]
    cross = wa.T@cov[:6,6:]@wb
    sa = np.sqrt(np.diag(wa.T@cov[:6,:6]@wa))
    sb = np.sqrt(np.diag(wb.T@cov[6:,6:]@wb))
    return np.diag(cross)/sa/sb


def month_lags(z,months):
    values = {}
    for m in np.unique(months):
        x = z[months==m].reshape(int((months==m).sum()),-1,13)
        fa = np.fft.fft(x[...,:6],axis=0)
        fb = np.fft.fft(x[...,6:],axis=0)
        spectral = np.einsum('fci,fcj->fij',fa.conj(),fb,optimize=True)
        values[int(m)] = np.fft.ifft(spectral,axis=0).real
    return values


def bootstrap_weights(n,rng):
    start = rng.integers(0,n,size=int(np.ceil(n/7)))
    ix = ((start[:,None]+np.arange(7))%n).ravel()[:n]
    return np.bincount(ix,minlength=n)


def run():
    if (OUT/'summary.json').exists():
        raise FileExistsError('Completed results exist; no automatic refitting.')
    OUT.mkdir(parents=True,exist_ok=True)
    hashes = {f:digest(SOURCE/f) for f in FILES}
    codehash = digest(__file__)
    (OUT/'preregistered_report.md').write_bytes(REPORT.read_bytes())
    dump(OUT/'precheck.json',{'source_hashes':hashes,'code_hash':codehash,'report_hash':digest(REPORT),
                           'seed':SEED,'bootstrap_reps':BOOT,'null_reps':NULL})
    pairs,chosen = mapping()
    pairs.to_json(OUT/'pairs.json',orient='records',force_ascii=False,indent=2)
    s = {'source_hashes':hashes,'code_hash':codehash,'seed':SEED,'python':sys.version,'executable':sys.executable,
         'a_names':A_NAMES,'b_names':B_NAMES,'pairs':len(pairs),'a_stations':int(pairs.wa_station.nunique()),
         'b_stations':int(pairs.wb_station.nunique()),'matching_pairs':int(chosen.sum()),'cases':{}}
    with threadpool_limits(limits=1):
        for round_number in [10,17]:
            covariances,main = {},{}
            for year in [2024,2023]:
                log(f'round={round_number}, year={year}: balanced panel')
                z,dates,audit = load_panel(pairs,round_number,year)
                months = np.asarray(dates.month)
                label = f'{round_number}_{year}'
                np.savez_compressed(OUT/f'panel_{label}.npz',z=z,months=months,
                     dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),matching=chosen)
                day_cross = np.einsum('dci,dcj->dij',z.reshape(len(z),-1,13),z.reshape(len(z),-1,13),optimize=True)
                case = {'audit':audit,'views':{}}
                saved = {}
                for view in ['level','within']:
                    cov = covariance(z,months,within=view=='within',day_cross=day_cross)
                    fit = canonical(cov)
                    case['views'][view] = fit
                    saved['cov_'+view] = cov
                    if view == 'within':
                        covariances[year] = cov
                        main[year] = fit
                if year == 2024 and main[year]['valid']:
                    # Shared calendar-cell means are removed jointly for the null.
                    cz = centered(z,months)
                    lags = month_lags(cz,months)
                    rng = np.random.default_rng(SEED+round_number)
                    null_roots,offsets = [],[]
                    for _ in range(NULL):
                        shifts = [int(rng.integers(1,len(lags[m]))) for m in range(1,13)]
                        cross = sum(lags[m][shift] for m,shift in zip(range(1,13),shifts))/np.prod(z.shape[:-1])
                        ncov = covariances[year].copy()
                        ncov[:6,6:],ncov[6:,:6] = cross,cross.T
                        nf = canonical(ncov)
                        assert nf['valid']
                        null_roots.append(nf['roots'])
                        offsets.append(shifts)
                    null_roots = np.asarray(null_roots)
                    reference = float(np.quantile(null_roots[:,0],.95))
                    main[year]['null_first_root_q95'] = reference
                    main[year]['null_pass'] = (np.asarray(main[year]['roots'])>reference).tolist()
                    saved.update(null_roots=null_roots,null_offsets=np.asarray(offsets))
                    log(f'round={round_number}: null reference complete')
                    boot_roots,boot_a,boot_b,failures = [],[],[],0
                    for rep in range(BOOT):
                        w = bootstrap_weights(len(z),rng)
                        bcov = covariance(z,months,w,True,day_cross)
                        bf = canonical(bcov)
                        if bf['valid']:
                            boot_roots.append(bf['roots']);boot_a.append(bf['remaining_a']);boot_b.append(bf['remaining_b'])
                        else:
                            failures += 1
                            boot_roots.append([np.nan]*6);boot_a.append([np.nan]*6);boot_b.append([np.nan]*7)
                        if rep == 0:
                            saved['first_boot_weights'] = w
                        if rep%50 == 49:
                            log(f'round={round_number}: bootstrap {rep+1}/{BOOT}')
                    saved.update(boot_roots=np.asarray(boot_roots),boot_remaining_a=np.asarray(boot_a),boot_remaining_b=np.asarray(boot_b))
                    good = np.isfinite(saved['boot_roots']).all(axis=1)
                    main[year]['bootstrap_success'] = int(good.sum())
                    main[year]['bootstrap_failures'] = failures
                    for key in ['roots','remaining_a','remaining_b']:
                        arr = saved['boot_'+key]
                        main[year]['interval_'+key] = np.quantile(arr[good],[.025,.975],axis=0).T.tolist() if good.any() else None
                    matched = canonical(covariance(z[:,chosen],months,within=True))
                    main[year]['matched'] = matched
                s['cases'][label] = case
                np.savez_compressed(OUT/f'results_{label}.npz',**saved)
                dump(OUT/'progress.json',s)
            fit24,fit23 = main[2024],main[2023]
            if fit24['valid'] and fit23['valid']:
                frozen = frozen_correlations(covariances[2023],fit24)
                fit24['frozen_2023_correlations'] = frozen.tolist()
                good_lower = np.asarray(fit24['interval_roots'])[:,0] >= .5
                strong = np.asarray(fit24['null_pass']) & good_lower & (frozen>=.5) & (fit24['bootstrap_success']>=180)
                fit24['strong_shared_axes'] = strong.tolist()
    assert hashes == {f:digest(SOURCE/f) for f in FILES}
    assert codehash == digest(__file__)
    s['finished_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')
    dump(OUT/'summary.json',s)
    print(json.dumps({'pairs':s['pairs'],'matching_pairs':s['matching_pairs'],'finished_at':s['finished_at'],
          'strong_axes':{str(r):s['cases'][f'{r}_2024']['views']['within'].get('strong_shared_axes') for r in [10,17]}},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    assert sys.version_info[:2] == (3,13) and not sysconfig.get_config_var('Py_GIL_DISABLED')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['inspect','run'])
    args = parser.parse_args()
    if args.mode == 'inspect':
        inspect()
    else:
        run()
