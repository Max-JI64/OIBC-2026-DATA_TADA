"""Independent QR/SVD, least-squares and direct resampling checks for DA-005."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.linalg import svdvals
from threadpoolctl import threadpool_limits

from weather_provider_information import ROOT,SOURCE,OUT,FILES,digest,dump


def absorb_direct(z,months):
    out = z.copy()
    for m in set(months):
        take = months == m
        out[take] -= out[take].mean(axis=0)
    return out


def direct_metrics(x):
    flat = x.reshape(-1,13)
    flat = flat-flat.mean(axis=0)
    a,b = flat[:,:6],flat[:,6:]
    qa = np.linalg.qr(a,mode='reduced')[0]
    qb = np.linalg.qr(b,mode='reduced')[0]
    roots = svdvals(qa.T@qb)
    ea = a-b@np.linalg.lstsq(b,a,rcond=None)[0]
    eb = b-a@np.linalg.lstsq(a,b,rcond=None)[0]
    return roots,(ea**2).sum(axis=0)/(a**2).sum(axis=0),(eb**2).sum(axis=0)/(b**2).sum(axis=0)


def source_cube(panel,pairs,round_number):
    dates = pd.to_datetime(panel['dates']).tz_localize('Asia/Seoul')
    # Reindex source station/time keys independently of the analysis joins.
    times = pd.DatetimeIndex([d+pd.Timedelta(hours=h+1) for d in dates for _ in range(len(pairs)) for h in range(9,17)])
    blocks = []
    for provider in ['wa','wb']:
        stations = np.tile(np.repeat(pairs[provider+'_station'].to_numpy(),8),len(dates))
        humidity = provider+('_humidity' if provider == 'wa' else '_rel_hum')
        cols = [provider+'_temp',humidity,provider+'_cloud',provider+'_wind_speed',provider+'_rain']
        if provider == 'wb':
            cols += ['wb_uv_idx']
        frame = pd.read_parquet(SOURCE/f'train/forecast_{provider}.parquet',columns=[provider+'_station','ts','round']+cols)
        frame = frame.loc[frame['round'].eq(round_number)].set_index([provider+'_station','ts'])
        assert frame.index.is_unique
        idx = pd.MultiIndex.from_arrays([stations,times],names=frame.index.names)
        values = frame.reindex(idx)[cols].to_numpy().reshape(len(dates),len(pairs),8,len(cols))
        result = [values[...,0],values[...,1],values[...,2],values[...,3]/(3.6 if provider == 'wb' else 1),
                  (values[...,4]>.1).astype(float),np.log1p(values[...,4])]
        if provider == 'wb':
            result.append(values[...,5])
        blocks.append(np.stack(result,axis=-1))
    return np.concatenate(blocks,axis=-1)


def verify():
    s = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT/'precheck.json').read_text(encoding='utf-8'))
    assert pre['source_hashes'] == s['source_hashes'] == {f:digest(SOURCE/f) for f in FILES}
    assert pre['code_hash'] == s['code_hash'] == digest(ROOT/'scripts/weather_provider_information.py')
    assert pre['report_hash'] == digest(OUT/'preregistered_report.md')
    pairs = pd.read_json(OUT/'pairs.json')
    assert len(pairs) == 52 and not pairs.duplicated().any()
    source_values = 0
    with threadpool_limits(limits=1):
        for round_number in [10,17]:
            fit24 = s['cases'][f'{round_number}_2024']['views']['within']
            for year in [2024,2023]:
                label = f'{round_number}_{year}'
                p = np.load(OUT/f'panel_{label}.npz')
                r = np.load(OUT/f'results_{label}.npz')
                rebuilt = source_cube(p,pairs,round_number)
                np.testing.assert_allclose(rebuilt,p['z'],atol=1e-12,rtol=0)
                source_values += rebuilt.size
                for view in ['level','within']:
                    z = absorb_direct(rebuilt,p['months']) if view == 'within' else rebuilt
                    flat = z.reshape(-1,13).copy()
                    flat -= flat.mean(axis=0)
                    direct_cov = flat.T@flat/len(flat)
                    np.testing.assert_allclose(direct_cov,r['cov_'+view],atol=1e-9,rtol=1e-10)
                    fit = s['cases'][label]['views'][view]
                    roots,ea,eb = direct_metrics(z)
                    np.testing.assert_allclose(roots,fit['roots'],atol=1e-9)
                    np.testing.assert_allclose(ea,fit['remaining_a'],atol=1e-9)
                    np.testing.assert_allclose(eb,fit['remaining_b'],atol=1e-9)
                    sd = np.std(flat,axis=0)
                    scores_a = (flat[:,:6]/sd[:6])@np.asarray(fit['a_weights'])
                    scores_b = (flat[:,6:]/sd[6:])@np.asarray(fit['b_weights'])
                    load_a = np.corrcoef(flat[:,:6].T,scores_a.T)[:6,6:]
                    load_b = np.corrcoef(flat[:,6:].T,scores_b.T)[:7,7:]
                    np.testing.assert_allclose(load_a,fit['a_loadings'],atol=1e-9)
                    np.testing.assert_allclose(load_b,fit['b_loadings'],atol=1e-9)
                    cross_a = np.corrcoef(flat[:,:6].T,scores_b.T)[:6,6:]
                    cross_b = np.corrcoef(flat[:,6:].T,scores_a.T)[:7,7:]
                    np.testing.assert_allclose((cross_a[:,:2]**2).sum(axis=1).mean(),fit['redundancy_a_first2'],atol=1e-9)
                    np.testing.assert_allclose((cross_b[:,:2]**2).sum(axis=1).mean(),fit['redundancy_b_first2'],atol=1e-9)
                cz = absorb_direct(rebuilt,p['months'])
                if year == 2023:
                    flat = cz.reshape(-1,13)
                    weights_a = np.asarray(fit24['a_weights'])/np.asarray(fit24['sd'])[:6,None]
                    weights_b = np.asarray(fit24['b_weights'])/np.asarray(fit24['sd'])[6:,None]
                    aa,bb = flat[:,:6]@weights_a,flat[:,6:]@weights_b
                    actual = np.array([np.corrcoef(aa[:,j],bb[:,j])[0,1] for j in range(6)])
                    np.testing.assert_allclose(actual,fit24['frozen_2023_correlations'],atol=1e-9)
                    continue
                chosen = p['matching']
                selected = pairs.loc[chosen]
                assert selected.wa_station.is_unique and selected.wb_station.is_unique and len(selected)==38
                # Independently prove maximum cardinality with augmenting paths.
                adjacency = {a:sorted(pairs.loc[pairs.wa_station.eq(a),'wb_station']) for a in sorted(pairs.wa_station.unique())}
                matches = {}
                def augment(a,seen):
                    for b in adjacency[a]:
                        if b in seen:
                            continue
                        seen.add(b)
                        if b not in matches or augment(matches[b],seen):
                            matches[b] = a
                            return True
                    return False
                assert sum(augment(a,set()) for a in adjacency) == 38
                mr,ma,mb = direct_metrics(cz[:,chosen])
                np.testing.assert_allclose(mr,fit24['matched']['roots'],atol=1e-9)
                np.testing.assert_allclose(ma,fit24['matched']['remaining_a'],atol=1e-9)
                np.testing.assert_allclose(mb,fit24['matched']['remaining_b'],atol=1e-9)
                shifted = cz.copy()
                for month,lag in zip(range(1,13),r['null_offsets'][0]):
                    take = p['months']==month
                    shifted[take,...,6:] = np.roll(cz[take,...,6:],-int(lag),axis=0)
                np.testing.assert_allclose(direct_metrics(shifted)[0],r['null_roots'][0],atol=1e-9)
                threshold = np.quantile(r['null_roots'][:,0],.95)
                np.testing.assert_allclose(threshold,fit24['null_first_root_q95'])
                weights = r['first_boot_weights']
                ix = np.repeat(np.arange(len(cz)),weights)
                boot_z = absorb_direct(rebuilt[ix],p['months'][ix])
                br,ba,bb = direct_metrics(boot_z)
                for key,first in [('roots',br),('remaining_a',ba),('remaining_b',bb)]:
                    np.testing.assert_allclose(first,r['boot_'+key][0],atol=1e-9)
                    good = np.isfinite(r['boot_roots']).all(axis=1)
                    np.testing.assert_allclose(np.quantile(r['boot_'+key][good],[.025,.975],axis=0).T,fit24['interval_'+key])
                assert int(good.sum()) == fit24['bootstrap_success'] == 200
                strong = (np.asarray(fit24['roots'])>threshold) & (np.asarray(fit24['interval_roots'])[:,0]>=.5) & (np.asarray(fit24['frozen_2023_correlations'])>=.5)
                np.testing.assert_array_equal(strong,fit24['strong_shared_axes'])
    checks = {'passed':True,'source_feature_values':int(source_values),'qr_cca_and_lstsq_cases':8,
              'matched_reconstructions':2,'maximum_matching_cardinality':38,'direct_first_null_replays':2,
              'direct_first_bootstrap_replays':2,'saved_null_draws':398,'saved_bootstrap_draws':400,
              'frozen_year_transfer_cases':2}
    dump(OUT/'verification.json',checks)
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    verify()
