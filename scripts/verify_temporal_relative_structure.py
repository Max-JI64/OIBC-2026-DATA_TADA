"""Independent source aggregation, SVD and direct scan checks for DA-004."""
import json
import hashlib
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from temporal_relative_structure import ROOT, SOURCE, PRIOR, OUT, REPORT, FILES, digest, dump


def direct_scan(z):
    candidates = range(90, len(z) - 89)
    changes = np.stack([z[k:].mean(axis=0) - z[:k].mean(axis=0) for k in candidates])
    factors = np.sqrt(np.asarray(list(candidates)) * (len(z) - np.asarray(list(candidates))) / len(z))
    sd = np.sqrt(((z - z.mean(axis=0)) ** 2).sum(axis=0) / (len(z) - 1))
    stats = np.abs(changes) * factors[:, None] / sd
    chosen = stats.argmax(axis=0)
    return stats, chosen + 90, stats[chosen, np.arange(z.shape[1])], changes[chosen, np.arange(z.shape[1])]


def verify():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert pre['source_hashes'] == s['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert pre['code_hash'] == s['code_hash'] == digest(ROOT / 'scripts/temporal_relative_structure.py')
    # Original preregistration uses LF; Windows write_text made a CRLF snapshot.
    normalized_snapshot = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').encode('utf-8')
    assert pre['report_hash'] == hashlib.sha256(normalized_snapshot).hexdigest()
    assert s['audit']['prior_code_hash'] == digest(ROOT / 'scripts/weather_conditional_relationships.py')
    for name, value in s['audit']['derived_panel_hashes'].items():
        assert digest(PRIOR / name) == value
    # Trusted artifact produced locally by this analysis; station strings are object dtype.
    p = np.load(OUT / 'panel.npz', allow_pickle=True)
    r = np.load(OUT / 'results.npz')
    ids, stations = p['site_ids'], p['stations']
    dates = pd.to_datetime(p['dates'])
    gen = pd.read_parquet(SOURCE / 'train/generation.parquet')
    start = gen.ts - pd.Timedelta(hours=1)
    take = gen.site_id.isin(ids) & start.dt.year.isin([2023, 2024]) & start.dt.hour.isin(range(9, 17))
    g = gen.loc[take].copy()
    g['day'] = start[take].dt.strftime('%Y-%m-%d')
    totals = g.groupby(['day', 'site_id']).energy.agg(['sum', 'size'])
    assert totals['size'].eq(8).all()
    sites = pd.read_csv(SOURCE / 'sites.csv', encoding='utf-8-sig').set_index('site_id')
    independently = totals['sum'].unstack().loc[p['dates'], ids].to_numpy() / (8 * sites.loc[ids].capacity_kw.to_numpy())
    np.testing.assert_allclose(independently, p['daily'], atol=1e-12, rtol=0)
    expected_ok = np.array([(stations == t).sum() >= 2 for t in stations])
    np.testing.assert_array_equal(expected_ok, p['eligible'])
    eligible = np.flatnonzero(expected_ok)
    peer_difference = []
    for j in eligible:
        others = (stations == stations[j]) & (np.arange(len(stations)) != j)
        peer_difference.append(independently[:, j] - independently[:, others].mean(axis=1))
    relative = np.column_stack(peer_difference)
    np.testing.assert_allclose(relative, p['relative'], atol=1e-12, rtol=0)
    phase = np.array([(d.dayofyear - 1) / (366 if d.is_leap_year else 365) for d in dates])
    significant = np.ones(len(eligible), dtype=bool)
    with threadpool_limits(limits=1):
        for degree in [3, 6]:
            x = np.column_stack([np.ones(len(dates))] + [f(2*np.pi*h*phase) for h in range(1, degree+1) for f in [np.sin, np.cos]])
            residual = relative - x @ np.linalg.lstsq(x, relative, rcond=None)[0]
            np.testing.assert_allclose(residual, r[f'residual_{degree}'], atol=1e-12, rtol=0)
            stats, cuts, peaks, changes = direct_scan(residual)
            case = s['cases'][str(degree)]
            np.testing.assert_allclose(stats, r[f'scores_{degree}'], atol=1e-10, rtol=1e-10)
            np.testing.assert_array_equal(cuts, case['cut_indices'])
            np.testing.assert_allclose(peaks, case['statistics'], atol=1e-10)
            np.testing.assert_allclose(changes * 100, case['delta_pp'], atol=1e-10)
            third_results, d90, passes = [], [], []
            for j, k in enumerate(cuts):
                before = residual[k-90:k, j]
                after = residual[k:k+90, j]
                d90.append(after.sum()/90 - before.sum()/90)
                thirds = [after[a:a+30].sum()/30 - before[b:b+30].sum()/30 for a, b in [(0, 60), (30, 30), (60, 0)]]
                third_results.append(thirds)
                passes.append(abs(d90[-1]) >= .02 and all(t * changes[j] > 0 for t in thirds))
            np.testing.assert_allclose(np.asarray(d90)*100, case['delta90_pp'], atol=1e-10)
            np.testing.assert_allclose(np.asarray(third_results)*100, case['thirds_pp'], atol=1e-10)
            np.testing.assert_array_equal(passes, case['persistent'])
            for block in [7, 14]:
                maxima = r[f'null_max_{degree}_{block}']
                assert len(maxima) == 499 and np.isfinite(maxima).all()
                direct_p = np.array([(1 + sum(t >= v for t in maxima)) / 500 for v in peaks])
                np.testing.assert_allclose(direct_p, case['blocks'][str(block)]['p_adjusted'])
                np.testing.assert_allclose(np.quantile(maxima, [.5, .95, .99]), case['blocks'][str(block)]['null_q50_q95_q99'])
                assert int((direct_p <= .05).sum()) == case['blocks'][str(block)]['significant_sites']
                significant &= direct_p <= .05
                ix = r[f'first_indices_{degree}_{block}']
                full_null = x @ np.linalg.lstsq(x, relative, rcond=None)[0] + residual[ix]
                null_res = full_null - x @ np.linalg.lstsq(x, full_null, rcond=None)[0]
                np.testing.assert_allclose(direct_scan(null_res)[2].max(), maxima[0], atol=1e-10)
    a, b = s['cases']['3'], s['cases']['6']
    close = np.abs(np.asarray(a['cut_indices']) - np.asarray(b['cut_indices'])) <= 60
    same = np.asarray(a['delta_pp']) * np.asarray(b['delta_pp']) > 0
    final = significant & close & same & np.asarray(a['persistent'])
    np.testing.assert_array_equal(significant, [v['all_four_significant'] for v in s['sites']])
    np.testing.assert_array_equal(final, [v['robust_candidate'] for v in s['sites']])
    assert int(final.sum()) == s['counts']['robust_candidates']
    assert len(set(np.asarray(stations)[eligible][final])) == s['counts']['candidate_stations']
    raw90_checks = []
    report_text = REPORT.read_text(encoding='utf-8')
    for j in np.flatnonzero(final):
        original_site = eligible[j]
        peers = (stations == stations[original_site]) & (np.arange(len(stations)) != original_site)
        raw_difference = independently[:, original_site] - independently[:, peers].mean(axis=1)
        k = a['cut_indices'][j]
        raw_change = (raw_difference[k:k+90].sum() - raw_difference[k-90:k].sum()) / 90 * 100
        seasonal_change = raw_change - a['delta90_pp'][j]
        expected_row = f"| {s['sites'][j]['site_id']} | {raw_change:+.3f} | {seasonal_change:+.3f} | {a['delta90_pp'][j]:+.3f} |"
        assert expected_row in report_text
        raw90_checks.append({'site_id': s['sites'][j]['site_id'], 'raw_relative_delta90_pp': float(raw_change),
                             'fitted_seasonal_delta90_pp': float(seasonal_change)})
    # Two-site leave-one-out comparisons must have mirror statistics, not two independent events.
    pairs = 0
    for station in np.unique(stations):
        ix = np.flatnonzero(stations[eligible] == station)
        if len(ix) == 2:
            np.testing.assert_allclose(relative[:, ix[0]], -relative[:, ix[1]], atol=1e-12)
            pairs += 1
    checks = {'passed': True, 'daily_source_cells': int(independently.size), 'relative_cells': int(relative.size),
              'svd_residual_and_scan_cases': 2, 'global_pvalues': int(len(eligible)*4),
              'saved_null_replicates': 1996, 'independent_first_null_replays': 4,
              'mirror_station_pairs': pairs, 'robust_candidates': int(final.sum()),
              'source_based_raw90_report_checks': raw90_checks}
    dump(OUT / 'verification.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    verify()
