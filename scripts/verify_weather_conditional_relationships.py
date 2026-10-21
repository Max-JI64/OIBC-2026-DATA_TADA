"""Verify saved DA-003 coefficients using direct residualization and SVD."""
import json
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from weather_conditional_relationships import (OUT, SOURCE, FILES, CONTRASTS,
    digest, dump, design, contrast_vector, support_mask, support_stats, block_weights, SEED)


def residualize_direct(a, months, weights):
    out = a.copy()
    for m in np.unique(months):
        idx = months == m
        w = weights[idx]
        if w.sum() == 0:
            out[idx] = 0
        else:
            out[idx] -= np.average(a[idx], axis=0, weights=w)
    return out


def verify():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert summary['source_hashes'] == pre['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert summary['code_hash'] == pre['code_hash'] == digest(OUT.parents[2] / 'scripts/weather_conditional_relationships.py')
    assert len(summary['cases']) == 4
    checked, direct_checks, support_checks = 0, 0, 0
    errors = []
    with threadpool_limits(limits=1):
        for label, case in summary['cases'].items():
            with np.load(OUT / f'{label}_panel.npz') as panel, np.load(OUT / f'{label}_results.npz') as saved:
                weather, y, months = panel['weather'], panel['response'], panel['months']
                x, names, _ = design(weather)
                nday, nsite, nhour = y.shape
                a = case['audit']
                assert nsite == 80 and nhour == 8 and a['balanced_rows'] == nday * nsite * nhour
                assert a['balanced_dates'] == nday and sum(a['dates_per_month']) == nday
                assert names == case['basis_names']
                assert int((y == 0).sum()) == a['zero_cf_rows']
                for c in CONTRASTS:
                    ends = [support_stats(support_mask(weather, c[k])) for k in ['a', 'b']]
                    assert ends == case['support'][c['id']]['endpoints']
                    assert all(e['supported'] for e in ends) == case['support'][c['id']]['supported']
                    support_checks += 1
                for mode in ['point', 'first_bootstrap']:
                    weights = np.ones(nday) if mode == 'point' else saved['first_boot_weights']
                    cx = residualize_direct(x, months, weights).reshape(-1, x.shape[-1])
                    cy = residualize_direct(y, months, weights).ravel()
                    root_w = np.sqrt(np.repeat(weights, nsite * nhour))
                    if mode == 'point':
                        np.testing.assert_allclose(cx.T @ cx, saved['xx'], atol=1e-6, rtol=1e-8)
                        np.testing.assert_allclose(cx.T @ cy, saved['xy'], atol=1e-7, rtol=1e-8)
                        np.testing.assert_allclose(cy @ cy, saved['yy'], atol=1e-7)
                    for model in ['full', 'no_uv']:
                        fit = case['models'][model]
                        assert fit['valid']
                        cols = fit['columns']
                        got = np.linalg.lstsq(cx[:, cols] * root_w[:, None], cy * root_w, rcond=None)[0]
                        want = np.asarray(fit['beta']) if mode == 'point' else saved['boot_' + model][0]
                        error = float(np.max(np.abs(got - want)))
                        errors.append(error)
                        np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-6)
                        direct_checks += 1
                for model, fit in case['models'].items():
                    matrix = saved['boot_' + model]
                    good = np.isfinite(matrix).all(axis=1)
                    assert good.sum() == fit['bootstrap_success']
                    assert fit['bootstrap_success'] + sum(fit['bootstrap_failures'].values()) == 200
                    for c in CONTRASTS:
                        if c['id'] not in fit['contrasts']:
                            continue
                        values = fit['contrasts'][c['id']]
                        delta = contrast_vector(c)[fit['columns']]
                        np.testing.assert_allclose(np.asarray(fit['beta']) @ delta, values['pp'])
                        np.testing.assert_allclose(np.quantile(matrix[good] @ delta, [.025, .975]), values['interval'])
                        checked += 1
                for c in CONTRASTS:
                    key = c['id']
                    if key not in case['uv_adjustment']:
                        continue
                    full, reduced = case['models']['full'], case['models']['no_uv']
                    delta = contrast_vector(c)
                    diff = saved['boot_full'] @ delta[full['columns']] - saved['boot_no_uv'] @ delta[reduced['columns']]
                    np.testing.assert_allclose(np.quantile(diff[np.isfinite(diff)], [.025, .975]), case['uv_adjustment'][key]['interval'])
                for c in CONTRASTS:
                    h = case['heterogeneity']['contrasts'][c['id']]
                    values = saved['site_values_' + c['id']]
                    assert len(values) == h['sites'] and int((values > 0).sum()) == h['positive']
                    assert int((values < 0).sum()) == h['negative']
                    if len(values):
                        np.testing.assert_allclose(np.quantile(values, [.25, .5, .75]), h['q25_median_q75'])
    # Locate the aborted second attempt's bootstrap iteration without repeating any fit.
    with np.load(OUT / 'observed_2024_panel.npz') as panel:
        months = panel['months']
        rng = np.random.default_rng(SEED)
        failed_at = None
        for i in range(200):
            w = block_weights(len(months), rng)
            if any(w[months == m].sum() == 0 for m in np.unique(months)):
                failed_at = i + 1
                break
    result = {'source_files_unchanged': 4, 'analysis_code_hash_matches': True,
              'direct_svd_fits': direct_checks, 'max_direct_beta_absolute_error': max(errors),
              'point_and_interval_contrasts': checked, 'support_checks': support_checks,
              'paired_uv_difference_and_site_summary_checks': True,
              'aborted_second_attempt_bootstrap_iteration': failed_at,
              'successful_run_point_fits': 32, 'successful_run_bootstrap_fits': 6400,
              'successful_run_site_fits': 320}
    dump(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    verify()
