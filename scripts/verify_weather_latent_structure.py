"""Re-read saved sufficient statistics and independently check DA-001 results."""
import json
import sys

import numpy as np

from weather_latent_structure import OUT, SOURCE, FILES, digest, dump
from eda_plot_utils import check_gallery


def verify():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert summary['source_hashes'] == pre['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert summary['code_hash'] == pre['code_hash'] == digest(OUT.parents[2] / 'scripts/weather_latent_structure.py')
    assert len(summary['primary']) == 6 and len(summary['sensitivity']) == 3
    checks, factor_checks = 0, 0
    for label, audit in summary['audit'].items():
        for a in audit:
            assert a['balanced_rows'] == a['balanced_dates'] * a['stations'] * 24
            assert sum(a['dates_per_month']) == a['balanced_dates'] and min(a['dates_per_month']) >= 14
            assert a['available_dates'] - a['excluded_dates'] == a['balanced_dates']
            assert a['balanced_rows'] <= a['raw_rows']
    for label, group in summary['primary'].items():
        for view, case in group.items():
            with np.load(OUT / f'{label}_{view}_stats.npz') as stats:
                for suffix, stored in [('', case['correlation']), ('23', None)]:
                    s = stats['sums' + suffix]
                    c = stats['crosses' + suffix]
                    n = len(s) * int(stats['ncell' + suffix])
                    mean = s.sum(0) / n
                    cov = c.sum(0) / n - np.outer(mean, mean)
                    sd = np.sqrt(np.diag(cov))
                    r = cov / np.outer(sd, sd)
                    if suffix == '':
                        np.testing.assert_allclose(r, stored, atol=1e-10)
                        eig = np.linalg.eigvalsh(r)[::-1]
                        np.testing.assert_allclose(eig, case['diagnostics']['eigenvalues'], atol=1e-10)
                        np.testing.assert_allclose(eig[:2].sum() / len(eig), case['diagnostics']['pca_2'])
                        assert case['selected_raw'] == int((eig > np.asarray(case['null_threshold'])).sum())
                        inv = np.linalg.inv(r)
                        partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
                        np.fill_diagonal(partial, 0)
                        zero_r = r.copy()
                        np.fill_diagonal(zero_r, 0)
                        kmo = (zero_r ** 2).sum() / ((zero_r ** 2).sum() + (partial ** 2).sum())
                        np.testing.assert_allclose(kmo, case['diagnostics']['kmo'], atol=1e-10)
                        f = case['factor']
                    else:
                        f = case['replication_2023']['factor']
                    if 'loadings' in f:
                        load = np.asarray(f['loadings'])
                        h = (load ** 2).sum(1)
                        np.testing.assert_allclose(h, f['communality'], atol=1e-10)
                        np.testing.assert_allclose(1 - h, f['uniqueness'], atol=1e-10)
                        res = r - load @ load.T
                        off = res[~np.eye(len(r), dtype=bool)]
                        np.testing.assert_allclose(np.sqrt((off ** 2).mean()), f['rms_residual'], atol=1e-10)
                        if f['valid']:
                            assert h.min() >= -1e-8 and h.max() <= 1 + 1e-8 and f['fixed_point_error'] < 1e-6
                        factor_checks += 1
                checks += 1
                if 'bootstrap' in case:
                    b = case['bootstrap']
                    assert b['attempts'] == b['success'] + sum(b['failure_reasons'].values()) == 100
                    if b['success']:
                        ci = np.asarray(b['loading_intervals'])
                        assert (np.diff(ci, axis=0) >= -1e-12).all()
                if case['factor']['valid'] and case['replication_2023']['factor']['valid']:
                    a = np.asarray(case['factor']['loadings'])
                    b = np.asarray(case['replication_2023']['aligned_loadings'])
                    cong = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
                    np.testing.assert_allclose(cong, case['replication_2023']['congruence'])
    for label, group in summary['sensitivity'].items():
        for view, case in group.items():
            base = summary['primary'][label][view]
            assert case['n'] == base['n'] and case['columns'] == base['columns'][:6]
            np.testing.assert_allclose(case['correlation'], np.asarray(base['correlation'])[:6, :6], atol=1e-10)
    result = {'primary_views': checks, 'sensitivity_views': 6, 'factor_reconstructions': factor_checks,
              'source_files_unchanged': len(FILES), 'analysis_code_hash_matches': True,
              'accounting_kmo_eigenvalues_pca_residuals': True,
              'bootstrap_accounting_and_same_row_sensitivity': True,
              **check_gallery(OUT, 8)}
    dump(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    verify()
