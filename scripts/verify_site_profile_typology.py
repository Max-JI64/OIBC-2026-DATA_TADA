"""Independent metric and profile reconstruction checks for saved DA-002 outputs."""
import json
import sys

import numpy as np

from site_profile_typology import OUT, SOURCE, FILES, digest, dump


def ari_direct(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    counts = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    np.add.at(counts, (a, b), 1)
    choose2 = lambda x: x * (x - 1) / 2
    total = choose2(len(a))
    cells = choose2(counts).sum()
    rows = choose2(counts.sum(axis=1)).sum()
    cols = choose2(counts.sum(axis=0)).sum()
    expected = rows * cols / total
    denominator = .5 * (rows + cols) - expected
    return 1. if abs(denominator) < 1e-14 else float((cells - expected) / denominator)


def silhouette_direct(x, labels):
    labels = np.asarray(labels)
    distance = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(axis=2))
    values = []
    for i in range(len(x)):
        own = labels == labels[i]
        if own.sum() == 1:
            values.append(0.)
            continue
        within = distance[i, own].sum() / (own.sum() - 1)
        other = min(distance[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
        values.append((other - within) / max(within, other))
    return float(np.mean(values))


def profiles_direct(cube, quarters, weights=None, condition=None):
    if weights is None:
        weights = np.ones(len(cube))
    if condition is None:
        condition = np.ones(cube.shape[:2], dtype=bool)
    raw = np.zeros((cube.shape[1], 4, cube.shape[2]))
    for site in range(cube.shape[1]):
        for q in range(1, 5):
            take = (quarters == q) & condition[:, site]
            raw[site, q-1] = np.average(cube[take, site], axis=0, weights=weights[take])
    shape = raw / raw.sum(axis=2, keepdims=True)
    return shape.reshape(len(shape), -1) / 2, shape, raw


def check_partition(x, part):
    centers = np.asarray(part['centers'])
    labels = np.asarray(part['labels'])
    sizes = np.bincount(labels, minlength=part['k'])
    np.testing.assert_array_equal(sizes, part['sizes'])
    dist2 = ((x[:, None] - centers[None, :]) ** 2).sum(axis=2)
    assert np.all(dist2[np.arange(len(x)), labels] <= dist2.min(axis=1) + 1e-10)
    np.testing.assert_allclose(dist2[np.arange(len(x)), labels].sum(), part['inertia'], atol=1e-10)
    np.testing.assert_allclose(silhouette_direct(x, labels), part['silhouette'], atol=1e-10)
    assert part['admissible'] == bool(sizes.min() >= max(5, int(np.ceil(.1 * len(x)))))


def verify():
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'precheck.json').read_text(encoding='utf-8'))
    assert s['source_hashes'] == pre['source_hashes'] == {f: digest(SOURCE / f) for f in FILES}
    assert s['code_hash'] == pre['code_hash'] == digest(OUT.parents[2] / 'scripts/site_profile_typology.py')
    with np.load(OUT / 'panel_2024.npz') as p24, np.load(OUT / 'panel_2023.npz') as p23, np.load(OUT / 'results.npz') as r:
        for panel, year in [(p24, '2024'), (p23, '2023')]:
            x, shapes, raw = profiles_direct(panel['cf'], panel['quarters'])
            np.testing.assert_allclose(x, panel['features'], atol=1e-12)
            np.testing.assert_allclose(shapes, panel['profiles'], atol=1e-12)
            np.testing.assert_allclose(raw, panel['raw_profiles'], atol=1e-12)
            assert panel['cf'].shape[1:] == (80, 11)
            a = s['audit'][year]
            assert a['balanced_rows'] == panel['cf'].size and sum(a['dates_per_quarter']) == len(panel['cf'])
        for part in s['candidates']:
            check_partition(p24['features'], part)
        check_partition(p23['features'], s['replication_2023'])
        chosen = s['diagnostic_partition']
        labels = np.asarray(chosen['labels'])
        np.testing.assert_allclose(ari_direct(labels, s['replication_2023']['labels']), s['replication_2023']['ari'])
        boots = r['bootstrap_labels']
        aris = [ari_direct(labels, v) for v in boots]
        np.testing.assert_allclose(aris, r['bootstrap_ari'], atol=1e-12)
        np.testing.assert_allclose(np.quantile(aris, [.05, .5, .95]), s['bootstrap']['ari_q05_median_q95'])
        coassign = np.zeros((len(labels), len(labels)))
        size_ok = []
        for v in boots:
            coassign += v[:, None] == v[None, :]
            size_ok.append(np.bincount(v, minlength=chosen['k']).min() >= 8)
        coassign /= len(boots)
        np.testing.assert_allclose(coassign, r['coassignment'])
        np.testing.assert_array_equal(size_ok, r['bootstrap_size_pass'])
        np.testing.assert_allclose((boots == labels).mean(axis=0), r['assignment_agreement'])
        assert s['bootstrap']['success'] + sum(s['bootstrap']['failures'].values()) == 200
        assert r['first_boot_weights'].sum() == len(p24['cf'])
        fboot, _, _ = profiles_direct(p24['cf'], p24['quarters'], r['first_boot_weights'])
        boot0 = boots[0]
        means = np.stack([fboot[boot0 == c].mean(axis=0) for c in range(chosen['k'])])
        distance = ((fboot[:, None] - means[None]) ** 2).sum(axis=2)
        assert np.all(distance[np.arange(len(labels)), boot0] <= distance.min(axis=1) + 1e-9)
        np.testing.assert_allclose(np.quantile(r['null_max_silhouette'], [.05, .5, .95]), s['reference']['silhouette_q05_median_q95'])
        assert np.isfinite(r['null_max_silhouette']).sum() == s['reference']['valid']
        eligible = r['weather_eligible']
        for panel, year in [(p24, 2024), (p23, 2023)]:
            xf, sh, _ = profiles_direct(panel['cf'][:, eligible], panel['quarters'], condition=panel['condition50'][:, eligible])
            np.testing.assert_allclose(xf, r[f'weather_features_{year}'])
            np.testing.assert_allclose(sh, r[f'weather_shapes_{year}'])
            check_partition(xf, s['weather_comparison'][f'moderate_{year}'])
        weather24 = s['weather_comparison']['moderate_2024']['labels']
        weather23 = s['weather_comparison']['moderate_2023']['labels']
        assert eligible.all()
        np.testing.assert_allclose(ari_direct(labels, weather24), s['weather_comparison']['full_vs_moderate_ari'])
        np.testing.assert_allclose(ari_direct(weather24, weather23), s['weather_comparison']['moderate_annual_ari'])
        amplitude = s['amplitude_sensitivity']['partition']
        check_partition(p24['raw_profiles'].reshape(80, -1) / 2, amplitude)
        np.testing.assert_allclose(ari_direct(labels, amplitude['labels']), s['amplitude_sensitivity']['ari_vs_shape'])
        gram = (p24['features'] - p24['features'].mean(0)) @ (p24['features'] - p24['features'].mean(0)).T
        eigen = np.linalg.eigvalsh(gram)[::-1]
        np.testing.assert_allclose(eigen[:2] / eigen.sum(), s['pca']['explained_variance_ratio_first_two'])
    result = {'source_files_unchanged': 3, 'analysis_code_hash_matches': True,
              'annual_profiles_reconstructed': 2, 'weather_profiles_reconstructed': 2,
              'partitions_checked_with_direct_silhouette_and_nearest_centers': 8,
              'bootstrap_ari_and_coassignment_checks': 200, 'first_boot_partition_check': True,
              'null_quantiles_and_amplitude_ari_checked': True, 'pca_eigenvalue_check': True,
              'reported_fit_calls': int(sum(s['fit_calls'].values()))}
    dump(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    verify()
