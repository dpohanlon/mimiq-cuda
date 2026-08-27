import numpy as np
import pytest

import mimiq_cuda


def _nb_data(seed=42, cells=4096, genes=12):
    rng = np.random.default_rng(seed)
    concentration = rng.uniform(2.0, 15.0, genes)
    mean = rng.uniform(0.5, 20.0, genes)
    probability = concentration / (concentration + mean)
    counts = rng.negative_binomial(
        concentration, probability, size=(cells, genes)
    ).astype(np.int32)
    return counts, mean, concentration


def test_nb_cpu_is_finite_and_upper_triangular():
    counts, mean, concentration = _nb_data()
    mi, chi2 = mimiq_cuda.mi_negative_binomial_cpu(
        counts, mean, concentration, max_bins=32
    )
    assert mi.shape == (counts.shape[1], counts.shape[1])
    assert np.all(np.isfinite(mi))
    assert np.all(np.isfinite(chi2))
    assert np.count_nonzero(np.tril(mi)) == 0
    assert np.all(mi >= 0)


def test_normal_and_zinb_cpu_smoke():
    counts, mean, concentration = _nb_data()
    std = counts.std(axis=0).clip(0.1)
    normal_mi, _ = mimiq_cuda.mi_normal_cpu(
        counts, counts.mean(axis=0), std, max_bins=16
    )
    zinb_mi, _ = mimiq_cuda.mi_negative_binomial_zi_cpu(
        counts, mean, concentration, np.zeros(counts.shape[1]), max_bins=16
    )
    nb_mi, _ = mimiq_cuda.mi_negative_binomial_cpu(
        counts, mean, concentration, max_bins=16
    )
    assert np.all(np.isfinite(normal_mi))
    np.testing.assert_array_equal(zinb_mi, nb_mi)


@pytest.mark.skipif(
    not mimiq_cuda.cuda_available(), reason="CUDA device unavailable"
)
def test_cuda_matches_cpu_exact_estimator():
    counts, mean, concentration = _nb_data(cells=8192, genes=16)
    cpu_mi, cpu_chi2 = mimiq_cuda.mi_negative_binomial_cpu(
        counts, mean, concentration, min_pop=25, max_bins=32
    )
    gpu_mi, gpu_chi2 = mimiq_cuda.mi_negative_binomial_gpu(
        counts, mean, concentration, min_pop=25, max_bins=32
    )
    np.testing.assert_allclose(gpu_mi, cpu_mi, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(gpu_chi2, cpu_chi2, rtol=2e-12, atol=2e-12)

    empirical_mean = counts.mean(axis=0)
    empirical_std = counts.std(axis=0).clip(0.1)
    for cpu_function, gpu_function, arguments in [
        (
            mimiq_cuda.mi_normal_cpu,
            mimiq_cuda.mi_normal_gpu,
            (empirical_mean, empirical_std),
        ),
        (
            mimiq_cuda.mi_negative_binomial_zi_cpu,
            mimiq_cuda.mi_negative_binomial_zi_gpu,
            (mean, concentration, np.full(counts.shape[1], 0.17)),
        ),
    ]:
        cpu = cpu_function(counts, *arguments, min_pop=25, max_bins=32)
        gpu = gpu_function(counts, *arguments, min_pop=25, max_bins=32)
        for actual, expected in zip(gpu, cpu):
            np.testing.assert_allclose(
                actual, expected, rtol=2e-12, atol=2e-12
            )


def test_rejects_invalid_input():
    counts, mean, concentration = _nb_data(cells=10, genes=3)
    counts[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        mimiq_cuda.mi_negative_binomial_cpu(counts, mean, concentration)
