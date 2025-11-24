import numpy as np
from src.gbm import simulate_gbm

def test_deterministic_drift():
    mu = [0.1]
    sigma = [0.0]
    corr = [[1.0]]
    s0 = [100]
    steps = 5
    dt = 1
    paths = simulate_gbm(mu, sigma, corr, s0, steps, n_paths=1, dt=dt)

    expected = s0[0] * np.exp(0.1 * np.arange(1, steps+1))
    assert np.allclose(paths[0,:,0], expected, atol=1e-8)


def test_constant_path_zero_mu_sigma():
    mu = [0]
    sigma = [0]
    corr = [[1.0]]
    s0 = [100]
    steps = 10
    dt = 1
    paths = simulate_gbm(mu, sigma, corr, s0, steps, n_paths=1, dt=dt)

    assert np.all(paths[0,:,0] == 100)


def test_path_shape():
    mu = [0.1, 0.2]
    sigma = [0.1, 0.3]
    corr = [[1, 0.2], [0.2, 1]]
    s0 = [100, 50]

    steps = 30
    n_paths = 25
    paths = simulate_gbm(mu, sigma, corr, s0, steps, n_paths, dt=1/252)

    assert paths.shape == (n_paths, steps, len(mu))


def test_no_nans():
    mu = [0.1, 0.1]
    sigma = [0.2, 0.2]
    corr = [[1, 0.5], [0.5, 1]]
    s0 = [100, 100]
    paths = simulate_gbm(mu, sigma, corr, s0, steps=50, n_paths=50, dt=1/252)

    assert not np.isnan(paths).any()


def test_empirical_correlation_close():
    mu = [0.05, 0.05]
    sigma = [0.2, 0.2]
    corr = [[1.0, 0.8],
            [0.8, 1.0]]
    s0 = [100, 100]

    paths = simulate_gbm(mu, sigma, corr, s0, steps=252, n_paths=5000, dt=1/252)

    log_returns = np.diff(np.log(paths), axis=1)
    r1 = log_returns[:, :, 0].flatten()
    r2 = log_returns[:, :, 1].flatten()

    empirical_corr = np.corrcoef(r1, r2)[0,1]

    assert np.isclose(empirical_corr, 0.8, atol=0.02)
