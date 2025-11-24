from __future__ import annotations
import numpy as np

def simulate_gbm(mu, sigma, corr, s0, steps, n_paths, dt):
    """
    Simulate correlated Geometric Brownian Motion (GBM) price paths

    Model:
        S_t = S_{t-1} * exp((μ - 0.5 σ²) * dt + σ * sqrt(dt) * ε_t )
        ε_t are correlated shocks produced via Cholesky decomposition.

    Args:
        mu (array): expected returns per asset
        sigma (array): volatilities per asset
        corr (matrix): correlation matrix
        s0 (array): initial prices
        steps (int): number of time steps
        n_paths (int): number of simulated paths
        dt (float): time step size

    Returns:
        ndarray of shape (n_paths, steps, n_assets): simulated price paths
    """

    # turn params into arrays for vectorized calculations
    mu = np.array(mu, dtype=float)
    sigma = np.array(sigma, dtype=float)
    s0 = np.array(s0, dtype=float)

    # build correlated shocks using Cholesky decomposition
    L = np.linalg.cholesky(corr)
    z = np.random.normal(size=(n_paths, steps, len(mu)))
    shocks = z @ L.T

    # calculate drift term to get average expected return for each timestep
    drift = (mu - 0.5 * sigma**2) * dt

    # volatility term
    vol = sigma * np.sqrt(dt)

    # calculate log returns
    log_returns = drift + vol * shocks
    log_returns = np.cumsum(log_returns, axis=1)

    # convert log returns into the price paths
    paths = s0 * np.exp(log_returns)
    return paths
