import numpy as np

def get_portfolio_values(paths, weights):
    """
    Calculate portfolio value over time from simulated asset price paths.

    Args:
        paths: ndarray (n_paths, steps, n_assets)
        weights: array-like of length n_assets

    Returns:
        ndarray (n_paths, steps)
    """
    weights = np.array(weights, dtype=float)

    # calculate weighted sum across the assets with no rebalancing
    portfolio = np.sum(paths * weights, axis=2)

    return portfolio


def normalize_weights(weights):
    """
    Normalize weights so they sum to 1
    """
    weights = np.array(weights, dtype=float)
    return weights / weights.sum()


def rebalance_portfolio(paths, weights, rebalance_step=21):
    """
    Rebalance portfolio by reallocating dollars to initial
    weights periodically.

    Args:
        paths: ndarray (n_paths, steps, n_assets)
               Simulated price paths from GBM
        weights: target weights that sum to 1.
        rebalance_step: how often to rebalance, roughly 21 to rebalance monthly

    Returns:
        ndarray (n_paths, steps) portfolio values.
    """

    weights = normalize_weights(weights)
    n_paths, steps, n_assets = paths.shape

    # start each portfolio with $1 for normalization
    portfolio_vals = np.zeros((n_paths, steps))
    portfolio_vals[:, 0] = 1.0  

    # dollar allocation per asset (n_paths, n_assets)
    # multiply the initial portfolio values by weights
    dollars = portfolio_vals[:, [0]] * weights

    for t in range(1, steps):

        # update dollar values for each asset based on its return from t-1 to t
        dollars = dollars * (paths[:, t, :] / paths[:, t-1, :])

        # rebalance periodically by summing and redistributing based on original weights
        if t % rebalance_step == 0:
            total_value = np.sum(dollars, axis=1)[:, None]
            dollars = weights * total_value

        # get portfolio value at time t by summing dollars across assets
        portfolio_vals[:, t] = np.sum(dollars, axis=1)

    return portfolio_vals



def compute_returns(portfolio_values):
    """
    Compute simple returns for each timestep.
    """
    return portfolio_values[:, 1:] / portfolio_values[:, :-1] - 1


def compute_drawdown(portfolio_values):
    """
    Compute max drawdown for each simulated path.
    """
    running_max = np.maximum.accumulate(portfolio_values, axis=1)
    drawdowns = (portfolio_values - running_max) / running_max
    return drawdowns.min(axis=1)
