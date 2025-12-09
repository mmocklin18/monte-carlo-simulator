import numpy as np
from src.dataloader import fetch_prices
from src.scenarios import apply_scenario
from src.gbm import simulate_gbm
from src.portfolio import rebalance_portfolio, get_portfolio_values


def load_historical_returns(tickers, start, end):
    """
    Fetch historical price data and calculate daily returns.
    """
    prices = fetch_prices(tickers=tickers, start=start, end=end)
    daily_returns = prices.pct_change().dropna()
    return prices, daily_returns

def estimate_mu_sigma(prices):
    """
    Estimate annualized mu and sigma from historical daily prices.
    prices: DataFrame with shape (T, N assets)
    """

    # calculate returns
    daily_rets = prices.pct_change().dropna()
    log_rets = np.log(1 + daily_rets)

    # mean + std of log returns
    mu_daily = log_rets.mean().values
    sigma_daily = log_rets.std().values

    # annualize (assuming 252 trading days)
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    return mu_annual, sigma_annual


def run_simulation(
    mu,
    sigma,
    corr,
    s0,
    steps,
    n_paths,
    dt,
    weights,
    rebalance_step,
    scenario="normal",
):
    """
    Runs a full Monte Carlo simulation:
        Adjust parameters for scenario
        Simulates GBM price paths
        Compute portfolio values with rebalancing
    """

    mu_adj, sigma_adj = apply_scenario(mu, sigma, scenario)

    # run GBM
    paths = simulate_gbm(
        mu=mu_adj,
        sigma=sigma_adj,
        corr=corr,
        s0=s0,
        steps=steps,
        n_paths=n_paths,
        dt=dt,
    )

    # simulate portfolio value paths
    portfolio_vals = rebalance_portfolio(
        paths,
        weights=weights,
        rebalance_step=rebalance_step,
    )

    return {
        "paths": paths,
        "portfolio_vals": portfolio_vals,
        "mu_adj": mu_adj,
        "sigma_adj": sigma_adj,
    }
