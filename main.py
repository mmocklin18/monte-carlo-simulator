from src.simulator import run_simulation, load_historical_returns
from src.simulator import estimate_mu_sigma
from src.visualize import plot_price_paths, plot_portfolio_paths
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--rebalance", type=int, default=21)
    return parser.parse_args()


def main():
    args = get_args()
    tickers=["SPY","NVDA","AAPL","AVGO"]

    # Load historical prices for chosen tickers
    prices, daily_returns = load_historical_returns(
        tickers=tickers,
        start="2013-01-01",
        end="2023-12-31",
    )

    print("Prices columns:", prices.columns.tolist())


    mu, sigma = estimate_mu_sigma(prices)

    corr = daily_returns.corr().values

    # 3. Starting prices based on last historical price
    s0 = prices.iloc[-1].values

    # 4. Equal weights by default (or set your own)
    weights = np.ones(len(tickers)) / len(tickers)

    # Basic MC configuration
    steps = 252
    dt = 1/252

    # Run the simulation
    result = run_simulation(
        mu=mu,
        sigma=sigma,
        corr=corr,
        s0=s0,
        steps=steps,
        n_paths=args.paths,
        dt=dt,
        weights=weights,
        rebalance_step=args.rebalance,
        scenario=args.scenario,
    )

    pv = result["portfolio_vals"]
    paths = result["paths"]

    print("\n====== Simulation Results ======")
    print(f"Tickers: {tickers}")
    print(f"Scenario: {args.scenario}")
    print(f"Estimated mu: {mu}")
    print(f"Estimated sigma: {sigma}")
    print(f"Correlation matrix:\n{corr}")
    print(f"Final portfolio values (first 5): {pv[:5, -1]}")
    print(f"Mean final value: {pv[:, -1].mean():.4f}")
    print("==========================\n")

    plot_price_paths(paths=paths, n_samples=100)
    plot_portfolio_paths(portfolio_vals=pv, n_samples=100)


if __name__ == "__main__":
    main()
