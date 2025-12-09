from src.simulator import run_simulation, load_historical_returns, estimate_mu_sigma
from src.visualize import plot_price_paths, plot_portfolio_paths
from src.portfolio import (
    probability_of_goal,
    value_at_risk,
    conditional_value_at_risk,
    compute_drawdown,
)
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
    tickers=["AMZN","AAPL","JPM","LULU","WBD","JNJ"]

    # Load historical prices for chosen tickers
    prices, daily_returns = load_historical_returns(
        tickers=tickers,
        start="2013-01-01",
        end="2023-12-31",
    )

    print("Prices columns:", prices.columns.tolist())


    mu, sigma = estimate_mu_sigma(prices)
    corr = daily_returns.corr().values

    # starting prices based on last historical price
    s0 = prices.iloc[-1].values

    weights = [0.2, 0.1, 0.1, 0.1, 0.3, 0.2]
    # basic configuration
    steps = 252
    dt = 1/252

    # run the simulation
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

    # example goal = 20% growth over the horizon
    target = 1.20
    prob_goal = probability_of_goal(pv, target=target)

    # example 5% tail for VaR / CVaR
    alpha = 0.05
    var_5 = value_at_risk(pv, level=alpha)
    cvar_5 = conditional_value_at_risk(pv, level=alpha)

    # max drawdown per path
    max_dd_per_path = compute_drawdown(pv)
    avg_max_dd = max_dd_per_path.mean()
    worst_max_dd = max_dd_per_path.min()

    print("\n------ Risk Metrics ------")
    print(f"Target multiple: {target:.2f}x")
    print(f"Probability of reaching target: {prob_goal:.3f}")
    print(f"VaR ({int(alpha*100)}% level): {var_5:.4f}")
    print(f"CVaR ({int(alpha*100)}% level): {cvar_5:.4f}")
    print(f"Average max drawdown: {avg_max_dd:.4f}")
    print(f"Worst max drawdown: {worst_max_dd:.4f}")
    print("==========================\n")


    plot_price_paths(paths=paths, n_samples=100)
    plot_portfolio_paths(portfolio_vals=pv, n_samples=100)

    

if __name__ == "__main__":
    main()
