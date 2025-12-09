import matplotlib.pyplot as plt
import numpy as np

def plot_price_paths(paths, n_samples=10):
    """
    Plot a few sample GBM price paths.
    """
    plt.figure(figsize=(10, 5))

    for i in range(min(n_samples, paths.shape[0])):
        plt.plot(paths[i], alpha=0.7)

    plt.title("Sample Simulated Price Paths")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def plot_portfolio_paths(portfolio_vals, n_samples=10):
    """
    Plot sample portfolio value paths.
    """
    plt.figure(figsize=(10, 5))

    for i in range(min(n_samples, portfolio_vals.shape[0])):
        plt.plot(portfolio_vals[i], alpha=0.7)

    plt.title("Sample Portfolio Value Paths")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()
