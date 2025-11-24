import numpy as np
from src.portfolio import (
    normalize_weights,
    rebalance_portfolio,
    compute_returns,
    compute_drawdown,
    get_portfolio_values
)

def test_portfolio_shape():
    # 5 paths, 10 steps, 3 assets
    paths = np.zeros((5, 10, 3))
    weights = [0.4, 0.3, 0.3]
    pv = get_portfolio_values(paths, weights)
    assert pv.shape == (5, 10)


def test_weights_applied_correctly():
    paths = np.array([
        [[100, 200],
         [110, 210]]
    ])
    weights = [0.5, 0.5]
    pv = get_portfolio_values(paths, weights)

    expected = np.array([[150, 160]])
    assert np.allclose(pv, expected)


def test_zero_weights():
    paths = np.random.rand(3, 5, 4)
    weights = [0, 0, 0, 0]
    pv = get_portfolio_values(paths, weights)
    assert np.allclose(pv, 0)


def test_normalize_weights():
    w = normalize_weights([2, 2, 2])
    assert np.allclose(w, [1/3, 1/3, 1/3])


def test_compute_returns():
    pv = np.array([[100, 110, 121]])
    r = compute_returns(pv)
    assert np.allclose(r, [[0.10, 0.10]])


def test_compute_drawdown():
    pv = np.array([[100, 120, 90, 130]])
    dd = compute_drawdown(pv)
    assert np.allclose(dd, [-0.25])


def test_rebalance_portfolio():
    paths = np.array([
        [[100, 100],
         [120, 100],
         [140, 100],
         [160, 100]]
    ])

    weights = [0.5, 0.5]

    pv_rebalanced = rebalance_portfolio(paths, weights, rebalance_step=1)
    pv_norebalance = get_portfolio_values(paths, weights)

    assert pv_rebalanced[0, -1] < pv_norebalance[0, -1]
