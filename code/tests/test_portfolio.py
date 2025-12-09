import numpy as np
from src.portfolio import (
    normalize_weights,
    rebalance_portfolio,
    compute_returns,
    compute_drawdown,
    get_portfolio_values,
    probability_of_goal,
    value_at_risk,
    conditional_value_at_risk
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

def test_probability_of_goal():
    pv = np.array([
        [100, 130], # hits goal
        [100, 110], # hits goal
        [100, 90],  # does not hit goal
    ])
    prob = probability_of_goal(pv, target=100)
    assert np.isclose(prob, 2/3)


def test_value_at_risk():
    pv = np.array([
        [1, 0.80], # -20%
        [1, 0.90], # -10%
        [1, 1.00], # 0%
        [1, 1.10], # +10%
        [1, 1.20], # +20%
    ])

    var = value_at_risk(pv, level=0.20)
    assert np.isclose(var, -0.12)


def test_conditional_value_at_risk():
    pv = np.array([
        [1, 0.80], # -20%
        [1, 0.90], # -10%
        [1, 1.00], # 0%
        [1, 1.10], # +10%
        [1, 1.20], # +20%
    ])

    cvar = conditional_value_at_risk(pv, level=0.20)
    assert np.isclose(cvar, -0.20)
