import pytest
import numpy as np
from modules.foundations.monte_carlo import gbm_simulation, simulation_stats


@pytest.fixture
def paths():
    return gbm_simulation(S0=100.0, mu=0.0003, sigma=0.01, T=30, n_simulations=500, seed=42)


def test_gbm_output_shape(paths):
    assert paths.shape == (31, 500)  # T+1 rows (includes day 0), n_simulations cols


def test_gbm_starts_at_S0():
    p = gbm_simulation(S0=150.0, mu=0.0, sigma=0.01, T=10, n_simulations=100, seed=42)
    assert np.all(p[0] == 150.0)


def test_gbm_all_prices_positive(paths):
    assert np.all(paths > 0)


def test_gbm_reproducible_with_seed():
    p1 = gbm_simulation(S0=100.0, mu=0.0003, sigma=0.01, T=30, n_simulations=100, seed=42)
    p2 = gbm_simulation(S0=100.0, mu=0.0003, sigma=0.01, T=30, n_simulations=100, seed=42)
    assert np.allclose(p1, p2)


def test_gbm_different_seeds_differ():
    p1 = gbm_simulation(S0=100.0, mu=0.0003, sigma=0.01, T=30, n_simulations=100, seed=1)
    p2 = gbm_simulation(S0=100.0, mu=0.0003, sigma=0.01, T=30, n_simulations=100, seed=2)
    assert not np.allclose(p1, p2)


def test_simulation_stats_keys(paths):
    stats = simulation_stats(paths)
    for key in ["median", "p5", "p95", "prob_loss", "mean"]:
        assert key in stats


def test_simulation_stats_p5_less_than_p95(paths):
    stats = simulation_stats(paths)
    assert stats["p5"] < stats["p95"]


def test_simulation_stats_prob_loss_in_range(paths):
    stats = simulation_stats(paths)
    assert 0.0 <= stats["prob_loss"] <= 1.0


def test_zero_volatility_paths_converge():
    p = gbm_simulation(S0=100.0, mu=0.001, sigma=0.0, T=10, n_simulations=50, seed=42)
    assert np.allclose(p[:, 0], p[:, 1])
