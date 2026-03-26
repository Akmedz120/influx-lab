import numpy as np


def gbm_simulation(
    S0: float,
    mu: float,
    sigma: float,
    T: int,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion (GBM).

    Parameters:
        S0           : Starting price
        mu           : Daily drift (use returns.mean())
        sigma        : Daily volatility (use returns.std())
        T            : Number of trading days to simulate
        n_simulations: Number of paths
        seed         : Random seed for reproducibility. Pass None (default) for
                       fresh randomness each call (use in UI). Pass an int for
                       reproducible output (use in tests).

    Returns:
        Array of shape (T+1, n_simulations) — price paths including day 0.

    Note: GBM assumes log-normal returns. It underestimates tail risk.
    Treat output as a range of outcomes, not a forecast.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0  # 1 trading day per step
    drift = (mu - 0.5 * sigma ** 2) * dt
    shocks = rng.normal(0, 1, (T, n_simulations))
    daily_log_returns = drift + sigma * np.sqrt(dt) * shocks
    paths = np.zeros((T + 1, n_simulations))
    paths[0] = S0
    for t in range(1, T + 1):
        paths[t] = paths[t - 1] * np.exp(daily_log_returns[t - 1])
    return paths


def simulation_stats(paths: np.ndarray) -> dict:
    """
    Summarize simulation results.

    Returns:
        median    : Median final price
        p5        : 5th percentile final price (bad scenario)
        p95       : 95th percentile final price (good scenario)
        prob_loss : Fraction of paths ending below starting price
        mean      : Mean final price
    """
    final = paths[-1]
    S0 = paths[0][0]
    return {
        "median": float(np.median(final)),
        "p5": float(np.percentile(final, 5)),
        "p95": float(np.percentile(final, 95)),
        "prob_loss": float(np.mean(final < S0)),
        "mean": float(np.mean(final)),
    }
