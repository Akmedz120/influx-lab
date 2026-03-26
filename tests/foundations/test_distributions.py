import pytest
import pandas as pd
import numpy as np
from modules.foundations.distributions import fit_normal, normal_pdf_range, normality_test


@pytest.fixture
def normal_returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=0.0, scale=0.01, size=1000))


@pytest.fixture
def skewed_returns():
    np.random.seed(42)
    return pd.Series(np.random.exponential(scale=0.01, size=1000) - 0.01)


def test_fit_normal_returns_two_floats(normal_returns):
    mu, sigma = fit_normal(normal_returns)
    assert isinstance(mu, float)
    assert isinstance(sigma, float)


def test_fit_normal_sigma_positive(normal_returns):
    _, sigma = fit_normal(normal_returns)
    assert sigma > 0


def test_fit_normal_approximates_params(normal_returns):
    mu, sigma = fit_normal(normal_returns)
    assert abs(mu) < 0.005
    assert abs(sigma - 0.01) < 0.002


def test_normal_pdf_range_shape(normal_returns):
    x, y = normal_pdf_range(normal_returns, n_points=50)
    assert len(x) == 50
    assert len(y) == 50


def test_normal_pdf_range_non_negative(normal_returns):
    x, y = normal_pdf_range(normal_returns)
    assert all(y >= 0)


def test_normal_pdf_range_covers_data(normal_returns):
    x, y = normal_pdf_range(normal_returns)
    assert x[0] <= normal_returns.min()
    assert x[-1] >= normal_returns.max()


def test_normality_test_normal_data_passes(normal_returns):
    result = normality_test(normal_returns)
    assert result["is_normal"] is True
    assert "p_value" in result
    assert "statistic" in result


def test_normality_test_skewed_data_fails(skewed_returns):
    result = normality_test(skewed_returns)
    assert result["is_normal"] is False
