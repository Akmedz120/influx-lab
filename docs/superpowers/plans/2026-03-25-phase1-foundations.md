# Phase 1: Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit research workspace with a full app skeleton and a fully working Foundations module (returns, distributions, Monte Carlo simulation).

**Architecture:** Streamlit multi-page app. Business logic lives in `modules/` (pure Python, fully testable). Streamlit pages in `pages/` are thin UI wrappers that call module functions. Data fetching is centralized in `modules/data/` with pickle-based caching.

**Tech Stack:** Python 3.11+, Streamlit, yfinance, fredapi, pandas, numpy, scipy, plotly, python-dotenv, pytest

---

## File Map

```
Financial-Models/
├── Home.py                            ← Landing page (text overview, no data in Phase 1)
├── config.py                          ← App settings, loads env vars
├── conftest.py                        ← pytest path setup (makes modules/ importable)
├── requirements.txt
├── .env                               ← FRED_API_KEY (gitignored)
├── .gitignore
├── pages/
│   ├── 1_Foundations.py               ← Foundations UI (returns, distributions, Monte Carlo)
│   ├── 2_Market_Pulse.py              ← Stub
│   ├── 3_Integration.py               ← Stub
│   ├── 4_Features.py                  ← Stub
│   ├── 5_ML_AI.py                     ← Stub
│   ├── 6_Sectors.py                   ← Stub
│   └── 7_Sandbox.py                   ← Stub
├── modules/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cache.py                   ← Pickle-based local cache
│   │   └── fetcher.py                 ← yfinance, FRED, CSV loaders
│   └── foundations/
│       ├── __init__.py
│       ├── returns.py                 ← Return calculations and stats
│       ├── distributions.py           ← Distribution fitting and analysis
│       └── monte_carlo.py             ← GBM simulation
├── data/
│   └── cache/                         ← Auto-created, gitignored
└── tests/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── test_cache.py
    │   └── test_fetcher.py
    └── foundations/
        ├── __init__.py
        ├── test_returns.py
        ├── test_distributions.py
        └── test_monte_carlo.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env` (template only, no real keys)
- Create: `config.py`
- Create: `modules/__init__.py`, `modules/data/__init__.py`, `modules/foundations/__init__.py`
- Create: `tests/__init__.py`, `tests/data/__init__.py`, `tests/foundations/__init__.py`
- Create: `data/cache/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.32.0
yfinance>=0.2.36
fredapi>=0.5.1
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
plotly>=5.19.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create .gitignore**

```
.env
data/cache/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.superpowers/
```

Note: `docs/superpowers/` is intentionally NOT gitignored — the spec and plan files there should be tracked in version control.

- [ ] **Step 3: Create .env (template)**

```
FRED_API_KEY=your_key_here
```

- [ ] **Step 4: Create config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
CACHE_TTL_DAILY = 24    # hours
CACHE_TTL_INTRADAY = 1  # hours
APP_TITLE = "Financial Research Workspace"
```

- [ ] **Step 5: Create conftest.py at project root**

```python
import sys
from pathlib import Path

# Make project root importable so `from modules.x import y` works in all tests
sys.path.insert(0, str(Path(__file__).parent))
```

- [ ] **Step 6: Create all empty __init__.py files**

Create empty files at:
- `modules/__init__.py`
- `modules/data/__init__.py`
- `modules/foundations/__init__.py`
- `tests/__init__.py`
- `tests/data/__init__.py`
- `tests/foundations/__init__.py`
- `data/cache/.gitkeep`

- [ ] **Step 7: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 8: Verify pytest can find modules**

```bash
pytest --collect-only
```

Expected: test files are discovered with no `ModuleNotFoundError`.

- [ ] **Step 9: Commit**

```bash
git init
git add requirements.txt .gitignore config.py modules/ tests/ data/ conftest.py
git commit -m "feat: project setup — dependencies, config, structure"
```

Note: `.env` is intentionally excluded from git add — it contains secrets.

---

## Task 2: Cache Module

**Files:**
- Create: `modules/data/cache.py`
- Create: `tests/data/test_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/data/test_cache.py`:

```python
import time
import pytest
from pathlib import Path
from modules.data.cache import get_cached, set_cached


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.data.cache.CACHE_DIR", tmp_path)


def test_cache_miss_returns_none():
    assert get_cached("nonexistent") is None


def test_cache_stores_and_retrieves():
    set_cached("key1", {"value": 42})
    assert get_cached("key1") == {"value": 42}


def test_expired_cache_returns_none():
    set_cached("key2", "data", ttl_hours=0.0001)  # ~0.36 seconds
    time.sleep(0.5)
    assert get_cached("key2") is None


def test_valid_cache_not_expired():
    set_cached("key3", [1, 2, 3], ttl_hours=24)
    assert get_cached("key3") == [1, 2, 3]


def test_cache_key_with_slashes():
    set_cached("prices/SPY/2024", "data")
    assert get_cached("prices/SPY/2024") == "data"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/data/test_cache.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — cache.py doesn't exist yet.

- [ ] **Step 3: Implement cache.py**

Create `modules/data/cache.py`:

```python
import pickle
import time
from pathlib import Path

CACHE_DIR = Path("data/cache")


def _cache_path(key: str) -> Path:
    safe_key = key.replace("/", "_").replace(" ", "_").replace(":", "_")
    return CACHE_DIR / f"{safe_key}.pkl"


def get_cached(key: str, ttl_hours: float = 24):
    """Return cached data if it exists and hasn't expired. Otherwise return None."""
    path = _cache_path(key)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        entry = pickle.load(f)
    if time.time() - entry["timestamp"] > entry["ttl_hours"] * 3600:
        path.unlink()
        return None
    return entry["data"]


def set_cached(key: str, data, ttl_hours: float = 24) -> None:
    """Store data in local cache with TTL."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    with open(path, "wb") as f:
        pickle.dump({"data": data, "timestamp": time.time(), "ttl_hours": ttl_hours}, f)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/data/test_cache.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/data/cache.py tests/data/test_cache.py
git commit -m "feat: add pickle-based local cache with TTL"
```

---

## Task 3: Price Data Fetcher (yfinance)

**Files:**
- Create: `modules/data/fetcher.py`
- Create: `tests/data/test_fetcher.py`

- [ ] **Step 1: Write failing tests**

Create `tests/data/test_fetcher.py`:

```python
import pytest
import pandas as pd
import numpy as np
import io
from unittest.mock import patch
from modules.data.fetcher import fetch_prices, load_csv


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.data.cache.CACHE_DIR", tmp_path)


def make_price_df(tickers, n=5):
    idx = pd.date_range("2024-01-01", periods=n)
    data = {t: np.linspace(100, 110, n) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_fetch_prices_returns_dataframe_with_correct_columns():
    mock_df = make_price_df(["AAPL"])
    with patch("modules.data.fetcher._yf_download", return_value=mock_df):
        result = fetch_prices(["AAPL"], "2024-01-01", "2024-12-31")
    assert isinstance(result, pd.DataFrame)
    assert "AAPL" in result.columns
    assert len(result) == 5


def test_fetch_prices_multi_ticker_returns_all_columns():
    mock_df = make_price_df(["SPY", "QQQ"])
    with patch("modules.data.fetcher._yf_download", return_value=mock_df):
        result = fetch_prices(["SPY", "QQQ"], "2024-01-01", "2024-12-31")
    assert "SPY" in result.columns
    assert "QQQ" in result.columns


def test_fetch_prices_uses_cache_on_second_call():
    mock_df = make_price_df(["QQQ"])
    with patch("modules.data.fetcher._yf_download", return_value=mock_df) as mock_fn:
        fetch_prices(["QQQ"], "2024-01-01", "2024-06-01")
        fetch_prices(["QQQ"], "2024-01-01", "2024-06-01")
    assert mock_fn.call_count == 1


def test_load_csv_returns_dataframe():
    csv_content = "date,price\n2024-01-01,100\n2024-01-02,101\n"
    file = io.StringIO(csv_content)
    result = load_csv(file)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "price" in result.columns


def test_load_csv_empty_raises():
    file = io.StringIO("")
    with pytest.raises(Exception):
        load_csv(file)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/data/test_fetcher.py -v
```

Expected: `ImportError` — fetcher.py doesn't exist yet.

- [ ] **Step 3: Implement fetcher.py**

Create `modules/data/fetcher.py`:

```python
import pandas as pd
import yfinance as yf
from modules.data.cache import get_cached, set_cached
from config import FRED_API_KEY, CACHE_TTL_DAILY


def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around yfinance.download — isolated for mocking in tests."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        # Single ticker returns flat DataFrame
        data = data[["Close"]].rename(columns={"Close": tickers[0]})
    return data


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for one or more tickers.
    Returns DataFrame with ticker columns and DatetimeIndex.
    Results are cached locally for 24 hours.
    """
    key = f"prices_{'_'.join(sorted(tickers))}_{start}_{end}"
    cached = get_cached(key, ttl_hours=CACHE_TTL_DAILY)
    if cached is not None:
        return cached
    data = _yf_download(tickers, start, end)
    set_cached(key, data, ttl_hours=CACHE_TTL_DAILY)
    return data


def load_csv(file) -> pd.DataFrame:
    """
    Load a CSV file (file-like object or path) into a DataFrame.
    Raises ValueError if the file is empty or unparseable.
    """
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("CSV file is empty")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or has no columns")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/data/test_fetcher.py -v
```

Expected: all passing tests PASS. (Skip any test that requires live network — those are integration tests.)

- [ ] **Step 5: Commit**

```bash
git add modules/data/fetcher.py tests/data/test_fetcher.py
git commit -m "feat: add yfinance price fetcher and CSV loader with caching"
```

---

## Task 4: FRED Data Fetcher

**Files:**
- Modify: `modules/data/fetcher.py`
- Modify: `tests/data/test_fetcher.py`

- [ ] **Step 1: Add failing test for fetch_fred**

Append to `tests/data/test_fetcher.py`:

```python
from modules.data.fetcher import fetch_fred


def test_fetch_fred_returns_series():
    mock_series = pd.Series([5.25, 5.25, 5.0], index=pd.date_range("2024-01-01", periods=3))
    with patch("modules.data.fetcher._fred_get_series", return_value=mock_series):
        result = fetch_fred("FEDFUNDS", "2024-01-01", "2024-03-01")
    assert isinstance(result, pd.Series)
    assert len(result) == 3


def test_fetch_fred_uses_cache():
    mock_series = pd.Series([5.25], index=pd.date_range("2024-01-01", periods=1))
    with patch("modules.data.fetcher._fred_get_series", return_value=mock_series) as mock_fn:
        fetch_fred("DGS10", "2024-01-01", "2024-01-01")
        fetch_fred("DGS10", "2024-01-01", "2024-01-01")
    assert mock_fn.call_count == 1
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/data/test_fetcher.py::test_fetch_fred_returns_series -v
```

Expected: `ImportError` — fetch_fred doesn't exist yet.

- [ ] **Step 3: Add fetch_fred to fetcher.py**

Append to `modules/data/fetcher.py`:

```python
def _fred_get_series(series_id: str, start: str, end: str) -> pd.Series:
    """Thin wrapper around fredapi — isolated for mocking in tests."""
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    return fred.get_series(series_id, observation_start=start, observation_end=end)


def fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """
    Fetch a FRED economic data series by series ID.
    Returns a pd.Series with DatetimeIndex.
    Results cached for 24 hours.

    Common series IDs:
      FEDFUNDS  — Federal funds rate
      DGS2      — 2-year Treasury yield
      DGS10     — 10-year Treasury yield
      BAMLH0A0HYM2 — High yield credit spread
      M2SL      — M2 money supply
      UMCSENT   — University of Michigan consumer sentiment
      AAIIBULL  — AAII bullish sentiment
    """
    key = f"fred_{series_id}_{start}_{end}"
    cached = get_cached(key, ttl_hours=CACHE_TTL_DAILY)
    if cached is not None:
        return cached
    data = _fred_get_series(series_id, start, end)
    set_cached(key, data, ttl_hours=CACHE_TTL_DAILY)
    return data
```

- [ ] **Step 4: Run all fetcher tests**

```bash
pytest tests/data/test_fetcher.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/data/fetcher.py tests/data/test_fetcher.py
git commit -m "feat: add FRED economic data fetcher with caching"
```

---

## Task 5: Returns Module

**Files:**
- Create: `modules/foundations/returns.py`
- Create: `tests/foundations/test_returns.py`

- [ ] **Step 1: Write failing tests**

Create `tests/foundations/test_returns.py`:

```python
import pytest
import pandas as pd
import numpy as np
from modules.foundations.returns import calculate_returns, annualize_return, annualize_volatility, compute_stats


@pytest.fixture
def prices():
    return pd.Series([100.0, 102.0, 101.0, 103.0, 105.0], name="SPY")


@pytest.fixture
def returns(prices):
    return calculate_returns(prices)


def test_returns_length(prices):
    r = calculate_returns(prices)
    assert len(r) == len(prices) - 1


def test_returns_first_value(prices):
    r = calculate_returns(prices)
    assert abs(r.iloc[0] - 0.02) < 1e-10


def test_returns_no_nans(prices):
    r = calculate_returns(prices)
    assert not r.isna().any()


def test_annualize_return_is_float(returns):
    result = annualize_return(returns)
    assert isinstance(result, float)


def test_annualize_volatility_non_negative(returns):
    result = annualize_volatility(returns)
    assert result >= 0


def test_annualize_volatility_weekly(returns):
    daily_vol = annualize_volatility(returns, frequency="daily")
    weekly_vol = annualize_volatility(returns, frequency="weekly")
    # Same returns but different frequency assumption — should give different values
    assert daily_vol != weekly_vol


def test_compute_stats_has_required_keys(returns):
    stats = compute_stats(returns)
    for key in ["mean_return", "volatility", "skewness", "kurtosis", "sharpe"]:
        assert key in stats


def test_compute_stats_sharpe_is_ratio(returns):
    stats = compute_stats(returns)
    # Sharpe = mean_return / volatility (simplified, no risk-free rate)
    expected = stats["mean_return"] / stats["volatility"] if stats["volatility"] != 0 else 0
    assert abs(stats["sharpe"] - expected) < 1e-10


def test_flat_prices_zero_volatility():
    flat = pd.Series([100.0, 100.0, 100.0, 100.0])
    r = calculate_returns(flat)
    assert annualize_volatility(r) == 0.0


def test_weekly_returns_fewer_than_daily():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.Series(range(100, 130), index=idx, dtype=float)
    daily = calculate_returns(prices, frequency="daily")
    weekly = calculate_returns(prices, frequency="weekly")
    assert len(weekly) < len(daily)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/foundations/test_returns.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement returns.py**

Create `modules/foundations/returns.py`:

```python
import numpy as np
import pandas as pd

FREQ_PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}


RESAMPLE_RULES = {"daily": None, "weekly": "W", "monthly": "ME"}


def calculate_returns(prices: pd.Series, frequency: str = "daily") -> pd.Series:
    """
    Calculate simple percentage returns from a price series.
    frequency: 'daily' (no resampling), 'weekly', or 'monthly'.
    Resamples prices to the target frequency before computing returns.
    """
    rule = RESAMPLE_RULES.get(frequency)
    if rule:
        prices = prices.resample(rule).last().dropna()
    return prices.pct_change().dropna()


def annualize_return(returns: pd.Series, frequency: str = "daily") -> float:
    """Compound annualized return from a returns series."""
    n = FREQ_PERIODS[frequency]
    return float((1 + returns.mean()) ** n - 1)


def annualize_volatility(returns: pd.Series, frequency: str = "daily") -> float:
    """Annualized volatility (standard deviation of returns)."""
    n = FREQ_PERIODS[frequency]
    return float(returns.std() * np.sqrt(n))


def compute_stats(returns: pd.Series, frequency: str = "daily") -> dict:
    """
    Compute key return statistics.
    Returns dict with: mean_return, volatility, skewness, kurtosis, sharpe.
    Sharpe uses no risk-free rate adjustment (simplified).
    """
    vol = annualize_volatility(returns, frequency)
    ret = annualize_return(returns, frequency)
    return {
        "mean_return": ret,
        "volatility": vol,
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),  # excess kurtosis (normal = 0)
        "sharpe": ret / vol if vol != 0 else 0.0,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/foundations/test_returns.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/foundations/returns.py tests/foundations/test_returns.py
git commit -m "feat: add returns module — pct change, annualized stats"
```

---

## Task 6: Distributions Module

**Files:**
- Create: `modules/foundations/distributions.py`
- Create: `tests/foundations/test_distributions.py`

- [ ] **Step 1: Write failing tests**

Create `tests/foundations/test_distributions.py`:

```python
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
    assert abs(mu) < 0.005       # close to true mean of 0
    assert abs(sigma - 0.01) < 0.002  # close to true std of 0.01


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/foundations/test_distributions.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement distributions.py**

Create `modules/foundations/distributions.py`:

```python
import numpy as np
import pandas as pd
from scipy import stats


def fit_normal(returns: pd.Series) -> tuple[float, float]:
    """Fit a normal distribution to the returns. Returns (mu, sigma)."""
    mu, sigma = stats.norm.fit(returns)
    return float(mu), float(sigma)


def normal_pdf_range(returns: pd.Series, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate x/y values for a normal distribution fitted to returns.
    Use to overlay a normal curve on a histogram for comparison.
    """
    mu, sigma = fit_normal(returns)
    x = np.linspace(returns.min(), returns.max(), n_points)
    y = stats.norm.pdf(x, mu, sigma)
    return x, y


def normality_test(returns: pd.Series) -> dict:
    """
    Shapiro-Wilk test for normality.
    Returns dict with statistic, p_value, is_normal.
    p_value > 0.05 = cannot reject normality.
    Samples up to 5000 observations (Shapiro-Wilk limit).
    """
    sample = returns.sample(min(len(returns), 5000), random_state=42)
    stat, p_value = stats.shapiro(sample)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": p_value > 0.05,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/foundations/test_distributions.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/foundations/distributions.py tests/foundations/test_distributions.py
git commit -m "feat: add distributions module — normal fit, PDF range, normality test"
```

---

## Task 7: Monte Carlo Module

**Files:**
- Create: `modules/foundations/monte_carlo.py`
- Create: `tests/foundations/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

Create `tests/foundations/test_monte_carlo.py`:

```python
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
    # With sigma=0 and positive drift, all paths should be identical regardless of seed
    p = gbm_simulation(S0=100.0, mu=0.001, sigma=0.0, T=10, n_simulations=50, seed=42)
    assert np.allclose(p[:, 0], p[:, 1])
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/foundations/test_monte_carlo.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement monte_carlo.py**

Create `modules/foundations/monte_carlo.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/foundations/test_monte_carlo.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests across all modules PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/foundations/monte_carlo.py tests/foundations/test_monte_carlo.py
git commit -m "feat: add Monte Carlo module — GBM simulation with stats"
```

---

## Task 8: App Skeleton

**Files:**
- Create: `Home.py`
- Create: `pages/2_Market_Pulse.py` through `pages/7_Sandbox.py`

No tests for UI layer.

- [ ] **Step 1: Create Home.py**

```python
import streamlit as st

st.set_page_config(page_title="Financial Research Workspace", layout="wide")

st.title("Financial Research Workspace")
st.caption("A local research environment for financial modeling, signal development, and market analysis.")

st.divider()

st.markdown("""
This workspace is organized into focused modules. Each module builds on the last.
Use the sidebar to navigate.

| Module | Purpose | Status |
|--------|---------|--------|
| **Foundations** | Returns, distributions, Monte Carlo simulation | ✅ Active |
| **Market Pulse** | Sentiment, fear/greed, risk appetite, macro stress | 🔜 Coming |
| **Integration** | Cross-asset correlations, macro regime analysis | 🔜 Coming |
| **Features** | Signal generation, derived metrics | 🔜 Coming |
| **ML / AI** | Pattern detection, predictive models on structured features | 🔜 Coming |
| **Sectors** | Sector rotation, relative strength, money flow | 🔜 Coming |
| **Sandbox** | Free experimentation | 🔜 Coming |

---
**Start in Foundations** — it's the base everything else is built on.
""")
```

- [ ] **Step 2: Create stub pages**

Create each of the following with the same pattern:

`pages/2_Market_Pulse.py`:
```python
import streamlit as st
st.set_page_config(page_title="Market Pulse", layout="wide")
st.title("Market Pulse")
st.caption("Sentiment, fear/greed index, risk appetite, macro stress, and international market conditions.")
st.info("This module is not yet built. Check back after Foundations is complete.")
```

`pages/3_Integration.py`:
```python
import streamlit as st
st.set_page_config(page_title="Integration", layout="wide")
st.title("Integration")
st.caption("Cross-asset correlations, macro regime detection, and lead-lag relationships between markets.")
st.info("This module is not yet built.")
```

`pages/4_Features.py`:
```python
import streamlit as st
st.set_page_config(page_title="Features", layout="wide")
st.title("Features")
st.caption("Generate structured signals: volatility regimes, momentum states, macro indicators, correlation clusters.")
st.info("This module is not yet built.")
```

`pages/5_ML_AI.py`:
```python
import streamlit as st
st.set_page_config(page_title="ML / AI", layout="wide")
st.title("ML / AI")
st.caption("Machine learning experiments on structured features — pattern detection, regime classification, predictive modeling.")
st.info("This module is not yet built.")
```

`pages/6_Sectors.py`:
```python
import streamlit as st
st.set_page_config(page_title="Sectors", layout="wide")
st.title("Sectors")
st.caption("Sector rotation analysis, relative strength, and money flow between S&P 500 sectors.")
st.info("This module is not yet built.")
```

`pages/7_Sandbox.py`:
```python
import streamlit as st
st.set_page_config(page_title="Sandbox", layout="wide")
st.title("Sandbox")
st.caption("Free experimentation — unconventional ideas, exploratory models, anything goes.")
st.info("This module is not yet built.")
```

- [ ] **Step 3: Verify app runs**

```bash
streamlit run Home.py
```

Expected: app opens at http://localhost:8501, sidebar shows all 7 pages, each stub page loads without errors.

- [ ] **Step 4: Commit**

```bash
git add Home.py pages/
git commit -m "feat: add app skeleton — home page and stub modules"
```

---

## Task 9: Foundations Page UI

**Files:**
- Create: `pages/1_Foundations.py`

No unit tests — UI layer only calls tested module functions.

- [ ] **Step 1: Create pages/1_Foundations.py**

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from modules.data.fetcher import fetch_prices
from modules.foundations.returns import calculate_returns, compute_stats
from modules.foundations.distributions import fit_normal, normal_pdf_range, normality_test
from modules.foundations.monte_carlo import gbm_simulation, simulation_stats

st.set_page_config(page_title="Foundations", layout="wide")
st.title("Foundations")
st.caption("Understand return distributions and uncertainty — the base everything else builds on.")

# ─── Section 1: Returns & Distributions ──────────────────────────────────────
st.header("Returns & Distributions")

col1, col2 = st.columns([3, 1])
with col1:
    tickers_input = st.text_input("Tickers (comma separated)", value="SPY, QQQ")
with col2:
    period = st.selectbox("Lookback Period", ["1Y", "2Y", "5Y", "10Y"], index=1)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
period_days = {"1Y": 365, "2Y": 730, "5Y": 1825, "10Y": 3650}
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=period_days[period])).strftime("%Y-%m-%d")

if tickers:
    with st.spinner("Loading data..."):
        try:
            prices = fetch_prices(tickers, start_date, end_date)
            returns_dict = {t: calculate_returns(prices[t]) for t in tickers if t in prices.columns}
            stats_dict = {t: compute_stats(r) for t, r in returns_dict.items()}

            # Distribution chart
            fig = go.Figure()
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
            for i, ticker in enumerate(returns_dict):
                r = returns_dict[ticker]
                color = colors[i % len(colors)]
                fig.add_trace(go.Histogram(
                    x=r, name=ticker, opacity=0.55,
                    histnorm="probability density", nbinsx=80,
                    marker_color=color
                ))
                x_norm, y_norm = normal_pdf_range(r)
                fig.add_trace(go.Scatter(
                    x=x_norm, y=y_norm,
                    name=f"{ticker} — normal fit",
                    line=dict(color=color, dash="dash", width=2)
                ))
            fig.update_layout(
                title="Return Distributions vs Normal",
                xaxis_title="Daily Return",
                yaxis_title="Probability Density",
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats table
            stats_df = pd.DataFrame(stats_dict).T
            stats_df.index.name = "Ticker"
            display_df = pd.DataFrame({
                "Ann. Return": stats_df["mean_return"].map("{:.1%}".format),
                "Ann. Volatility": stats_df["volatility"].map("{:.1%}".format),
                "Skewness": stats_df["skewness"].map("{:.2f}".format),
                "Excess Kurtosis": stats_df["kurtosis"].map("{:.2f}".format),
                "Sharpe (no Rf)": stats_df["sharpe"].map("{:.2f}".format),
            })
            st.dataframe(display_df, use_container_width=True)

            # So what — one insight per ticker
            for ticker, s in stats_dict.items():
                nt = normality_test(returns_dict[ticker])
                skew_desc = (
                    "negatively skewed — losses tend to be larger than gains"
                    if s["skewness"] < -0.3
                    else "positively skewed — gains tend to be larger than losses"
                    if s["skewness"] > 0.3
                    else "roughly symmetric"
                )
                kurt_desc = (
                    "fat tails — extreme moves happen more often than a normal model would predict"
                    if s["kurtosis"] > 1
                    else "thin tails — fewer extreme moves than a normal model"
                    if s["kurtosis"] < -0.5
                    else "near-normal tail behavior"
                )
                normal_note = "" if nt["is_normal"] else " The returns are **not** normally distributed (Shapiro-Wilk p < 0.05) — standard risk models may understate actual risk."
                st.info(
                    f"**{ticker}**: Returns are {skew_desc}, with {kurt_desc}. "
                    f"Annualized volatility: {s['volatility']:.1%}, Sharpe: {s['sharpe']:.2f}.{normal_note}"
                )

        except Exception as e:
            st.error(f"Could not load data: {e}")

# ─── Section 2: Monte Carlo ───────────────────────────────────────────────────
st.divider()
st.header("Monte Carlo Simulation")
st.caption(
    "Simulates 1,000 possible future price paths using Geometric Brownian Motion (GBM). "
    "GBM assumes log-normal returns — it **underestimates tail risk**. "
    "This is a range of outcomes, not a forecast."
)

mc_col1, mc_col2, mc_col3 = st.columns(3)
with mc_col1:
    mc_ticker = st.text_input("Ticker", value="SPY", key="mc_ticker")
with mc_col2:
    horizon_label = st.selectbox("Horizon", ["30 days", "90 days", "1 year"], index=1)
with mc_col3:
    n_sims = st.selectbox("Paths", [500, 1000, 2000], index=1)

horizon_map = {"30 days": 30, "90 days": 90, "1 year": 252}
T = horizon_map[horizon_label]

if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        try:
            mc_prices = fetch_prices([mc_ticker], start_date, end_date)
            mc_col = mc_ticker if mc_ticker in mc_prices.columns else mc_prices.columns[0]
            mc_rets = calculate_returns(mc_prices[mc_col])
            mu_d = float(mc_rets.mean())
            sigma_d = float(mc_rets.std())
            S0 = float(mc_prices[mc_col].iloc[-1])

            paths = gbm_simulation(S0=S0, mu=mu_d, sigma=sigma_d, T=T, n_simulations=n_sims)
            mc_stats = simulation_stats(paths)

            # Fan chart
            # Note: plotting individual path traces is visually clear but gets slow above
            # ~150 traces. We subsample using `step` to cap rendered paths at ~100.
            # If this feels slow, a future improvement is switching to a filled area
            # (percentile band) chart instead of individual path lines.
            fig2 = go.Figure()
            x_axis = list(range(T + 1))
            step = max(1, n_sims // 100)  # cap at ~100 visible paths
            for i in range(0, n_sims, step):
                fig2.add_trace(go.Scatter(
                    y=paths[:, i], x=x_axis, mode="lines",
                    line=dict(color="rgba(100,149,237,0.15)", width=1),
                    showlegend=False, hoverinfo="skip"
                ))
            # Percentile lines
            fig2.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), x=x_axis, name="5th pct (bad)", line=dict(color="#EF553B", width=2.5)))
            fig2.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), x=x_axis, name="Median", line=dict(color="#00CC96", width=2.5)))
            fig2.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), x=x_axis, name="95th pct (good)", line=dict(color="#636EFA", width=2.5)))

            fig2.update_layout(
                title=f"{mc_ticker} — {n_sims} simulated paths over {horizon_label}",
                xaxis_title="Trading Days",
                yaxis_title="Price ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Starting Price", f"${S0:.2f}")
            m2.metric("Median Outcome", f"${mc_stats['median']:.2f}", delta=f"{(mc_stats['median']/S0 - 1):.1%}")
            m3.metric("Worst 5% Scenario", f"${mc_stats['p5']:.2f}", delta=f"{(mc_stats['p5']/S0 - 1):.1%}", delta_color="inverse")
            m4.metric("Probability of Loss", f"{mc_stats['prob_loss']:.1%}")

            st.info(
                f"**So what:** Starting at ${S0:.2f}, the median path ends at ${mc_stats['median']:.2f} after {horizon_label}. "
                f"In the worst 5% of scenarios, the price falls to ${mc_stats['p5']:.2f} ({(mc_stats['p5']/S0 - 1):.1%}). "
                f"There is a {mc_stats['prob_loss']:.1%} probability of loss based on historical return patterns. "
                f"The wide spread in paths reflects genuine uncertainty — not imprecision in the model."
            )

        except Exception as e:
            st.error(f"Could not run simulation: {e}")
```

- [ ] **Step 2: Run the app and verify Foundations page works end to end**

```bash
streamlit run Home.py
```

Navigate to Foundations. Test:
1. Enter "SPY, QQQ" — distribution chart loads, stats table appears, "so what" messages show
2. Enter a single ticker — works
3. Enter an invalid ticker — shows error gracefully
4. Run Monte Carlo on "SPY" with 90-day horizon — fan chart renders, metrics show

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Final commit**

```bash
git add pages/1_Foundations.py
git commit -m "feat: Foundations page UI — returns, distributions, Monte Carlo"
```

---

## Done: Phase 1 Complete

At this point:
- ✅ App skeleton running with 7 modules (1 active, 6 stubbed)
- ✅ Data layer: yfinance + FRED + CSV upload with caching
- ✅ Foundations module: returns, distributions, Monte Carlo
- ✅ All business logic covered by tests
- ✅ "So what" explanations on every section

**Next plan:** `2026-03-25-phase2a-market-pulse-core.md` — VIX, yield curve, credit spreads, sentiment, DXY, global heatmap.
