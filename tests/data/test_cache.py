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
