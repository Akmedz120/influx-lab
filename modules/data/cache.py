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
