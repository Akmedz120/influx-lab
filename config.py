import os
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
CACHE_TTL_DAILY = 24    # hours
CACHE_TTL_INTRADAY = 1  # hours
APP_TITLE = "InFlux Lab"
