import sys
from pathlib import Path

# Make project root importable so `from modules.x import y` works in all tests
sys.path.insert(0, str(Path(__file__).parent))
