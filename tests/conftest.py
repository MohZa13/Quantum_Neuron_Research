"""Make tests/ importable (notebook_test_utils) regardless of invocation dir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
