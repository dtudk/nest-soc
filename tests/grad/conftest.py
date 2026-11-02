"""
Shared setup of the ``nest.grad`` tests.

Puts this directory on ``sys.path`` so the test files import the
gradient-check cases (``cases_*.py``) as ordinary modules, loaded once, and
registers the ``slow`` marker (``uv run pytest tests/grad -m "not slow"``
skips the expensive cell checks).
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: expensive cell gradient checks")
