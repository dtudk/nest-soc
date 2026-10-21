"""
Determines which files are included when importing the whole package.
"""

import sys

from nest import constants as constants
from nest import properties as properties
from nest import layers as layers
from nest import cell as cell
from nest import boundary as boundary
from nest import degradation as degradation

import nest.packages.eis as _eis

sys.modules["nest.eis"] = _eis
eis = _eis

__all__ = [
    "constants",
    "properties",
    "layers",
    "cell",
    "boundary",
    "degradation",
    "eis",
]
