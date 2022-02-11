from ._version import __version__, __version_info__

from .atm import make_atmosphere
from .utils import BBSED
from .sim import SimpleSimulator
from .catalog import MockStarCatalog
from .sst import SSTFactory

import os
datadir = os.path.join(os.path.dirname(__file__), "data")
