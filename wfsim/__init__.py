from ._version import __version__, __version_info__

from .atm import make_atmosphere
from .utils import BBSED
from .simulator import SimpleSimulator
from .catalog import MockStarCatalog
from .sst import SSTFactory, SSTBuilder
from .app import WFApp

import os
datadir = os.path.join(os.path.dirname(__file__), "data")
