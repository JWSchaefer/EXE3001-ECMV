import os

os.environ["ECMV_BASE_PATH"] = str(__path__[0])

from .store import Manager

with Manager() as manager:
    manager.clone_and_validate()

from . import extract
from . import features

__all__ = ["extract", "features"]
