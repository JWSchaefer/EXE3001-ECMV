import os
import shutil
import subprocess

import pandas as pd

from .errors import DataError

from .constants import CACHE, DSET, VERBOSE
from .decorators import get_confirmation


def get_store() -> pd.DataFrame:
    df = pd.read_csv(CACHE.path, index_col=0)
    return df


def get_paths() -> list:

    results = []

    def _it(i):

        if os.path.isfile(i.path) and "jpg" in i.name:
            results.append((i))

        elif os.path.isdir(i.path):
            for j in os.scandir(i):
                _it(j)

    for i in os.scandir(DSET.path):
        _it(i)

    return results


class Manager(object):

    def __init__(self):
        if VERBOSE:
            print(f"Dataset location: {DSET.path}")

    def __enter__(self):

        if VERBOSE:
            print(f"Entering dataset manager context.")

        if not os.path.exists(DSET.path):
            if VERBOSE:
                print(f"{DSET.path} not found. Creating it.")
            os.mkdir(DSET.path)

        return self

    def __exit__(self, type, value, traceback):
        if VERBOSE:
            print(f"Exiting dataset manager context.")
        if not self.shallow_check():
            if VERBOSE:
                print(f"Dataset faled shallow check.")
            self.clean_dset()

    def clone_and_validate(self):
        if not self.shallow_check():
            self.clone_dset(
                question=f"Rice image dataset not found in {DSET.path}.\nDownload it? (Required for the ecmv package to function)"
            )

    def shallow_check(self):
        if VERBOSE:
            print("Performing shallow check.")

        result = len(get_paths()) == 75_000

        if result and VERBOSE:
            print("Shallow check passed.")

        elif (not result) and VERBOSE:
            print("Shallow check failed.")

        return result

    @get_confirmation
    def clone_dset(self, **kwargs):
        self.clean_dset()
        process = subprocess.run(["git", "clone", DSET.url, DSET.path])

    def clean_dset(self):
        if VERBOSE:
            print(f"Cleaning dataset.")
        for i in os.scandir(DSET.path):
            try:
                if os.path.isfile(i.path) or os.path.islink(i.path):
                    os.unlink(i.path)

                elif os.path.isdir(i.path):
                    shutil.rmtree(i.path)

            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (i.path, e))
