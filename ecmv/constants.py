import os

from appdirs import user_data_dir

MOD = "ecmv"


class DSET:
    url = r"https://github.com/JWSchaefer/EXE3001-RiceData.git"
    path = os.path.join(user_data_dir(MOD, "digiLab Solutions Ltd"))


class CACHE:
    path = os.path.join(os.environ["ECMV_BASE_PATH"], "store", "extract.csv")


if "ECMV_VERBOSE" in os.environ:
    VERBOSE = os.environ["ECMV_VERBOSE"].lower() == "true"

else:
    VERBOSE = False
