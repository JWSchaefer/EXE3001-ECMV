from .features import Features
from .store import get_store, get_paths
from .decorators import check_features

import numpy as np
import pandas as pd

from tqdm import tqdm


@check_features
def generate_dataset(*features, **kwargs) -> pd.DataFrame:
    """
    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y
        Description of parameter `y` (with type not specified).


    A function to extract features from the 75,000 images in the CINAR & KOKLU
    rice dataset

    https://www.muratkoklu.com/datasets/
    https://doi.org/10.1016/j.compag.2021.106285

    Parameters
    ----------
    `*features : list[Callable(str) | Feature]`
        An array of features to be extracted from each image in the dataset.
        Must be either:
            a) A function f(path) -> float accepting a path to an jpg file
            b) A features.Features enum corrasponding to a precalculated
               feature

    Returns
    -------
    `data : pd.Dataframe`
        A pandas dataframe with column names corrasponding to the features
        provided
    """

    seed = kwargs["seed"]
    shuffle = kwargs["seed"]

    names = [f.__name__ if callable(f) else f.name for f in features]

    store = get_store()
    paths = get_paths()

    cach = [f for f in features if isinstance(f, Features)]
    calc = [f for f in features if callable(f)]

    cached_data = pd.DataFrame(
        {f.name: store[f.name] for f in cach}, columns=[f.name for f in cach]
    )

    extracted_data = pd.DataFrame(
        [[f(i.path) for f in calc] for i in tqdm(paths, ncols=80)],
        columns=[f.__name__ for f in calc],
    )

    data = pd.concat([cached_data, extracted_data], axis=1)[names]

    if shuffle:
        data = data.sample(data.shape[0], random_state=seed)

    return data


@check_features
def test(*features, **kwargs) -> pd.DataFrame:

    seed = kwargs["seed"]
    shuffle = kwargs["shuffle"]

    names = [f.__name__ if callable(f) else f.name for f in features]

    store = get_store()
    paths = get_paths()

    store["path"] = [i.path for i in paths]

    if shuffle:
        sample = store.sample(1, random_state=seed).reset_index()
    else:
        sample = store.reset_index()

    calc = [f for f in features if callable(f)]
    cach = [f for f in features if isinstance(f, Features)]

    cached_data = pd.DataFrame(
        {f.name: sample[f.name] for f in cach}, columns=[f.name for f in cach]
    )[:1].reset_index()

    extracted_data = pd.DataFrame(
        [[f(sample["path"][0]) for f in calc]],
        columns=[f.__name__ for f in calc],
    ).reset_index()

    return pd.concat([cached_data, extracted_data], axis=1)[names]
