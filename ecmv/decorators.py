import functools
import numpy as np

from .errors import FeatureError
from .features import Features


def check_features(function):
    """
    A decorator that validates the features provided to the inner function are either:
        A pre-evaluated feature defined by the features.Features enum
        A callabele

    Parameters
    ----------
    function : Callable
        The function to be wrapped by this decorator
    """

    @functools.wraps(function)
    def wrapper(*features, **kwargs):

        names = [f.__name__ if callable(f) else f.name for f in features]

        if len(names) != len(np.unique(names)):
            raise FeatureError(
                "Features must have unique names. Ensure the provided functions"
                + "do not share names with each other or the predefined features"
                + " provided"
            )

        if "shuffle" not in kwargs.keys():
            kwargs["shuffle"] = False

        if "seed" not in kwargs.keys():
            kwargs["seed"] = None

        for f in features:
            if not (isinstance(f, Features) or callable(f)):
                raise FeatureError(
                    f"Feature {f} must be {[str(f) for f in Features]} or Callable"
                )
        result = function(*features, **kwargs)

        return result

    return wrapper


def get_confirmation(function):
    """
    ...

    Parameters
    ----------
    function : Callable
        The function to be wrapped by this decorator
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):

        question = kwargs["question"] if "question" in kwargs.keys() else "Continue?"

        answer = input(f"{question} [y/n]: ")

        if answer.lower() in ["y", "yes"]:
            return function(*args, **kwargs)

        elif answer.lower() in ["n", "no"]:
            return None

        else:
            question = "Invalid input, please try again."
            wrapper(*args, **kwargs)

    return wrapper
