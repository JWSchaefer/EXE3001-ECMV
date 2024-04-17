from enum import Enum


class Features(Enum):
    """
    An Enum defining the precalculated features

    Attributes
    ----------
    Fname : str
        The jpg file name

    Class : str
        The rice species identifier
        A = Arborio
        B = Basmati
        I = Ipsala
        J = Jasmine
        K = Karacadag

    Perimiter : float
        The non-dimensional perimiter of the rice grain. (Normalised by the image size)

    Length : float
        The non-dimensional length of the rice grain. (Normalised by the image size)

    Width : float
        The non-dimensional width of the rice grain. (Normalised by the image size)
    """

    Class = 1
    Length = 2
    Width = 3
    Perimeter = 4


def get_feature_names():
    return [f.name for f in Features]
