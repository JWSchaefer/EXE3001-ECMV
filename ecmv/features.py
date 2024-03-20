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

    FName = 1
    Class = 2
    Length = 3
    Width = 4
    Perimeter = 5


def get_feature_names():
    return [f.name for f in Features]
