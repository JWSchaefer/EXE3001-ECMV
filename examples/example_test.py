import ecmv

from PIL import Image

from ecmv.features import Features
from matplotlib import pyplot as plt


def foo(path):

    with Image.open(path) as im:
        im.show()

    return 0.0


sample = ecmv.extract.test(
    Features.Class,
    Features.Length,
    Features.Width,
    Features.FName,
    foo,
    shuffle=True,
    seed=42,
)

print(sample)
