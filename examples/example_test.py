import ecmv

from PIL import Image

from ecmv.features import Features
from matplotlib import pyplot as plt

import skimage.measure


def foo(path):

    with Image.open(path) as im:

        shannon_r = skimage.measure.shannon_entropy(im.getchannel("R"))
        shannon_g = skimage.measure.shannon_entropy(im.getchannel("G"))
        shannon_b = skimage.measure.shannon_entropy(im.getchannel("B"))

        print(f"R:\t{shannon_r:.5f}")
        print(f"G:\t{shannon_g:.5f}")
        print(f"B:\t{shannon_b:.5f}")

        im.show()

    return 0.0


sample = ecmv.extract.test(
    Features.Class,
    Features.Length,
    Features.Width,
    foo,
    shuffle=True,
    seed=42,
)

print(sample)
