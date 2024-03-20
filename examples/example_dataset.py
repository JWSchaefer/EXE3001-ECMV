import ecmv

import numpy as np
import seaborn as sns

from PIL import Image
from ecmv.features import Features

from matplotlib import pyplot as plt


def mean_red(path):
    with Image.open(path) as im:

        red = im.getchannel("R")

        return np.mean(red)

    return np.nan


df = ecmv.extract.generate_dataset(
    Features.Class,
    Features.Length,
    Features.Width,
    Features.Perimeter,
    mean_red,
)


sns.pairplot(df, hue="Class")
plt.savefig("/Users/joe/dist.png")
plt.show()
