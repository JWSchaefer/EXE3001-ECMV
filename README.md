# Exeter College Machine Vision 

Code for the EXE3002 - Classifiers and Machine Vision written assignment

![Rice Image](https://github.com/JWSchaefer/EXE30001-RiceData/blob/main/Jasmine/Jasmine%20(10070).jpg?raw=True)


## Index

- [Overview](#overview)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)
- [Support](#support)

## Overview

This package aims to provide a robust framework within which to demonstrate your machine vision skills

It should serve as the starting point for your code in the machine vision aspect of the EXE3002 Classifiers and Machine Vision written assignment.

### Dataset

The package allows easy processing of the [CINAR & KOKLU](https://www.muratkoklu.com/datasets/) rice dataset.
It containts 75,000 images of 5 species of rice grain. 

**Arborio, Basmati, Ipsala, Jasmine, and Karacadag**


### Aim

The aim is the maximise the performance of a 2 node ensamble decision tree clasifier when predicting the class (species) of features extracted from images of previously unseen rice grains.



```python
classifier = DecisionTreeClassifier(max_depth=2)
```

This should be achieved by designing functions that accept a path to an image as the input and return a derived quantity as the output.

e.g. 

```python

def mean_red(path): # Calculates the mean intesity of the red channel
    with PIL.Image.open(path) as im: # Open the immage using PIL
        red = im.getchannel("R")     # Extract the red channel
        return np.mean(red)          # Return the mean value
```

These functions can be passed to `ecmv.extract.test` to evaluate them on an random image as part of an iterative design process.

```python 
ecmv.extract.test(mean_red, seed=42)) # Evaluate mean_red on a single image
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.547808</td>
    </tr>
  </tbody>
</table>

They can also be passed to the `ecmv.extract.generate_dataset` function to apply them across the full dataset.

```python 
ecmv.extract.generate_dataset(mean_red) # Evaluate mean_red on all images
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.547808</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22.880752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.906224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22.915984</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.652096</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>74995</th>
      <td>18.285168</td>
    </tr>
    <tr>
      <th>74996</th>
      <td>33.421280</td>
    </tr>
    <tr>
      <th>74997</th>
      <td>20.791216</td>
    </tr>
    <tr>
      <th>74998</th>
      <td>17.440032</td>
    </tr>
    <tr>
      <th>74999</th>
      <td>17.440896</td>
    </tr>
  </tbody>
</table>
<p>75000 rows × 1 columns</p>

There are 3 pre-calculated features: `Length`, `Width`, and `Perimiter` can be rapidly evaluated. 
The image filename and class are also known.

```python 

  ecmv.extract.test(
      Features.Length, Features.Width, Features.Perimeter, mean_red, seed=42
  ) # Evaluate on a single image
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Length</th>
      <th>Width</th>
      <th>Perimeter</th>
      <th>mean_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K</td>
      <td>0.852247</td>
      <td>0.559312</td>
      <td>1.209401</td>
      <td>19.547808</td>
    </tr>
  </tbody>
</table>



*See [Examples](#examples) for full code*

### Reccomended Packages

You may want to utilise the following python packages when building your functions.

1. **Numpy** 
    Popular scientific computing package with powerful N-dimensional array object, sophisticated (broadcasting) functions, and useful linear algebra, Fourier transform, and random number capabilities


2. **PIL** 
    Image processing with extensive file format support, an efficient internal representation.

3. **OpenCV**
   Advanced open source Computer vision library with over 2500 easy to use algorithms.





## Installation

Requires python $>=$ 3.8

Install the package via `pip` or your favourite package manager

```bash 
$ pip install ecmv
```

When you first import the module you will be asked to download the dataset. The module cannot be used without doing this step.
The dataset is about 205.73 Mb and may take a few minutes to download.

```bash
$ python -c 'import ecmv'
```
```
Rice image dataset not found in PATH_TO_DATASET.
Download it? (Required for the ecmv package to function) [y/n]: y
Cleaning dataset
Cloning into '/Users/joe/Library/Application Support/ecmv'...
remote: Enumerating objects: 74714, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 74714 (delta 0), reused 3 (delta 0), pack-reused 74711
Receiving objects: 100% (74714/74714), 205.73 MiB | 4.76 MiB/s, done.
Resolving deltas: 100% (3/3), done.
Updating files: 100% (75002/75002), done.
```


If any errors are encountered, they should be resolevd automatically. If they persist you can debug them as follows:

1. Set the `ECMV_VERBOSE` environment variable. This will force the programme to output key information relating to the handling of the dataset




   ```bash
   # Mac / Linux
   $ export ECMV_VERBOSE=True
   ```
    ```bash
   # Windows
   > set ECMV_VERBOSE=True
   ```

2. Import the `ecmv` package
   ```bash
   $ python -c "import ecmv"
   ```
   ```
   Dataset location: PATH_TO_DATASET
   ...
   ```

3. Remove the directory 
   ```bash
   # Mac / Linux
   $ sudo rm -r PATH_TO_DATASET
   ```
  
   ```bash
   # Windows
   > rd /s PATH_TO_DATASET
   ```



## Examples

### Example 1 - Function Test

Test a function `foo` on a single rice image

**Code** 
```python
# example_test.py

# Imports
import ecmv

from PIL import Image

from ecmv.features import Features
from matplotlib import pyplot as plt



def foo(path): # Test function

    with Image.open(path) as im: # Load image
        im.show() # SHow it

    return 0.0 # Return a number


sample = ecmv.extract.test( # Apply function to a random image
    Features.Class, Features.Length, Features.Width, foo, shuffle=True, seed=42
)
```
**Output**
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Length</th>
      <th>Width</th>
      <th>foo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I</td>
      <td>0.956632</td>
      <td>0.482663</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>

![Rice Image](https://github.com/JWSchaefer/EXE30001-RiceData/blob/main/Ipsala/Ipsala%20(9456).jpg?raw=True)

### Example 2 - Function Application

Extract the mean red channel value from every rice image

**Code**

```python
# example_dataset.py

# Imports
import ecmv

import numpy as np
import seaborn as sns

from PIL import Image
from ecmv.features import Features

from matplotlib import pyplot as plt


def mean_red(path): # Calculates the mean intesity of the red channel
    with PIL.Image.open(path) as im:   # Open the immage using PIL
        red = im.getchannel("R")       # Extract the red channel
        return np.mean(red)            # Return the mean value


# Apply to full dataset
df = ecmv.extract.generate_dataset(
    Features.Class,
    Features.Length,
    Features.Width,
    Features.Perimeter,
    mean_red,
)


# Produce pairplot
sns.pairplot(df, hue="Class")
plt.show()
```
**Input**
```bash
$ python example_test.py
```
**Output**
![Pairplot](https://github.com/JWSchaefer/EXE30001-RiceData/blob/main/dist.png?raw=true)

### Example 3 - Classification

An example evaluating the performance of the random forrest classifier using the precalculaed features

**Code**
```python
# example_classify.py

# Imports
import ecmv

import numpy as np

from ecmv.features import Features

from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Extract precalculated features
df = ecmv.extract.generate_dataset(
    Features.Class, Features.Length, Features.Width, Features.Perimeter
)

# Split into output varible (Class) and observed features (Length, Width, Perimeter)
y = df["Class"]
X = df[["Length", "Width", "Perimeter"]]

# Split into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.333, random_state=42
)

# Train the 2 node ensemble decision tree clasifier on the training set
classifier = DecisionTreeClassifier(max_depth=2, random_state=42)
classifier.fit(X_train, y_train)

# Evalute the performance of the model on the test set
score = classifier.score(X_test, y_test) * 100
print(f"Classifier Score: {score:3.2f}%")
```
**Input**
```bash
$ python example_classify.py
```
**Output**
```
Classifier Score: 75.99%
```
## Documentation

### Structure
```
ecmv
├── features
│   ├── Features 
│   └── get_feature_names
└── extract
    ├── generate_dataset 
    └── test
```

### Features Module
#### `features.Features` 

An Enum defining precalculated features



```python
class Features(Enum):
    FName     = 1
    Class     = 2
    Length    = 3
    Width     = 4
    Perimeter = 5
```

**Attributes**

```
Fname : str
  The jpg file name

Class : str
  The rice species identifier

  A ─> Arborio
  B ─> Basmati
  I ─> Ipsala
  J ─> Jasmine
  K ─> Karacadag

Perimiter : float
  The non-dimensional perimiter of the rice grain. 
  (Normalised by the image size)

Length : float
  The non-dimensional length of the rice grain. 
  (Normalised by the image size)

Width : float
  The non-dimensional width of the rice grain. 
  (Normalised by the image size)
```

#### `features.get_feature_names` 

An getter for the names of the available preclculated features

```python
def get_feature_names() -> list[str]:
    ...
``` 
**Returns**
```
names : list[str]
  A list of the names of the available preclculated features
```

### Extract Module
#### `extract.generate_dataset`

A function to extract features from the 75,000 images in the [CINAR & KOKLU](https://www.muratkoklu.com/datasets/) rice dataset.

```python
@check_features
def generate_dataset(*features, shuffle = False, seed = 42) -> pd.DataFrame:
    ...
```
**Parameters**
``` 
*features : Callable(str) | Feature
    An array of features to be extracted from each image in the dataset.
    Must be either:
        a) A function f(path) -> float accepting a path to an jpg file
        b) A features.Features enum corrasponding to a precalculated
           feature

shuffle : bool = True
    A boolean to determine if the images are to be shuffled prior to extraction.

seed : int = None
    Ensures a repeatable shuffle if not None.
```
**Returns**
```
data : pd.Dataframe
    A pandas dataframe where each row contains the features extracted from an image
```
#### `extract.test`

A function to extract features from the a single im age from the [CINAR & KOKLU](https://www.muratkoklu.com/datasets/) rice dataset for testing and development purposes.

```python
@check_features
def test(*features, shuffle = False, seed = 42) -> pd.DataFrame:
    ...
```
**Parameters**
```
*features : Callable(str) | Feature
    An array of features to be extracted from each image in the dataset.
    Must be either:
        a) A function f(path) -> float accepting a path to an jpg file
        b) A features.Features enum corrasponding to a precalculated
           feature

shuffle : bool = True
    A boolean to the features shoukld be extracted from a random image

seed : int = None
    Ensures a repeatable shuffle if not None.
```
**Returns**
```
data : pd.Dataframe
    A pandas dataframe where each row contains the features extracted from an image
```
## Support

If you are struggling to use this code, please contact your supervisor.

