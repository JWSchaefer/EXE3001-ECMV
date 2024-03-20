import ecmv

import numpy as np

from ecmv.features import Features

from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


df = ecmv.extract.generate_dataset(
    Features.Class, Features.Length, Features.Width, Features.Perimeter
)

y = df["Class"]
X = df[["Length", "Width", "Perimeter"]]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.333, random_state=42
)

classifier = DecisionTreeClassifier(max_depth=2, random_state=42)
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test) * 100
print(f"Classifier Score: {score:3.2f}%")
