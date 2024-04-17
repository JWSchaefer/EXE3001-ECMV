import ecmv

import numpy as np

from ecmv.features import Features

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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
classifier = RandomForestClassifier(max_depth=2, n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evalute the performance of the model on the test set
score = classifier.score(X_test, y_test) * 100
print(f"Classifier Score: {score:3.2f}%")
