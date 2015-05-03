import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


### Pre-process training data ###

df_train = pd.read_csv("train.csv")
df_train.head()
feats = df_train.drop("revenue", axis=1) 
X = feats.values #features
y = df_train["revenue"].values #target


# Build a forest and compute the feature importances
forest = ExtraTreesRegressor(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(15):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
