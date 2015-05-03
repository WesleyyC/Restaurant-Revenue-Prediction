import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

df_train = pd.read_csv("testwithrevs.csv")
df_train.head()
feats = df_train.drop("42", axis=1) 
X = feats.values #features
y = df_train["42"].values #target


# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)

ranking = rfe.ranking_.reshape(digits.images[0].shape)

print ranking