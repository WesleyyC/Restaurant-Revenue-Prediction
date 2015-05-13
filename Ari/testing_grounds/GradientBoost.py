import numpy as np
import pandas as pd
from math import sqrt
from sklearn import ensemble
from sklearn.metrics import mean_squared_error



###############################################################################
# Load data
df_train = pd.read_csv("train_numerical_head.csv")
df_train.head()
feats = df_train.drop(str(42), axis=1) 
X_train = feats.values #features
y_train = df_train[str(42)].values #target


df_test = pd.read_csv("test_numerical_head.csv")
df_train.head()
X_test = feats.values #features


###############################################################################
# Preprocess
for i in range(0, len(y_train)-1):
    if y_train[i]>10000000:
        print "works"
        y_train[i]=10000000


###############################################################################
# Fit regression model


params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,
          'learning_rate': 0.001, 'loss': 'lad'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

###############################################################################
# Prediction
result = clf.predict(X_test)
result = np.asarray(result)
np.savetxt("result.csv", result, delimiter=",")

rmse = sqrt(mean_squared_error(y_train, clf.predict(X_train)))
print "GradientBoostingRegressor RMSE: " , rmse
