from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

###############################################################################
# Load data
### Preprocess Testing Data ###

df_test = pd.read_csv("test.csv")
df_test.head()
test_feats = df_test.drop("Bit City", axis=1)
test_feats = test_feats.drop("Type", axis=1)
test_feats = test_feats.drop("P3", axis=1)
test_feats = test_feats.drop("P7", axis=1)
test_feats = test_feats.drop("P8", axis=1)
test_feats = test_feats.drop("P9", axis=1)
test_feats = test_feats.drop("P10", axis=1)
test_feats = test_feats.drop("P12", axis=1)
test_feats = test_feats.drop("P13", axis=1)
test_feats = test_feats.drop("P14", axis=1)
test_feats = test_feats.drop("P15", axis=1)
test_feats = test_feats.drop("P16", axis=1)
test_feats = test_feats.drop("P17", axis=1)
test_feats = test_feats.drop("P18", axis=1)
test_feats = test_feats.drop("P21", axis=1)
test_feats = test_feats.drop("P24", axis=1)
test_feats = test_feats.drop("P25", axis=1)
test_feats = test_feats.drop("P26", axis=1)
test_feats = test_feats.drop("P27", axis=1)
test_feats = test_feats.drop("P30", axis=1)
test_feats = test_feats.drop("P31", axis=1)
test_feats = test_feats.drop("P32", axis=1)
test_feats = test_feats.drop("P33", axis=1)
test_feats = test_feats.drop("P34", axis=1)
test_feats = test_feats.drop("P35", axis=1)
test_feats = test_feats.drop("P36", axis=1)
test_feats = test_feats.drop("P37", axis=1)
# stop here for best 15


### Pre-process training data ###

df_train = pd.read_csv("train.csv")
df_train.head()
feats = df_train.drop("Bit City", axis=1)
feats = feats.drop("revenue", axis=1)
feats = feats.drop("Type", axis=1)
feats = feats.drop("P3", axis=1)
feats = feats.drop("P7", axis=1)
feats = feats.drop("P8", axis=1)
feats = feats.drop("P9", axis=1)
feats = feats.drop("P10", axis=1)
feats = feats.drop("P12", axis=1)
feats = feats.drop("P13", axis=1)
feats = feats.drop("P14", axis=1)
feats = feats.drop("P15", axis=1)
feats = feats.drop("P16", axis=1)
feats = feats.drop("P17", axis=1)
feats = feats.drop("P18", axis=1)
feats = feats.drop("P21", axis=1)
feats = feats.drop("P24", axis=1)
feats = feats.drop("P25", axis=1)
feats = feats.drop("P26", axis=1)
feats = feats.drop("P27", axis=1)
feats = feats.drop("P30", axis=1)
feats = feats.drop("P31", axis=1)
feats = feats.drop("P32", axis=1)
feats = feats.drop("P33", axis=1)
feats = feats.drop("P34", axis=1)
feats = feats.drop("P35", axis=1)
feats = feats.drop("P36", axis=1)
feats = feats.drop("P37", axis=1)
# stop here for best 15


X = feats.values #features
y = df_train["revenue"].values #target



for i in range(0, len(y)-1):
    if y[i]>10000000:
        y[i]=10000000

offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

###############################################################################
# Fit regression model

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

kf = KFold(len(y), n_folds=137, shuffle=True)

y_pred = np.zeros(len(y), dtype=y.dtype) # where we'll accumulate predictions



# CV Loop
for train_index, test_index in kf:
    # for each iteration of the for loop we'll do a test train split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    t = StandardScaler()
    X_train = t.fit_transform(X_train)
    clf.fit(X_train, y_train) # Train clf_2 on the training data


    X_test = t.transform(X_test)
    y_pred[test_index] = clf.predict(X_test) # Predict clf_2 using the test and store in y_pred



### Prediction ###
result = clf.predict(test_feats)
result = np.asarray(result)
np.savetxt("result.csv", result, delimiter=",")

rmse = sqrt(mean_squared_error(y, y_pred))
print "GradientBoostingRegressor RMSE: " , rmse




###############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

###############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
