from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
import scipy as sci


### Plotting function ###

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

def plot_r2(y, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.scatter(y, y_pred, marker='.')
    plt.xlabel("Actual Target"); plt.ylabel("Predicted Target")
    plt.title(title)
    xmn, xmx = plt.xlim()
    ymn, ymx = plt.ylim()
    mx = max(xmx, ymx) 
    buff = mx * .1
    plt.text(xmn + buff, mx - buff, "R2 Score: %f" % (r2_score(y, y_pred), ), size=15)
    plt.plot([0., mx], [0., mx])
    plt.xlim(xmn, mx)
    plt.ylim(ymn, mx)





### Pre-process training data ###

df_train = pd.read_csv("train.csv")
df_train.head()
feats = df_train.drop("revenue", axis=1) 
X = feats.values #features
y = df_train["revenue"].values #target




### AdaBoost ###

from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler


kf = KFold(len(y), n_folds=15, shuffle=True)

y_pred1 = np.zeros(len(y), dtype=y.dtype) # where we'll accumulate predictions
y_pred2 = np.zeros(len(y), dtype=y.dtype) # where we'll accumulate predictions

clf_1 = ensemble.GradientBoostingClassifier()
clf_2 = AdaBoostRegressor(ensemble.GradientBoostingClassifier(),
                         n_estimators=50, random_state=None)




# CV Loop
for train_index, test_index in kf:
    # for each iteration of the for loop we'll do a test train split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    t = StandardScaler()
    X_train = t.fit_transform(X_train)
    clf_1.fit(X_train, y_train) # Train clf_1 on the training data
    clf_2.fit(X_train, y_train) # Train clf_2 on the training data


    X_test = t.transform(X_test)
    y_pred1[test_index] = clf_1.predict(X_test) # Predict clf_1 using the test and store in y_pred
    y_pred2[test_index] = clf_2.predict(X_test) # Predict clf_2 using the test and store in y_pred
    



plot_r2(y, y_pred1, "Performance of CV DecisionTreeRegressor")
plt.show()
r2_score(y, y_pred1)
rmse = sqrt(mean_squared_error(y, y_pred1))

print "GradientBoostingClassifier CV 1 rmse: " , rmse


plot_r2(y, y_pred2, "Performance of CV AdaBoost")
plt.show()
r2_score(y, y_pred2)
rmse = sqrt(mean_squared_error(y, y_pred2))

print "Linear Regression CV 2 rmse: " , rmse
