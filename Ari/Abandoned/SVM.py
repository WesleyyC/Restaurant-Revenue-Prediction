from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
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

from sklearn.preprocessing import StandardScaler


# Fit regression model
clf_2 = SVR(kernel='rbf', C=1e3, gamma=0.1)


kf = KFold(len(y), n_folds=15, shuffle=True)

# CV Loop
for train_index, test_index in kf:
    # for each iteration of the for loop we'll do a test train split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    t = StandardScaler()
    X_train = t.fit_transform(X_train)
    clf_2.fit(X_train, y_train) # Train clf_2 on the training data

    X_test = t.transform(X_test)
    y_pred2[test_index] = svr_rbf.fit(X, y).predict(X)
    

### Prediction ###
result = clf_2.predict(test_feats)
result = np.asarray(result)
np.savetxt("result.csv", result, delimiter=",")

rmse = sqrt(mean_squared_error(y, y_pred2))
print "Bagging RMSE: " , rmse





y_rbf = svr_rbf.fit(X, y).predict(X)


plt.plot(X, y_rbf, c='g', label='RBF model')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()