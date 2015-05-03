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

### Preprocess Testing Data ###

### Pre-process training data ###

df_train = pd.read_csv("train.csv")
df_train.head()
feats = df_train
print feats.head()
feats = feats.drop("1", axis=1)
feats = feats.drop("6", axis=1)
feats = feats.drop("7", axis=1)
feats = feats.drop("8", axis=1)
feats = feats.drop("10", axis=1)
feats = feats.drop("11", axis=1)
feats = feats.drop("12", axis=1)
feats = feats.drop("13", axis=1)
feats = feats.drop("15", axis=1)
feats = feats.drop("17", axis=1)
feats = feats.drop("18", axis=1)
feats = feats.drop("19", axis=1)
feats = feats.drop("21", axis=1)
feats = feats.drop("22", axis=1)
feats = feats.drop("26", axis=1)
feats = feats.drop("27", axis=1)
feats = feats.drop("29", axis=1)
feats = feats.drop("30", axis=1)
feats = feats.drop("32", axis=1)
feats = feats.drop("33", axis=1)
feats = feats.drop("34", axis=1)
feats = feats.drop("36", axis=1)
feats = feats.drop("37", axis=1)
feats = feats.drop("38", axis=1)
feats = feats.drop("39", axis=1)
feats = feats.drop("40", axis=1)
feats = feats.drop("41", axis=1)
feats = feats.drop("42", axis=1)
# stop here for best 15


X = feats.values #features
y = df_train["42"].values #target



for i in range(0, len(y)-1):
    if y[i]>10000000:
        y[i]=10000000



df_test = pd.read_csv("test.csv")
df_test.head()
test_feats = df_test
print feats.head()
# stop here for best 15



### Bagging ###

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler


kf = KFold(len(y), n_folds=100, shuffle=True)

y_pred2 = np.zeros(len(y), dtype=y.dtype) # where we'll accumulate predictions

clf_2 = BaggingRegressor(base_estimator=None, n_estimators=15, 
    #max_samples=1.0, max_features=1.0, 
    bootstrap=True, 
    bootstrap_features=True, oob_score=False, n_jobs=600, 
    random_state=None, verbose=0)




# CV Loop
for train_index, test_index in kf:
    # for each iteration of the for loop we'll do a test train split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    t = StandardScaler()
    X_train = t.fit_transform(X_train)
    clf_2.fit(X_train, y_train) # Train clf_2 on the training data


    X_test = t.transform(X_test)
    y_pred2[test_index] = clf_2.predict(X_test) # Predict clf_2 using the test and store in y_pred
    print "q"


print "x"
### Prediction ###
result = clf_2.predict(test_feats)
result = np.asarray(result)
np.savetxt("result.csv", result, delimiter=",")
print "y"
rmse = sqrt(mean_squared_error(y, y_pred2))
print "Bagging RMSE: " , rmse

plot_r2(y, y_pred2, "Performance of Bagging")
#plt.show()
r2_score(y, y_pred2)

