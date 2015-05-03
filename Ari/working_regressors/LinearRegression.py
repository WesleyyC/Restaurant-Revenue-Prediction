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




for i in range(0, len(y)-1):
    if y[i]>10000000:
        print "sdfjsd"
        X.pop(i)
        y.pop(i)



### Linear Regression ###

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


kf = KFold(len(y), n_folds=15, shuffle=True)

y_pred = np.zeros(len(y), dtype=y.dtype) # where we'll accumulate predictions
lr = LinearRegression()

# CV Loop
for train_index, test_index in kf:
    # for each iteration of the for loop we'll do a test train split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    t = StandardScaler()
    X_train = t.fit_transform(X_train)
    lr.fit(X_train, y_train) # Train on the training data


    X_test = t.transform(X_test)
    y_pred[test_index] = lr.predict(X_test) # Predict using the test and store in y_pred
    



plot_r2(y, y_pred, "Performance of CV Linear Regression")
plt.show()
r2_score(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))

print "Linear Regression CV rmse: " , rmse
