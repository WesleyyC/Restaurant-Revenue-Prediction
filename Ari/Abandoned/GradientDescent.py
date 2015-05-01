
	
# Grid Search for Algorithm Tuning
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import Ridge


### Plotting function ###

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDClassifier


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



### Preprocessing ### 

dataset = pd.read_csv("train.csv")
dataset.head()
test = pd.read_csv("test.csv")
dataset.head()
feats = dataset.drop("revenue", axis=1) 
X = feats.values #features
y = dataset["revenue"].values #target

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
	max_depth=1, random_state=0).fit(X, y)

clf.score(X_test, y_test)                 

np.savetxt("result.csv", result, delimiter=",")




