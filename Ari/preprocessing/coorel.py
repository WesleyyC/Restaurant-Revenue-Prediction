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


df = pd.read_csv("test.csv")
df2 = pd.read_csv("train.csv")
for i in range(1, 2):
    x = df[str(i)]
    y = df[str(3)]
    title = "Feature #", i
    plot_r2(x, y, title)
    plt.show()
    x = 100*df[str(i)]
    y = df[str(3)]
    title = "100 Feature #", i
    plot_r2(x, y, title)
    plt.show()
