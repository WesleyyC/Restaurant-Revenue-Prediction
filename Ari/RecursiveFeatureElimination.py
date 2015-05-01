
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


df_train = pd.read_csv("train.csv")
feats = df_train.drop("revenue", axis=1) 

X = feats.values #features
y = df_train["revenue"].values #target


# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')

t = StandardScaler()
X = t.fit_transform(X)
y = t.fit_transform(y)




count = 0
for elem in y:
	print elem
	count += 1
	if count > 10:
		break


count = 0
for elem in X:
	print elem
	count += 1
	if count > 10:
		break

rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()