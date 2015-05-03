import numpy as np
import pandas as pd

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
# Drop features
p_to_drop = [	1, 1, 1, 1, 1, 1, 1, 
				1, 1, 1, 1, 1, 0, 0, 
				0, 0, 0, 0, 1, 1, 1, 
				1, 1, 0, 0, 0, 0, 1, 
				1, 0, 0, 0, 0, 0, 0,
				0, 0]

for i in range(5, 42):
	print i
	if p_to_drop[i-5] == 0:
		df_train = df_train.drop(str(i), axis=1)
		df_test = df_test.drop(str(i), axis=1)




###############################################################################
# Save to File
df_train = np.asarray(df_train)
df_train = np.asarray(df_test)
np.savetxt("result_train.csv", df_train, delimiter=",")
np.savetxt("result_test.csv", df_test, delimiter=",")


#plot_r2(y, y_pred2, "Performance of GradientBoostingRegressor")
#plt.show()
#r2_score(y, y_pred2)
