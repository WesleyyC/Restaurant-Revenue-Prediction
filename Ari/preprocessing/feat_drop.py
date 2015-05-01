import numpy as np
import pandas as pd

###############################################################################
# Load data
df_train = pd.read_csv("train_numerical_head.csv")
df_train.head()

df_test = pd.read_csv("test_numerical_head.csv")
df_train.head()


###############################################################################
# Drop features
p_to_drop = [	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
				0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 
				0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
				0]

for i in range(5, 42):
	if p_to_drop[i-5] == 0:
		df_train = df_train.drop(i, axis=1)
		df_test = df_test.drop(i axis=1)




###############################################################################
# Save to File
result = np.asarray(result)
np.savetxt("result_train.csv", df_train, delimiter=",")
np.savetxt("result_test.csv", df_test, delimiter=",")


#plot_r2(y, y_pred2, "Performance of GradientBoostingRegressor")
#plt.show()
#r2_score(y, y_pred2)
