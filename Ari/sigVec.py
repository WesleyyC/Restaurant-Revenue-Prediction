import numpy as np
import pandas as pd
import scipy as sci



df = pd.read_csv("str_num_test.csv")


count = 0
for i in range (1, 137):
	row = next(df.iterrows())
	for j in range(0, len(row)-1):
		if row[j] == 0:
			count += 1

total = 137.0*42.0

print "Zeros: ", count, "\nPercent: ", (count/total)*100
