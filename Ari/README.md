##############
### README ###
##############

FOLDER STRUCTURE:
	"testing_grounds" is where we run tests, try things.
	"test_files" is where we store mods to the test csv
	"train_files" is where we store mods to the train csv
	"working_regressors" is where we keep working regressors
	"preprocessing" is where we keep working preprocessing techniques
	"needs_work" is where we keep works in progress, or backlogged techniques

UNDER CONSTRUCTION: 
	Grid Search
	Gradient Boost


Feature Elimination:

(1)	Recursive Feature Elimination
	
(2) Boruta

(3) ExtraTreesRegressor

Regressor Methods:


(1) Linear Regression
	RMSE : 3.3M
	Comments: Poor performance. Baseline.


(2) AdaBoost
	RMSE : 2.4M
	Comments: Better performance. Will try improving.


(3) BaggingRegressor
	RMSE : 2.6M

(4) LogitBoost
	RMSE : 
