# RestaurantRevenuePrediction
Term project for COSI 123A Statistical Machine Learning Course @ Brandeis Univeristy 

https://www.kaggle.com/c/restaurant-revenue-prediction/data 

Author: Ari Ben-Elazar, Will Burstein, Wesley Wei Qian 


### Update README.md for each commit
- What approach you use?
- What's the score?
- What's the file associated with this approach?
- Name and Date
- Keep the file and start a new file for your next approach. :)

### Approach Records

1st
- Approach: Linear Regression with standardized Data and regression tree on the data the training data that Wesley cleaned
- didn't label any of the variables as categorical and used first 120 rows as training partition and last 17 rows as testing 
- partitition.  Note that I didn't adjust any parameters of the model, as I just wanted to get a baseline score for how this -data performs out of the box- clearly a very hard dataset to work with!
- Score: 2757036.58 RMSE for linear regression and 3364990.53 RMSE for regression tree.  
- File: linearRegression_RegressionTree_20150402
- Name/Date: Will

2nd
- Approach: Linear Regression with standardized Data and regression tree on the data the training data that Wesley cleaned
- didn't label any of the variables as categorical and used first 120 rows as training partition and last 17 rows as testing 
- partitition.  Note that I didn't adjust any parameters of the model, as I just wanted to get a baseline score for how this -data performs out of the box- clearly a very hard dataset to work with!
- Score: 2757036.58 RMSE for linear regression and 3364990.53 RMSE for regression tree.  
- File: linearRegression_RegressionTree_20150402
- Name/Date: Will

3rd
- Approach: bagging with guesses on which features are catigorical.  Took guesses for combinations up to 10 features that were 
categorical.  Tried on training partition of first 120 rows and tested model on last 17 rows of data.
- tried this approach with both a linear term, interactive term, and quadratic term on the features 
- Score: so far the quadratic term has worked best with ~1660000 RMSE on the test partition and RMSE of 1808855.28789 on Kaggle  
- File: termProjScript_20150402
- Name/Date: Will 4/12/15


### Data Description
#### File Description
- train.csv: the training set. Use this dataset for training your model. 
- test.csv: the test set. To deter manual "guess" predictions, Kaggle has supplemented the test set with additional "ignored" data. These are not counted in the scoring.
- sampleSubmission.csv: a sample submission file in the correct format

#### Field Description
- Id : Restaurant id. 
- Open Date : opening date for a restaurant
- City : City that the restaurant is in. Note that there are unicode in the names. 
- City Group: Type of the city. Big cities(class "1" in our processed data), or Other(class "0" in our processed data). 
- Type: Type of the restaurant. FC: Food Court(class "2" in our processed data), IL: Inline (class "1" in our processed data), DT: Drive Thru(class "3" in our processed data), MB: Mobile(class "4" in our processed data)
- P1, P2 - P37: There are three categories of these obfuscated data. Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
- Revenue: The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. Please note that the values are transformed so they don't mean real dollar values. 
