# RestaurantRevenuePrediction
Term project for COSI 123A Statistical Machine Learning Course @ Brandeis Univeristy 

https://www.kaggle.com/c/restaurant-revenue-prediction/data 

Author: Ari Ben-Elazar, Will Burstein, Wesley Wei Qian 

### To Do
- SVM
- Feature exploring


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

4th
- Approach: bagging with combination of features, but it doesn't seem to work and it should be since if one parameter is not useful, the bagging will simply move tuning the parameter to 0.
- But feature categorical approach should work.
- Score: so far the quadratic term has worked best with ~1460000 RMSE on the test partition and RMSE of ~2000000 on Kaggle  
- File: ./Bagging with Feature Combination/train.m
- Name/Date: Wes 4/12/15

5th
- Approach: convert the problem to a classify probelm and use fitctree to do classify.
- Does not work very well. So I would use fitensemble to do classify.
- Score: so far the quadratic term has worked best with ~90000 RMSE on the test partition and RMSE of ~2400000 on Kaggle  
- File: ./Classify Approach/train.m
- Name/Date: Wes 4/12/15

6th
- Approach: convert the problem to a classify probelm and use fitensemble to classify.
- It seems working better than fitctree, but still not good enough.
- Score: so far the quadratic term has worked best with ~1.60000 RMSE on the test partition and RMSE of ~1.900000 on Kaggle  
- File: ./Classify Fitensemble/train.m
- Name/Date: Wes 4/13/15

7th
- Approach: bagging with linear x2fx and fresample of 0.5
- Bagging seems to be the best approach while working with x2fx, which makes me suspect that if there is something we can do to expand the dimension.
- Moreover, we can start use this approach to deal with seperate case, like big city vs other city.
- Score: so far the quadratic term has worked best with ~2.40000 RMSE on the test partition and RMSE of 1707777.01698 on Kaggle
- File: ./Will Continue/train.m
- Name/Date: Wes 4/13/15

8th
- Approach: cluster city and do two bagging for big city/other city
- The problem for this apprach is the training sample gets much smaller after seperating cases.
- The RMSE for other city is about ~0.9 M while for big city is about ~9 M, so maybe we can use the result for other city and use the big city result from 7th approach.
- Score: RMSE of ~1.8 M on Kaggle
- File: ./City Cluster/train.m
- Name/Date: Wes 4/13/15

9th
- Approach: Combine the result from citycluster(train0) and and will continue(train1).
- Score: RMSE of 1732591.32258 on Kaggle under name kingjim.
- File: ./Combine City and Will/train.m
- Name/Date: Wes 4/14/15

10th
- Approach: Basically inherit the will continue training model but set the outliner revunue (>1.2e7) to 1.2e7 and retrain again.
- Score: RMSE of 1700507.32258 on Kaggle under name kingjim.
- File: ./Get Rid of Outline/train.m
- Name/Date: Wes 4/14/15

11th
- Approach: cluster city and do two bagging for big city/other city.
- Different from approach 8, this time I set boundary for outliner revenue
- The RMSE for other city is about ~0.9 M while for big city now is only about ~6 M.
- Score: RMSE of ~1.77 M on Kaggle, which is not bad, but still can be improved. So making we can look into other features.
- File: ./City Cluster Continue/train.m
- Name/Date: Wes 4/14/15

12th
- Approach: Basically inherit the will continue training model but set the outliner revunue (>1e7) to 1e7 and retrain again.
- Basically I only change the threshold
- Score: RMSE of 1691707 on Kaggle.
- File: ./Get Rid of Outline Continue/train.m
- Name/Date: Wes 4/14/15

13th
- Continue from 12, I go one step further and change the threshold to 0.95e7
- Score: RMSE of 1698759 on Kaggle, which is close to 12.
- File: ./Get Rid of Outline Continue/train.m
- Name/Date: Wes 4/14/15

14th
- Support Vector approach, libvim linbrary, using linaer kenal
- Try several tunning option, the score is not very impressive. Might move on and try something else.
- Score: RMSE of ~1.8M on Kaggle.
- File: ./SVM/train.m
- Name/Date: Wes 4/15/15

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
