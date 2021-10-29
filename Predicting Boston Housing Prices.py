# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:39:32 2021
@author: Eriny
"""

import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from IPython import get_ipython
# Pretty display 
get_ipython().run_line_magic('matplotlib', 'inline')

### Reading the data, and separating the features and prices for homes into different pandas dataframes.
## Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']                  ## Target Variable (y)   = (MEDV)
features = data.drop('MEDV', axis = 1) ## Features (x1, x2, x3) = (RM, LSTAT, PTRATIO)
## Success
#print('Boston housing dataset has {0} data points with {1} variables each'.format(*data.shape))

## Features Observation
vs.FeatureObservation(data, features, prices)

### Data Exploration
## Calculating Statistics like (max, min, mean, median, std, percentiles)
#minimum_price = np.min(prices)   ## Minimum prices
# minimum_price = prices.min()   ## Alternative using pandas

#maximum_price = np.max(prices)   ## Maximum prices
# maximum_price = prices.max()   ## Alternative using pandas

#mean_price = np.mean(prices)     ## Mean of the prices
# mean_price = prices.mean()     ## Alternative using pandas

#median_price = np.median(prices) ## Median of the prices
# median_price = prices.median() ## Alternative using pandas

#std_price = np.std(prices)       ## Standard deviation of the prices
# std_price = prices.std(ddof=0) ## Alternative using pandas 

#first_quartile = np.percentile(prices, 25)
#third_quartile = np.percentile(prices, 75)
#inter_quartile = third_quartile - first_quartile

## Show the calculated statistics
#print("Statistics for Boston housing dataset:\n")
#print("Minimum price: ${:,.2f}".format(minimum_price))
#print("Maximum price: ${:,.2f}".format(maximum_price))
#print("Mean price: ${:,.2f}".format(mean_price))
#print("Median price ${:,.2f}".format(median_price))
#print("Standard deviation of prices: ${:,.2f}".format(std_price))
#print("First quartile of prices: ${:,.2f}".format(first_quartile))
#print("Second quartile of prices: ${:,.2f}".format(third_quartile))
#print("Interquartile (IQR) of prices: ${:,.2f}\n".format(inter_quartile))

    
### Developing a Model

## The coefficient of determination for a model: Is a useful statistic in regression analysis, 
##                                               as it often describes how “good” that model is 
##                                               at making predictions.
## Define a Performance Metric
def PerformanceMetric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """

    """                  Mean Squared Error for the linear regression model
    r2_score(): R² = 1 - __________________________________________________
                         Mean Squared Error for the simple model
    The values for R2 range from 0 to 1, which captures the percentage of squared correlation between 
    the predicted and actual values of the target variable. 
    A model with an R2=0 is no better than a model that always predicts the mean of the target variable, 
    a model with an R2=1 perfectly predicts the target variable. 
    Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the features.
    """
    score = r2_score(y_true, y_predict)
    return score


## Calculate the performance of this model
score = PerformanceMetric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3]) ## True Value, Prediction
print("This model has a coefficient of determination, R²: {:.3f}\n".format(score))


## Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10, shuffle=True)
## Success
#print("Training and testing split was successful.")

## Check if split is actually correct -- 80% train and 20% train
#print("Training-Set / Dataset", float(X_train.shape[0]) / float(features.shape[0]))
#print("Testing-Set / Dataset", float(X_test.shape[0]) / float(features.shape[0]))
#print("Rows of X(Features):",features.shape[0], "Rows")

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)

vs.ModelComplexity(X_train, y_train)


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # ShuffleSplit works iteratively compared to KFOLD
    # It saves computation time when your dataset grows
    # X.shape[0] is the total number of elements
    # n_splits is the number of re-shuffling & splitting iterations.
    cross_validation_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0) 

    # Create a DecisionTreeRegressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = dict(max_depth = range(1, 11))

    # Transform 'PerformanceMetric' into a scoring function using 'make_scorer' 
    # We initially created PerformanceMetric using R2_score
    scoring_fnc = make_scorer(PerformanceMetric)

    # Create the grid search object
    # You would realize we manually created each, including scoring_func using R^2
    grid = GridSearchCV(regressor, params, cv=cross_validation_sets, scoring=scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    ## Store the predicted prices
    prices = []

    for k in range(10):
        ## Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = k)

        ## Fit the data
        reg = fitter(X_train, y_train)

        ## Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)

        ## Result
        print("\t\tTrial {}: ${:,.2f}".format(k+1, pred))

    ## Display price range
    print("*** Range in predicted prices: ${:,.2f} ***\n".format(max(prices) - min(prices)))


## How we got the 'max_depth' param?
## Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' = {} for the optimal model.\n".format(reg.get_params()['max_depth']))
## or reg.get_params()['max_depth'] --> We can access our value from reg.get_params(), a dictionary, using dict['key']


# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

vs.DrawHistogram(prices, reg, client_data)

## 
PredictTrials(features, prices, fit_model, client_data)

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
   
    
'''
from sklearn.model_selection RandomizedSearchCV
def fit_model_2(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # ShuffleSplit works iteratively compared to KFOLD
    # It saves computation time when your dataset grows
    # X.shape[0] is the total number of elements
    # n_iter is the number of re-shuffling & splitting iterations.
    cross_validation_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a DecisionTreeRegressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = dict(max_depth = range(1, 11))

    # Transform 'PerformanceMetric' into a scoring function using 'make_scorer' 
    # We initially created PerformanceMetric using R2_score
    scoring_fnc = make_scorer(PerformanceMetric)

    # Create the grid search object
    # You would realize we manually created each, including scoring_func using R^2
    rand = RandomizedSearchCV(regressor, params, cv=cross_validation_sets, scoring=scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    rand = rand.fit(X, y)

    # Return the optimal model after fitting the data
    return rand.best_estimator_
'''
 
"""
# Import NearestNeighbors
from sklearn.neighbors import NearestNeighbors

# Set number of neighbors
num_neighbors=5

def nearest_neighbor_price(x):
    # x is your vector and X is the data set.
    def find_nearest_neighbor_indexes(x, X):
        # Instantiate
        neigh = NearestNeighbors(num_neighbors)
        # Fit
        neigh.fit(X)
        distance, indexes = neigh.kneighbors(x)
        return indexes
        # This returns, the position, say for example [4, 55, 22]
        # array([[357, 397, 356, 141, 395]])
    indexes = find_nearest_neighbor_indexes(x, features)
    # Create list
    sum_prices = []
    # Loop through the array
    for i in indexes:
        # Append the prices to the list using the index position i
        sum_prices.append(prices[i])
    # Average prices
    neighbor_avg = np.mean(sum_prices)
    # Return average
    return neighbor_avg

# Test if it's working with a list [4, 55, 22]
arr_test = np.array([4, 55, 22]).reshape(1, -1)
print(nearest_neighbor_price(arr_test))

# client_data = [[5, 17, 15], # Client 1
               #[4, 32, 22], # Client 2
               #[8, 3, 12]]  # Client 3

# Loop through data, this is basically doing the following
# print(nearest_neighbor_price([5, 17, 15]))
# print(nearest_neighbor_price([4, 32, 22]))
# print(nearest_neighbor_price([8, 3, 12]]))
index = 0
for i in client_data:
    arr = np.array(i).reshape(1, -1)
    val=nearest_neighbor_price(arr)
    index += 1
    # num_neighbours is constant at 5
    # index changes from 1 to 2 to 3
    # value changes respectively from $372,540.00 to $162,120.00 to $897,120.00
    print("The predicted {} nearest neighbors price for home {} is: ${:,.2f}".format(num_neighbors,index, val))
"""