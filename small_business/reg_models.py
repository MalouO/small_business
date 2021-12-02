#Imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import math
import itertools
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Pipe import pipe


#Import preprocessed data from other file

X, y, X_test, y_test = pipe()
#buckets rating
# Linear Regression
def linearReg(X, y):
    model = sm.OLS(y, X).fit(X, y) # Finds the best beta
    return model.summary(), model

# LinearRegression with ElasticNet
def LinRegElastic(X, y):
    alphas = [0.01, 0.1, 1]
    l1_ratios = [0.2, 0.5, 0.8]
    hyperparams = itertools.product(alphas, l1_ratios)
    for hyperparam in hyperparams:
        alpha = hyperparam[0]
        l1_ratio = hyperparam[1]
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        r2 = cross_val(model, X, y, cv=5).mean()
        print(f"alpha: {alpha}, l1_ratio: {l1_ratio},   r2: {r2}")

# Lasso, Ridge
def feature_perm(X, y, alpha = 0.2):
    ridge = Ridge(alpha=alpha).fit(X, y)
    lasso = Lasso(alpha=alpha).fit(X, y)
    return cross_val(ridge, X, y, cv=5, scoring=['r2', 'mse']), cross_val(lasso, X, y, cv=5, scoring=['r2', 'mse'])

# SGD Regression
def SGDReg(X, y, learning_rate, loss='mse', penalty='l2', alpha=0.0001, l1_ratio=0.15,
    max_iter = 1000, random_state= 42, *args, **kwargs):

    sgd_reg = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
    max_iter = max_iter, random_state= random_state,
    learning_rate=learning_rate).fit(X, y)
    sgd_model_cv = cross_val(sgd_reg,
                              X,
                              y,
                              cv = 5,
                              scoring = ['r2','mse'] )
    return sgd_model_cv, sgd_reg

# KNN Regressor
def best_k( X, y):
    score = []
    neighbours = []

    for k in range(1,25):

        # Instanciate the model
        knn_model = KNeighborsRegressor(n_neighbors = k)

        # Train the model on the scaled Training data
        cv_results = cross_val(knn_model, X ,y)

        # Append the score and k
        score.append(cv_results['test_score'].mean())
        neighbours.append(k)
    best = dict(zip(neighbours, score))
    best_k  = min(best, key=best.get)
    return best_k

def KNNReg(X, y, best_k, *args, **kwargs):
    knn_reg = KNeighborsRegressor(n_neighbors=best_k, n_jobs = -1).fit(X, y)
    knn_reg_cv = cross_val(knn_reg,
                              X,
                              y,
                              cv = 5,
                              scoring = 'mse')
    return knn_reg_cv, knn_reg

#SVM
def SVMReg(X, y):
    svm_reg = SVR(epsilon=0.1, C=1, kernel='linear').fit(X, y)
    svm_reg_cv = cross_val(svm_reg,
                              X,
                              y,
                              cv = 5,
                              scoring = 'mse')
    return svm_reg_cv, svm_reg

# cross_val
def cross_val(model, X, y, cv=5, scoring=['r2', 'mse'], *args, **kwargs):
    cv_results =  cross_validate(model, X, y, cv=cv, scoring=scoring,  n_jobs =-1)
    return pd.DataFrame(cv_results)

# Grid Search
def GridSearch(model, grid, X, y):
    search = GridSearchCV(model, grid,
                           scoring = 'r2',
                           cv = 5,
                           max_iter = 1000,
                           n_jobs =-1
                          )
    search.fit(X, y)
    return search.best_score_ , search.best_params_ , search.best_estimator_

# Learning Curves
def learning_curves(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator = model,
                                                                X = X,
                                                                y = y,
                                                                train_sizes = [50,100,250,500,750,1000,1250],
                                                                cv = 10,
                                                                scoring = 'r2')
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training score')
    plt.plot(train_sizes, test_scores_mean, label = 'Test score')
    plt.ylabel('r2 score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves', fontsize = 18, y = 1.03)
    plt.legend()
    return plt.show()

def metrics(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    rsquared = r2_score(y, y_pred)
    print('MSE =', round(mse, 2))
    print('RMSE =', round(rmse, 2))
    print('MAE =', round(mae, 2))
    print('R2 =', round(rsquared, 2))

# Decision Tree
def decisionTreeReg(X, y):
    tree = DecisionTreeRegressor()
    tree_cv = cross_val(tree, X, y, scoring = "r2", cv=5)
    return tree_cv, tree

# Random Forest

def randForestReg(X, y):

    forest = RandomForestRegressor(n_estimators=100)
    forest_cv = cross_val(forest, X, y, scoring = "r2", cv=5)
    return forest_cv, forest
#Boosters
def adaBoost(X, y):
    adaboost = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=3),
        n_estimators=50)
    ada_cv = cross_val(adaboost, X, y, scoring = "r2", cv=5)
    return ada_cv, adaboost

def GradBoost(X, y):
    gradboost = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3)
    gradboost_cv = cross_val(gradboost, X, y, scoring = "r2", cv=5)
    return gradboost_cv, gradboost

def XGBReg():
    XGB_Reg= XGBRegressor(max_depth=10, n_estimators=100, learning_rate=0.1)

def get_prediction(model, input):
    return model.predict(input)
