""" Collection of functions for dataset cleaning.
Includes methods for dropping null values, basic imputation, collecting
missing values, and using random forest classifiers to fill in missing
numerical and categorical data.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/Users/deankatsaros/Desktop/housing_prices_kaggle/datasets/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.preprocessing import MinMaxScaler

# Machine learning
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostRegressor

# preprocessing scaler
from sklearn.preprocessing import RobustScaler

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

#import imputers 
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def drop_nulls(data, threshold: float):
    """ drops data where more than *threshold*, where threshold is a percentage, of
    the data is missing
    -------------------------
    inputs:
      > dataframe: data
      > a number: threshold
        the cutoff below which data is not deleted.
    -------------------------   
    returns:
      > dataframe data with missing data deleted.
    """
    # get a series of the percentages of null values
    # make a list of the column names with more than
    # the threshold worth of missing values
    # Drop these columns
    nulls = (data.isnull().sum() / len(data)*100).sort_values(ascending=False)
    null_cols=list(nulls[nulls >= threshold].index)
    data=data.drop(null_cols, axis=1)
    return data, null_cols

def collect_missing(data, threshold: float):
    """Collects column names for columns missing less data but who are 
    missing less than a threshold of the data. 
    -------------------------  
    Inputs: 
      > dataframe: data, 
      > A number: threshold, 
        which is the cutoff for collecting the col name
    returns:
    - - - - - - - - - - - - - - - - - - - - - - - - -  
      > list of the missing columns, a list of the missing columns with numerical data, and 
    a list of missing columns with categorical data
    """
    cols_missing = [col for col in data.columns
                    if (data[col].isnull().sum() / len(data) * 100) < threshold
                        and ((data[col].isnull().sum() / len(data)*100) > 0)]

    #missing less than threshold numerical type
    num_missing = [col for col in cols_missing if data[col].dtype != 'object']

    #missing less than threshold and categorical type
    cat_missing = [col for col in cols_missing if data[col].dtype == 'object']

    return cols_missing, num_missing, cat_missing

def basic_imputer(data, cols_missing, imputer_object, imputer_num):
    """implements a basic imputer for a list of column names for data.
    uses a different imputer if data is categorical vs numerical. 
    - - - - - - - - - - - - - - - - - - - - - - - - -
    inputs:
      > dataframe: data
      > list of missing columns to impute: cols_missing
      > categorical imputer: imputer_object
      > numerical imputer: imputer_num
    - - - - - - - - - - - - - - - - - - - - - - - - -  
    returns:
      > dataframe data with imputed values.
    """
    for col in cols_missing:
        if data[col].dtype == 'object':
            data[col] = imputer_object.fit_transform(data[[col]]).squeeze()
        else:
            data[col] = imputer_num.fit_transform(data[[col]])
    return data

def predictions(data, target, drop_c):
    """
    Takes in a pandas dataframe: data, 
    a column to predict consisting of categorical data: target,
    and a list of columns to drop: drop_c.
    note that data cannot contain missing values.
    Returns a random forest classifier fitted to the data and target. 
    prints cross validation scores using k-fold cross validation. 
    """
    X = data.drop(drop_c, axis = 1)
    y = data[target]

    le = LabelEncoder()
    scaler = MinMaxScaler()

    # encodes columns values to train on the model
    for col in X.columns:
        if X[col].dtype != 'object':
            X[col] = le.fit_transform(X[[col]]) # calls the column as a data frame if numerical, as numeric labels. 
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col]) # calls colunmn as data series if object. Le.fit_transform takes categories-> numeric labels

    model = RandomForestClassifier()
    model.fit(X,y)
    #cross validate with k-fold cross-validation.
    kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print("Model generated fitting cleaned data to column {}.".format(target))
    print("Scores for each fold:", scores)
    print("Mean score:", scores.mean())
    return model

def predictions_num(data, target, drop_c, scaler):
    """ 
    Takes in a pandas dataframe: data, 
    a column to predict: target,
    a scaler to scale the data for use in the classifier,
    a list of columns to drop: drop_c.
    note that data cannot contain missing values.
    Returns a random forest regressor fitted to the data and target. 
    prints cross validation scores using k-fold cross validation. 
    """
    X = data.drop(drop_c, axis = 1)
    y = data[target]

    le = LabelEncoder()

    for col in X.columns:
        if X[col].dtype != 'object':
            X[col] = scaler.fit_transform(X[[col]]) # as dataframe -> scales
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col])

    model = RandomForestRegressor()
    model.fit(X, y)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print("Model generated fitting cleaned data to column {}.".format(target))
    print("Scores for each fold:", scores)
    print("Mean score:", scores.mean())
    
    return model

def predict_missing_vals(data_, model, target, drop_c = []):
    """predicts the missing values in specified columns using the inputed model.
   - - - - - - - - - - - - - - - - - - - - - - - -   
    inputs:
      > dataframe: data_
      > model to predict with: model
      > target to predict: target
      > lsit of columns to drop: drop_c
        where the columns are those missing alot of values.
   - - - - - - - - - - - - - - - - - - - - - - - - -   
    returns:
      > dataframe data_ with predicted values in the columns in
    """
    # encode dataframe with label encoder.
    df_encoded = data_.drop(drop_c + [target], axis=1, errors='ignore')
    le = LabelEncoder()
    for col in df_encoded.columns:
        if df_encoded[col].dtype != 'object':
            df_encoded[col] = le.fit_transform(df_encoded[[col]])
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col])

    # Predict missing values
    features_to_predict = df_encoded.drop([target], axis=1, errors = 'ignore')
    y_pred = model.predict(features_to_predict)
    # change missing values in the original dataframe
    data_[target] = y_pred

    return data_
