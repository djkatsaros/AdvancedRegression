""" Predicts a target property from a given data set. Performs data cleaning using basic imputation
if most data isn't missing, or machine learning (random forest) if a lot of data is missing.
Reports cross-validation values for the models giving predictinos.
See main for description of inputs.
**based on work in the kaggle housing prices competition**
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
import re

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import seaborn as sns
import matplotlib.pyplot as plt

import sys

sys.path.append('/Users/deankatsaros/Desktop/housing_prices_kaggle/')

def main(data: str,test: str, toPredict: str, minToDrop: float, maxToImpute: float ):
    """inputs:
    > train_data and test_data paths to datasest 
    > toPredict: parameter to predict
    > minToDrop: If more than this percentage missing, drop the data column
    > maxToImpute: If less than this is missing, use basic imputation to fill in missing values
    Performs the data cleaning, missing value imputation/prediction, and model prediction for the dataset.
    """
    from data_cleaning import drop_nulls, collect_missing, basic_imputer, predictions, predictions_num, predict_missing_vals

    #Data cleaning:
    data, tooMuchMissing = drop_nulls(data, minToDrop)
    cols_missing, num_missing, cat_missing = collect_missing(data, maxToImpute)
    # Plot data heatmap, examine how much is missing after dropping high missing values 
    plt.figure(figsize=(16, 6))
    sns.heatmap(data.isnull(), cbar=False)
    plt.title('Missing values before imputation')
    plt.show()   
    print("cols with more than {} missing values:".format(minToDrop), tooMuchMissing)
    imputer_object=SimpleImputer(strategy = "most_frequent") # categ. imputer 
    imputer_num = IterativeImputer(estimator=RandomForestRegressor()) # numerical imputer 
    data = basic_imputer(data, cols_missing, imputer_object, imputer_num) # impute missing values
    nulls = (data.isnull().sum() / len(data) * 100).sort_values(ascending = False)
    cols_with_high_missing = list(nulls[nulls > 0].index) #columns with more than 10% and less than 60% missing.
    print("After dropping columns missing {0} percent or more of their data, and imputing values based on the most frequent values in the dataset in columns missing less than {1} percent of their data, the remaining columns with missing data are {2}".format(minToDrop,maxToImpute,cols_with_high_missing) )
    
    # Create dataframe of missing values
    data_clean = data.dropna()
    data_missing = data[data.isna().any(axis=1)]
    
    # list of columns to drop, with target at the end.
    drop_cols = cols_with_high_missing
    drop_cols.append(toPredict)
    scaler = RobustScaler()
    models = {}    # store models
    # predict missing values for columns with high missing
    for col in drop_cols[:-1]:
        if data_clean[col].dtype == 'object':
            models[col] = predictions(data_clean, col, drop_cols)
        if data_clean[col].dtype != 'object':
            models[col] = predictions_num(data_clean, col, drop_cols, scaler)
    
    #missing_vals = {}
    for j in range(len(drop_cols)-1):
        drop_temp = drop_cols[0:j] + drop_cols[j+1:len(drop_cols)]
        predict_missing_vals(data_missing, models[drop_cols[j]], drop_cols[j],
                                                 drop_temp )
    # Check for missing values by plotting heatmap.    
    plt.figure(figsize=(16, 6))
    sns.heatmap(data_missing.isnull(), cbar=False)
    plt.title('Missing values after imputation')
    plt.show()

    # Concatenate clean and missing dataframe into one dataframe. 
    data_complete = pd.concat([data_missing, data_clean], axis=0)
    print("The shape of the completed dataframe is: ", data_complete.shape)
    df = data_complete
    print("Cleaned data has the following numbers of null values:")
    df.isnull().sum()
    df[toPredict] = np.log(df[toPredict])
    X = df.drop(toPredict, axis=1)
    y = df[toPredict]

    scaler = MinMaxScaler()
    encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype != "object":   
            X[col] = scaler.fit_transform(X[[col]]) # if numeric, scaler takes a dataframe 
        if X[col].dtype == "object":
            X[col] = X[col].astype(str)
            X[col] = encoder.fit_transform(X[col]) # if categoric, encoder takes a series
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #Train test split 
    # Set parameters for catboost regressor 
    params = {'random_strength': 1,
        'n_estimators': 100,
        'max_depth': 7,
        'loss_function': 'RMSE',
        'learning_rate': 0.1,
        'colsample_bylevel': 0.8,
        'bootstrap_type': 'MVS',
        'bagging_temperature': 1.0}

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train) # fit model
    y_pred = model.predict(X_test) # predict values for the test set

    #print errors:
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("R2: ", r2_score(y_test, y_pred))
    print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
    # plot predictions vs true values
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='black', linewidth=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Predictions vs. True Values")
    plt.show()
    
    df_test=test.drop(tooMuchMissing, axis=1)
    (df_test.isnull().sum()/len(df_test) * 100).sort_values(ascending=False)
    test_cols_missing, test_num_missing, test_cat_missing = collect_missing(df_test, maxToImpute)
    df_test = basic_imputer(df_test, test_cols_missing, imputer_object, imputer_num) # impute missing values
    test_nulls = (df_test.isnull().sum() / len(df_test) * 100).sort_values(ascending = False)
    test_cols_with_high_missing = list(test_nulls[test_nulls > 0].index) #columns with more than 10% and less than 60% missing.
    test_drop_cols = test_cols_with_high_missing
    #test_drop_cols.append(toPredict)
    for j in range(len(test_drop_cols)-1):
        test_drop_temp = test_drop_cols[0:j] + test_drop_cols[j+1:len(test_drop_cols)]
        predict_missing_vals(df_test, models[test_drop_cols[j]], test_drop_cols[j],
                                                 test_drop_temp )
    # ecode/ scale test set data
    for col in df_test.columns:
        if df_test[col].dtype == 'object':
            df_test[col] = encoder.fit_transform(df_test[col])
        if df_test[col].dtype != 'object':
            df_test[col] = scaler.fit_transform(df_test[[col]])

    log_predictions = model.predict(df_test)
    predictions_exp = np.exp(log_predictions)

if __name__ == "__main__":
    # import training and test set
    data_name = input("Enter path to training dataset: ")
    test_name = input("Enter path to test data: ")
    try:
        data=pd.read_csv(data_name)
    except FileNotFoundError or pandas.errors.EmptyDataError:
        data_name = input("Path to data not found or not a csv file. Enter correct file path: ")
    try:
        test=pd.read_csv(test_name)
    except FileNotFoundError or pandas.errors.EmptyDataError:
        test_name = input("Path to data not found or not a csv file. Enter correct file path: ")
        
    # Set working parameters
    toPredict = input("Enter quantity to predict: ")
    try:
        data[toPredict]
    except KeyError:
        toPredict = input("Quantity not a part of dataset, enter a quantity to predict from the specified data: ")
    
    minToDrop = float(input("Enter threshold above which to drop data: ")) # if more than minToDrop missing, drop data.
    maxToImpute = float(input("Enter threshold below which to use basic imputation: ")) # if less than maxToImpute missing, use basic imputer to fill in missing values
    main(data,test, toPredict, minToDrop, maxToImpute)


