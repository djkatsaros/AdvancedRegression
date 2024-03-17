import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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

#import imputers.
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import seaborn as sns
import matplotlib.pyplot as plt

import sys

sys.path.append('/Users/deankatsaros/Desktop/housing_prices_kaggle/')

def drop_nulls(data, threshold: float):
    """ drops data where more than *threshold*, where threshold is a percentage, of
    the data is missing
    inputs:
    dataframe: data
    a number: threshold 
        the cutoff below which data is not deleted.
    returns:
    dataframe data with missing data deleted.
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
    Inputs: 
    a dataframe: data, 
    A number: threshold, 
        which is the cutoff for collecting the col name
    returns:
    lists of the missing columns, a list of the missing columns with numerical data, and 
    a list of missing columns with categorical data
    """
    cols_missing = [col for col in data.columns 
                    if (data[col].isnull().sum() / len(data) * 100) < threshold
                        and ((data[col].isnull().sum() / len(data)*100) > 0)]

    #missing <threshold numerical type
    num_missing = [col for col in cols_missing if data[col].dtype != 'object']

    #missing <threshold and categorical type
    cat_missing = [col for col in cols_missing if data[col].dtype == 'object']

    return cols_missing, num_missing, cat_missing

def basic_imputer(data, cols_missing, imputer_object, imputer_num):
    """implements a basic imputer for a list of column names for data.
    uses a different imputer if data is categorical vs numerical. 
    inputs:
    dataframe: data
    list of missing columns to impute: cols_missing
    categorical imputer: imputer_object
    numerical imputer: imputer_num
    returns:
    dataframe data with imputed values.
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
    a column to predict: target,
    and a list of columns to drop: drop_c.
    note that data cannot contain missing values.
    Returns a random forest classifier fitted to the data and target. 
    prints cross validation scores using k-fold cross validation. 
    """
    X = data.drop(drop_c, axis = 1)
    y = data[target]
        
    le = LabelEncoder()
    scaler = MinMaxScaler()
    
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
    and a list of columns to drop: drop_c.
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
    inputs:
    dataframe: data_
    model to predict with: model
    target to predict: target
    columns to drop: drop_c
        where the columns are those missing alot of values.
    returns:
    dataframe data_ with predicted values in the columns in 
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


if __name__ == "__main__":
    # import training and test set
    data=pd.read_csv('/Users/deankatsaros/Desktop/housing_prices_kaggle/train.csv')
    test=pd.read_csv('/Users/deankatsaros/Desktop/housing_prices_kaggle/test.csv')
    # Set working parameters
    toPredict = "SalePrice"
    minToDrop = 60   # if more than 60% missing, drop data.
    maxToImpute = 10 # if less than 10% missing, use basic imputer to fill in missing values

    #Data cleaning:
    data, tooMuchMissing = drop_nulls(data, minToDrop)
    cols_missing, num_missing, cat_missing = collect_missing(data, maxToImpute)

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
    #return predictions_exp
