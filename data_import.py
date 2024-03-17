import pandas as pd
def _data_import(path: str, train_set: str, test_set: str):

    """
    Imports train/test set from a specified path and names of test_set and train_set.
    Returns data and test set, data in first entry and test set in the second entry.
    Handles the case where the file is not found. 
    """
    #Make path names
    data_path = path + train_set
    test_path = path + test_set
    #Data import
    try:
        data=pd.read_csv(data_path)
        test=pd.read_csv(test_path)
        return(data,test)
    except FileNotFoundError:
        print("File or directory not found, or path name incorrect")


    
