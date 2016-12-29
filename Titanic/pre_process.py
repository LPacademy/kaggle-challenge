# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 22:32:16 2016

@author: takashi
"""

import pandas as pd
pd.set_option("expand_frame_repr", False)
pd.set_option(      "max_columns",    16)
pd.set_option(         "max_rows",     8)

FILE_PATH_TRAIN_DATA    = "train.csv"
FILE_PATH_PREDICT_DATA  = "test.csv"
KEYS_FOR_ML        = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
KEYS_FOR_NORM      = [                        "SibSp", "Parch", "Fare"]
BINARIZED_CAT_DICT = {
                "Sex":      ["male", "female"],
                "Pclass":   [ 1,  2,  3],
                "Embarked": ["C","Q","S"],
                }


def load_titanic_csv(csv_file_path):
    """
    Import a Titanic csv file as Pandas DataFrame instance.
    Parameters:
        csv_file_path: file path of data for Kaggle Titanic Compe
    Returns:
        raw_df: 
    """
    raw_df = pd.read_csv(csv_file_path, header=0, index_col="PassengerId")
    if not "Survived" in raw_df.keys():
        raw_df["Survived"] = None
    
    return raw_df

    
def txt_w_border(txt):
    BORDER_CHARS = 15 * "#"
    ret_txt = " ".join(["\n\n", BORDER_CHARS, txt, BORDER_CHARS])
    return ret_txt
    
    
def print_info(df, start=0, end=4):
    """
    Print some information of the input dataframe.
    Parameters:
        df: Dataframe to print its info
        start: start index for sample
        end: last index for sample
    Returns:
        sample_df: The first max 10 dataframe
    """
    print txt_w_border("Info")
    print df.info()
    
    print txt_w_border("Statistics")
    print df.describe()
    
    try:
        sample_df = df[start:end].copy()
        print txt_w_border("Samples, " + str(start) + " to " + str(end))
    except:
        sample_df = df.head()
        print txt_w_border("Samples, df.head")
    print sample_df
    
    return sample_df

    
def filter_cols(df):
    """
    Pick up by keyword list.
    Parameters:
        df: Dataframe
        (Option) keys: Keyword list
    Returns:
        filt_col_df: Filtered dataframe
    """
    comm_keys = list( set(df.keys()) & set(KEYS_FOR_ML) )
    filt_col_df = df.copy()[comm_keys]
    
    return filt_col_df
    
    
def binarize(df):
    """
    Binarize categorial dataframe.
    Parameters:
        df: Dataframe
    Returns:
        out_df: Numerized dataframe
    """
    comm_keys = list( set(df.keys()) & set(BINARIZED_CAT_DICT.keys()) )
    ret_df = df.copy()
    for key in comm_keys:
        val = BINARIZED_CAT_DICT[key]
        ret_df[key] = ret_df[key].astype("category")
        ret_df[key] = ret_df[key].cat.set_categories(val)
        
    ret_df = pd.get_dummies(ret_df, columns=comm_keys, drop_first=True)
    
    return ret_df

    
def normarize(df, **kw):
    """
    Nomarize dataframe with skipping NaN, then fill NaN with zero.
    Parameters:
        df: Dataframe
        (Option) key_for_fillna
    Returns:
        out_df: Nomarized dataframe
    """
    comm_keys = list( set(df.keys()) & set(KEYS_FOR_NORM) )
    
    try:
        t = df[comm_keys].fillna(kw["key_for_fillna"])
    except:
        t = df[comm_keys]
    finally:
        ret_df = df.copy()
        ret_df[comm_keys] = (t - t.mean()) / t.std()
    
    return ret_df
    


def pre_process_all(csv_file_path):
    """
    Preprocess the followings at once.
    - Load csv file in csv_file_path.
    - Filter attributes according to KEYS_FOR_ML.
    - Binarize the
    Parameters:
        csv_file_path: File path of which the file to be imported
        KEYS_FOR_ML: List of index of dataframe to pick up
    Returns:
        norm_df: Processed dataframe
    """
    print txt_w_border("Import " + csv_file_path)
    raw_df       = load_titanic_csv(csv_file_path)
    print_info(raw_df)
    
    print txt_w_border("Filter columns in " + csv_file_path)
    filt_col_df  = filter_cols(raw_df)
    print filt_col_df.head()
    
    print txt_w_border("Binarize " + str(BINARIZED_CAT_DICT.keys()) + " of " + csv_file_path)
    binarized_df = binarize(filt_col_df)
    print_info(binarized_df)
    
    print txt_w_border("Normarize " + str(KEYS_FOR_NORM) + " of " + csv_file_path)
    norm_df = normarize(binarized_df)
    print_info(norm_df)
    
    return norm_df

def extract_train_target(csv_file_path):
    # Prepare explanatory / predictor dataframe and objective / predicted dataframe
    explanatory_df = pre_process_all(csv_file_path)
    target_df = explanatory_df[["Survived"]].copy()
    explanatory_df.drop("Survived", axis=1, inplace=True)
    
    return explanatory_df, target_df
    
if __name__=="__main__":
    train_df, train_target_df = extract_train_target(FILE_PATH_TRAIN_DATA)
    test_df,       predict_df = extract_train_target(FILE_PATH_PREDICT_DATA)

    