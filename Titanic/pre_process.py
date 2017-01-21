#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


import numpy as np
import pandas as pd
pd.set_option("expand_frame_repr", False)
pd.set_option(      "max_columns",    16)
pd.set_option(         "max_rows",     8)

FILE_PATH_TRAIN_DATA    = "train.csv"
FILE_PATH_PREDICT_DATA  = "test.csv"
#KEYS_FOR_ML        = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
KEYS_FOR_ML        = ["Survived", "Pclass", "Sex",  "Age", "SibSp", "Parch", "Fare"]
KEYS_FOR_NORM      = [                              "Age", "SibSp", "Parch", "Fare"]
BINARIZED_CAT_DICT = {"Sex":      ["male", "female"],
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
        raw_df["Survived"] = np.NaN
    
    return raw_df

    
def txt_w_border(txt):
    BORDER_CHARS = 15 * "#"
    ret_txt = " ".join(["\n", BORDER_CHARS, txt, BORDER_CHARS])
    return ret_txt
    
    
def print_info(df, start=0, end=4, details=1):
    """
    Print some information of the input dataframe.
    Parameters:
        df: Dataframe to print(its info
        start: start index for sample
        end: last index for sample
    Returns:
        sample_df: The first max 10 dataframe
    """
    print(txt_w_border("Info"))
    print(df.info())
    
    if details>1:
        print(txt_w_border("Statistics"))
        print(df.describe(percentiles=[]))
    
    if details>2:
        try:
            sample_df = df[start:end].copy()
            print(txt_w_border("Samples, " + str(start) + " to " + str(end)))
        except:
            sample_df = df.head()
            print(txt_w_border("Samples, df.head"))
        print(sample_df)
    
    return None

    
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
        
    ret_df = pd.get_dummies(ret_df, columns=comm_keys, drop_first=True).astype("float")
    
    return ret_df

    
def normalize(df):
    """
    Nomarize dataframe without NaN.
    Fill NaN with kw["key_of_fillna"].
    Add NaN col
    Parameters:
        df: Dataframe
        (Option) key_of_fillna
    Returns:
        out_df: Nomarized dataframe
    """
    comm_keys = list( set(df.keys()) & set(KEYS_FOR_NORM) )
    
    ret_df = df.copy()
    t = ret_df[comm_keys]
    ret_df[comm_keys] = (t - t.mean()) / t.std()
    
    return ret_df
    

def add_null_flag_cols(df, del_single_cat_cols=False):
    keys_of_null_cols = [ k + "_null" for k in df.keys()]
                         
    ret_df = df.copy()
    ret_df[keys_of_null_cols] = ret_df[df.keys()].isnull()
    if del_single_cat_cols:
        ret_df = pd.get_dummies(ret_df, columns=keys_of_null_cols, drop_first=True)
    
    return ret_df
    

def merge_SibSp_Parch_to_FamSize(df):
    ret_df = df.copy()
    ret_df["FamSize"] = ret_df["SibSp"] + ret_df["Parch"]
    ret_df.drop(["SibSp","Parch"], axis=1, inplace=True)
    return ret_df

    
def pre_proc_per_df(df, del_single_cat_cols=False):
    ret_df = df.copy()
    
#    print(txt_w_border("Merging SibSp and Parch to FamSize")
#    ret_df = merge_SibSp_Parch_to_FamSize(ret_df)
    
    print(txt_w_border("Filtering " + str(KEYS_FOR_ML)))
    ret_df = filter_cols(ret_df)
    
    print(txt_w_border("Adding null flag columns"))
    ret_df = add_null_flag_cols(ret_df, del_single_cat_cols)
    
    print(txt_w_border("Binarizing " + str(BINARIZED_CAT_DICT.keys())))
    ret_df = binarize(ret_df)
    
    print(txt_w_border("Nomarizing " + str(KEYS_FOR_NORM)))
    ret_df = normalize(ret_df)
    
    key_of_fill = 0.
    print(txt_w_border("Filling null with " + str(key_of_fill)))
    ret_df = ret_df.fillna(key_of_fill)
    
    return ret_df


def pre_proc_all():
    print(txt_w_border("Importing " + FILE_PATH_TRAIN_DATA))
    raw_train_df = load_titanic_csv(FILE_PATH_TRAIN_DATA)
    
    print(txt_w_border("Importing " + FILE_PATH_PREDICT_DATA))
    raw_test_df = load_titanic_csv(FILE_PATH_PREDICT_DATA)
    
    raw_all_df  = pd.concat([raw_train_df, raw_test_df])
    t_df = pre_proc_per_df(raw_all_df, del_single_cat_cols=True)
    
    def split_train_target(df, tf_surv):
        temp_df = df[df.Survived_null_True==tf_surv]
        target_df = temp_df[["Survived"]].copy()
        train_df  = temp_df.drop(["Survived", "Survived_null_True"], axis=1)
        return train_df, target_df
    train_df, train_target_df = split_train_target(t_df, tf_surv=False)
    test_df,  test_target_df  = split_train_target(t_df, tf_surv=True)
    
    return train_df, train_target_df, test_df, test_target_df
    
def main():
    parser = argparse.ArgumentParser(description='Exploring Survival on the Titanic -kaggle-')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of data in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

     # Load the Titanic dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    
if __name__=="__main__":
    main()
    