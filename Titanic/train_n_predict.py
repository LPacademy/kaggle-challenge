# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:27:21 2016

@author: takashi
"""

from pre_process import extract_train_target
from pre_process import FILE_PATH_TRAIN_DATA
from pre_process import FILE_PATH_PREDICT_DATA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.svm as svm
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train_by_SVC_RBF(C, gamma, split_train_df, split_train_target_df):
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(split_train_df, split_train_target_df)
    
    return svc

    
def predict_by_SVC_RBF(svc, predictor_df, file_path_save_results=None):
    predict_arr = svc.predict(predictor_df)
    
    # Convert to pandas dataframe from numpy array
    temp_df = predictor_df.copy()
    temp_df["Survived"] = predict_arr
    predict_df = temp_df[["Survived"]].copy()
    
    if file_path_save_results==None:
        print "Not saved predicted results." 
    else:
        try:
            predict_df.to_csv(file_path_save_results)
            print "Saved predicted results in", file_path_save_results+"."
        except:
            print "Failed saving predicted results."
    
    return predict_df

    
def eval_prediction(predict_df, target_df):
    """
    Evaluate predicted objective variables
    Parameters:
        predict_df: Predicted objective variables in dataframe
        target_df: Answer of objective variables in dataframe
    Returns:
        conf_mat: Confusion matrix
        accuracy: Accuracy
    """
    conf_mat = metrics.confusion_matrix(predict_df, target_df)
    accuracy = metrics.accuracy_score(predict_df, target_df)
    return conf_mat, accuracy

def eval_svc_by_x_val():
    """
    Evaluate SVC model
    """
    
    
if __name__=="__main__":
    # Prepare train data and test data
    train_df, train_target_df = extract_train_target(FILE_PATH_TRAIN_DATA)
    test_df,       predict_df = extract_train_target(FILE_PATH_PREDICT_DATA)
    
    # Split the train data for cross validation
    split_train_df, split_val_df, split_train_target_df, split_val_target_df = train_test_split(train_df, train_target_df, test_size=0.15)
    
    # Train and evaluate the model
    C_idx_set, gamma_idx_set = np.mgrid[-10:10:0.5,-10:10:0.5]
    cols_for_res = ["svc", "score_mean", "score_std"]   #######
    results_set  = pd.DataFrame(columns=cols_for_res)   #######
    score_mean_set   = np.zeros_like(C_idx_set)
    score_std_set    = np.zeros_like(C_idx_set)
    score_mean_prev  = 0.0
    for i, (C, gamma) in enumerate(zip(2.0**C_idx_set.reshape(-1), 2.0**gamma_idx_set.reshape(-1))):
        svc = train_by_SVC_RBF(C, gamma, split_train_df, split_train_target_df)
        
        # Assess the model with cross validation data
        scores = cross_val_score(svc, train_df.as_matrix(), train_target_df.as_matrix().reshape(-1), cv=10)
        score_mean_set.reshape(-1)[i] = score_mean = scores.mean()
        score_std_set.reshape(-1)[i]  = score_std  = scores.std()
        temp = pd.DataFrame([[svc, score_mean, score_std]], columns=cols_for_res, index=[i])
        results_set = results_set.append(temp) #######
        print "{0:4d}/{1:4d} Param C: {2:0.2e}, gamma: {3:0.2e}, ".format(i, C_idx_set.size, C, gamma),
        print "Score mean: {0:0.3f}, std: {1:0.3f}".format(scores.mean(), scores.std())
        if scores.mean()>score_mean_prev:
            svc_best = svc
            score_mean_prev = scores.mean()
        
    
    # Plot images
    plt.figure(figsize=(16, 9))
    def subplots_im_data(subplot_pos, im_data, title):
        plt.subplot(subplot_pos)
        plt.imshow(im_data, interpolation="nearest", cmap="nipy_spectral", extent=[-10,9.5,9.5,-10])
        plt.title(title)
        plt.colorbar()
    subplots_im_data(221, C_idx_set,     "Index of C, 2**X")
    subplots_im_data(222, gamma_idx_set, "Index of Gamma, 2**X")
    subplots_im_data(223, score_mean_set,"Mean of Score" )
    subplots_im_data(224, score_std_set, "Std of Score")
    plt.show()
    
    # Predict with the test.csv
    print "Parameters of the estimator, svc.C = {0:0.2e}, svc.gamma = {1:0.2e}".format(svc_best.C, svc_best.gamma)
    print results_set[results_set.svc==svc_best]
    predict_df = predict_by_SVC_RBF(svc_best, test_df, file_path_save_results="predict.csv")
    