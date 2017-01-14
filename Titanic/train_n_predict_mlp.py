# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 00:59:32 2017

@author: takashi
"""


from pre_process import pre_proc_all
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier as MLPClf
import matplotlib.pyplot as plt
plt.rcParams["font.family"]       = "monospace"
plt.rcParams["font.size"]         = 16
plt.rcParams["axes.grid"]         = True
plt.rcParams["axes.facecolor"]    = "white"
plt.rcParams["legend.fontsize"]   = "small"
plt.rcParams["legend.loc"]        = "best"
plt.rcParams["figure.facecolor"]  = "white"
plt.rcParams["axes.titlesize"]    = "medium"
import pandas as pd
import numpy as np


def predict_n_save(clf, predictor_df, file_path_save_results=None):
    predict_arr = clf.predict(predictor_df)
    
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


    
if __name__=="__main__":
    # Prepare train data and test data
    train_df, train_target_df, test_df, predict_df = pre_proc_all()
    
    # Train and evaluate the model
#    hidden_layer_sizes_set, alpha_set = 2.0**np.mgrid[-10.:10.:3.,-10.:10.:3.]
    alpha_set = 2.0**np.arange(-15, 5, 0.5)
    cols_for_res = ["clf", "score_mean", "score_std"]   #######
    results_set  = pd.DataFrame(columns=cols_for_res)   #######
#    for i, (hls, alpha) in enumerate(zip(hidden_layer_sizes_set.flatten(), alpha_set.flatten())):
    for i, alpha in enumerate(alpha_set.flatten()):
        # Define the classifyer model
        clf = MLPClf(hidden_layer_sizes=(9, 9, 9),
                     alpha=alpha,
                     learning_rate_init=0.01
                     )
        
        # Assess the model with cross validation data
        tgt_arr = train_target_df.as_matrix().flatten()
        scores  = cross_val_score(clf, train_df, tgt_arr, cv=20)
        
        temp = pd.DataFrame([[clf, scores.mean(), scores.std()]], columns=cols_for_res, index=[i])
        results_set = results_set.append(temp) #######
        print "{0:4d}/{1:4d} Param alpha: {2:0.2e}, ".format(i, alpha_set.size, alpha),
        print "Score mean: {0:0.3f}, std: {1:0.3f}".format(scores.mean(), scores.std())
        
    mean = results_set.score_mean
    std  = results_set.score_std
    t = mean - 1.5*std
    print results_set[t==t.max()][["score_mean","score_std"]]
    
    
    plt.plot(mean)
    # Plot images
#    score_mean_set = mean.as_matrix().reshape(C_set.shape)
#    score_std_set  = std.as_matrix().reshape(C_set.shape)
#    plt.figure("SVC", figsize=(16, 9))
#    def subplots_im_data(subplot_pos, im_data, title=""):
#        plt.subplot(subplot_pos)
#        plt.imshow(im_data, interpolation="nearest", cmap="nipy_spectral")
#        plt.title(title)
#        plt.colorbar()
#    subplots_im_data(231, np.log2(C_set),     "Index of C, 2**x")
#    subplots_im_data(232, np.log2(gamma_set), "Index of Gamma, 2**x")
#    subplots_im_data(234, score_mean_set,                   "Mean of Score" )
#    subplots_im_data(235, score_std_set,                    "Std of Score")
#    subplots_im_data(236, score_mean_set-1.5*score_std_set, "Mean-1.5*Std")
#    plt.show()
    
    
    # Predict with the test.csv
    
#    predict_df = predict_by_SVC_RBF(svc_best, test_df, file_path_save_results="predict.csv")
    