# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 04:39:11 2017

@author: takashi
"""


from pre_process import pre_proc_all
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNC
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
    n_neighbors_set = range(1, 30)
    cols_for_res = ["clf", "score_mean", "score_std"]   #######
    results_set  = pd.DataFrame(columns=cols_for_res)   #######
    for i, n_neighbors in enumerate(n_neighbors_set):
        # Define the classifyer model
        clf = KNC(n_neighbors=n_neighbors)
        
        # Assess the model with cross validation data
        tgt_arr = train_target_df.as_matrix().reshape(-1)
        scores  = cross_val_score(clf, train_df, tgt_arr, cv=4)
        
        temp = pd.DataFrame([[clf, scores.mean(), scores.std()]], columns=cols_for_res, index=[i])
        results_set = results_set.append(temp) #######
        print "{0:4d}/{1:4d} Param n_neighbors: {2:2d}, ".format(i, len(n_neighbors_set), n_neighbors),
        print "Score mean: {0:0.3f}, std: {1:0.3f}".format(scores.mean(), scores.std())
        
    mean = results_set.score_mean
    std  = results_set.score_std
    t = mean - 1.5*std
    print results_set[t==t.max()][["score_mean","score_std"]]
                      
    # Plot images
    plt.figure(figsize=(16, 9))
    def subplots_data(subplot_pos, data, label, title=""):
        plt.subplot(subplot_pos)
        plt.plot(n_neighbors_set, data, "o-", label=label)
        plt.title(title)
        plt.legend()
    subplots_data(111, mean + std * 1.5,      "Mean + 1.5 Std")
    subplots_data(111, results_set.score_mean,"Mean of Score" )
    subplots_data(111, mean - std * 1.5,      "Mean - 1.5 Std")
    plt.show()
    
    # Predict with the test.csv
    
#    predict_df = predict_by_SVC_RBF(svc_best, test_df, file_path_save_results="predict.csv")
    