##################################################################
####
####
####     this program is to load data, analysis data
####
####           Dec. 2016.
####          Hiroyuki  Miyoshi
##################################################################





import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import h5py
import sys


WORKDIR = "/Users/MiyoshiHiroyuki/Documents/LPixel/study/Kaggle/Titanic/scripts"
DATADIR = "/Users/MiyoshiHiroyuki/Documents/LPixel/study/Kaggle/Titanic/data"


def writehdf5():
    output_file = "traindata.h5"
    h5file = h5py.File(output_file,'w')

    Embarked_list = ['S','Q','C']
    Sex_list = ['male','female']
    Cabin_list = ['A','B','C','D','E','F','G','T']
    outputcategory = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Survived']
    mat = np.empty((0,len(outputcategory)))

    with open(DATADIR+'/train.csv', 'rb') as csvfile:
        #spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        spamreader = csv.reader(csvfile)
        first=True
        for row in spamreader:
            print(row)
            if(first==True):
                allcategory = row
                print("# allcategory == ", allcategory)
                print("# outputcategory == ", outputcategory)
                first=False
            else:
                mat_each = np.empty((1,0))
                for cat in outputcategory:
                    if(row[allcategory.index(cat)] == str(-1)):
                        val = -1
                    elif(cat == 'Sex'):
                        val = Sex_list.index(str(row[allcategory.index(cat)]))
                    elif(cat == 'Embarked'):
                        val = Embarked_list.index(row[allcategory.index(cat)])
                    elif(cat == 'Cabin'):
                        val = Cabin_list.index(row[allcategory.index(cat)][0])
                    else:
                        val = row[allcategory.index(cat)]
                    mat_each = np.append(mat_each,val)

                mat = np.append(mat, np.reshape(mat_each,[1,-1]),axis=0)
                print(mat.shape)
                #raw_input()
        foldername = "train"
        h5file.create_group(foldername)
        h5file.create_dataset(foldername+"/data",data= mat)
        h5file.create_dataset(foldername+"/value",data=outputcategory)
        h5file.flush()
        h5file.close()



def main():
    with open(DATADIR+'/train.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print ', '.join(row)

if __name__ == '__main__':
    writehdf5()
