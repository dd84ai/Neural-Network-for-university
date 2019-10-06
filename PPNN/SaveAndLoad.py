# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:40:17 2019

@author: User
"""

import pandas as pd
import numpy as np
#%matplotlib inline

np.random.seed(2)

def Load(show):
    # Load the data
    train = pd.read_csv("C:/data/input/train.csv")
    #test = pd.read_csv("C:/data/input/test.csv")
    Y_train = train["label"]
    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1) 
    # free some space
    del train 
    if(show):
        import seaborn as sns
        sns.countplot(Y_train)
    Y_train.value_counts()
    return X_train,Y_train
