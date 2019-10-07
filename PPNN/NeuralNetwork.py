# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:44:27 2019

@author: User
"""


from SaveAndLoad import Load
from Extra import NormalizeAndSplit
from Model import SetModel, Train, Validate
#If True,shows value distribution
#X-pictures,Y-their identification
X_train,Y_train=Load(False)

def CheckData():
    # Check the data
    print(X_train.isnull().any().describe())
    #print(test.isnull().any().describe())
CheckData()
    
X_norm,X_val,Y_norm,Y_val = NormalizeAndSplit(X_train,Y_train)
model = SetModel()  
#ModelVizualize(model)

#First paremeter =quantity of epochs to train NN
history = Train(2, model,X_norm,X_val,Y_norm,Y_val)
Y_pred, Y_pred_classes, Y_true= Validate(model,X_val,Y_val)