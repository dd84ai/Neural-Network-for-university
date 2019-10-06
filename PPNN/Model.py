# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:29:13 2019

@author: User
"""

import numpy as np
np.random.seed(2)
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Flatten#Dropout, 
from keras.layers.convolutional import Conv2D#, MaxPooling2D
#from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
def SetModel():
    # Set the CNN model 
    # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
    
    #https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    model = Sequential()
    
    model.add(Conv2D(filters = 4, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    #model.add(Conv2D(filters = 32, kernel_size=(5,5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
    #                 activation ='relu', input_shape = (28,28,1)))
    #model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
    #                 activation ='relu'))
    #model.add(MaxPool2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))
    
    #model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
    #                 activation ='relu'))
    #model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
    #                 activation ='relu'))
    #model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    #model.add(Dropout(0.25))
    
    #model.add(Flatten())
    #model.add(Dense(256, activation = "relu"))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation = "softmax"))
    return model
    
def LearningRate():
    # Define the optimizer
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    # Compile the model
    #model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    # Set a learning rate annealer
    return ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)


def Train(epochs,model, X_train,X_val,Y_train,Y_val):
    learning_rate_reduction = LearningRate()
    #epochs = 30 Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86
    
    # Without data augmentation i obtained an accuracy of 0.98114
    #history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
    #          validation_data = (X_val, Y_val), verbose = 2)
    
    # With data augmentation to prevent overfitting (accuracy 0.99286)
    
    datagen = ImageDataGenerator()  # randomly flip images
    datagen.fit(X_train)

    # Fit the model
    return model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

from Extra import plot_confusion_matrix
def Validate(model,X_val,Y_val):
    # Predict the values from the validation dataset
    Y_pred = model.predict(X_val)
    # Convert predictions classes to one hot vectors 
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val,axis = 1) 
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 