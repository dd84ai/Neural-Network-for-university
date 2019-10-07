# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:42:48 2019

@author: User
"""
import matplotlib.pyplot as plt
def ModelVizualize(model):
    #Ver2.0 Visualizing
    #https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
    from keras.utils.vis_utils import plot_model
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
def GetVisual1(history):
    #Ver2.0 Vizualization
    #https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras
    #plotting training and validation loss
    import matplotlib.pyplot as plt
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='green', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def GetVisual2(history):
    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    
    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    return legend
    
def GetVisual3(history,epochs):
    #Ver2.0 Vizualization
    #plotting training and validation accuracy
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, color='red', label='Training acc')
    plt.plot(epochs, val_acc, color='green', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def GetVisual4(model,X_val,Y_val):
        print("on valid data")
        pred1=model.evaluate(X_val,Y_val)
        print("accuaracy", str(pred1[1]*100))
        print("Total loss",str(pred1[0]*100))

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
import numpy as np            
def GetVisualErrors(Y_pred, Y_pred_classes, Y_true, X_val):
    # Display some error results 
    # Errors are difference between predicted labels and true labels
    errors = (Y_pred_classes - Y_true != 0)
    
    Y_pred_classes_errors = Y_pred_classes[errors]
    Y_pred_errors = Y_pred[errors]
    Y_true_errors = Y_true[errors]
    X_val_errors = X_val[errors]

    # Probabilities of the wrong predicted numbers
    Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
    
    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    
    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    
    # Sorted list of the delta prob errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)
    
    # Top 6 errors 
    most_important_errors = sorted_dela_errors[-6:]
    
    # Show the top 6 errors
    display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
