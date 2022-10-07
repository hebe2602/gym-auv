from cProfile import label
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def do_predictions(model:nn.Module, dataloader:torch.utils.data.DataLoader):
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch) # only one batch in the test set
        
        y_pred  = y_pred.numpy()
        y_batch = y_batch.numpy()
        
    model.train()
    return y_pred, y_batch


def runtime_analysis():
    raise NotImplementedError


def plot_mse():
    raise NotImplementedError



def plot_predictions(y_pred, y):

    plt.figure(figsize=(15,15))
    plt.plot(y_pred, label='Predicted risk')
    plt.plot(y, label='True risk')
    plt.legend()
    plt.show()

def plot_loss(training_loss, validation_loss):
    
    epochs = np.arange(len(training_loss))

    plt.figure(figsize=(15,15))
    plt.plot(epochs, training_loss, label='Training_loss')
    plt.plot(epochs, validation_loss, label='Validation loss')
    plt.legend()
    plt.show()

