from math import ceil
from pyexpat.model import XML_CTYPE_EMPTY
import torch
# from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd



n_sensors = 180
# model = LidarCNN(n_sensors=n_sensors, output_channels=[4, 4, 4, 4])

def load_data_wo_batches(path_x:str, path_y:str, normalize:bool=True, shuffle:bool=True) -> list:
    '''
    Load data and split into training, validation and test set.
    '''
    print('Loading data from', path_x, path_y)

    X = np.loadtxt(path_x)
    Y = np.loadtxt(path_y)


    if shuffle:
        np.random.shuffle(X)
        np.random.shuffle(Y)


    
    X = torch.Tensor(X) #(N_batch,1,180)
    Y = torch.Tensor(Y)
    X = X[:,None, :]
    Y = Y[:,None]
    

    # Training set
    X_train = X[:350,:,:]#.detach().clone()
    Y_train = Y[:350,:]#.detach().clone()
    # Validation set
    X_val = X[350:425,:,:]#.detach().clone()
    Y_val = Y[350:425,:]#.detach().clone()
    # Test set
    X_test = X[425:,:,:]#.detach().clone()
    Y_test = Y[425:,:]#.detach().clone()

    print('Shape of X_train:', X_train.shape, X_train.dtype)
    print('Shape of Y_train:', Y_train.shape, Y_train.dtype)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_data(path_x:str, path_y:str, train_test_split:float=0.7, train_val_split:float=0.2, normalize:bool=True, shuffle:bool=True) -> list:
    '''
    Load data and split into training, validation and test set.
    '''
    print('Loading data from', path_x, path_y)

    X = np.loadtxt(path_x)
    Y = np.loadtxt(path_y)

    data_size  = len(X)
    train_size = int(train_test_split * data_size)
    val_size   = int(train_val_split * train_size)
    test_size  = data_size - train_size
    train_size = train_size - val_size

    if normalize:
        x_mean, x_std = X.mean(), X.std()
        X = (X - x_mean)/x_std

    if shuffle:
        np.random.shuffle(X)
        np.random.shuffle(Y)

    X = X[:,None, :]
    Y = Y[:,None]

    # Training set
    X_train, Y_train = divide_batches(X[:train_size,:,:], Y[:train_size,:], batch_size=100)
    # Validation set
    X_val = torch.Tensor(X[train_size:train_size+val_size,:,:])
    Y_val = torch.Tensor(Y[train_size:train_size+val_size,:])
    # Test set
    X_test = torch.Tensor(X[-test_size:,:,:])
    Y_test = torch.Tensor(Y[-test_size:,:])

    print('Shape of X_train:', len(X_train), X_train[0].shape, X_train[1].dtype)
    print('Shape of Y_train:', len(Y_train), Y_train[0].shape, Y_train[1].shape, Y_train[2].shape)
    print('Shape of X_val:', X_val.dtype)

    print(data_size, train_size, val_size, test_size)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def divide_batches(X:np.ndarray, Y:np.ndarray, batch_size:int):
    num_batches = len(X) // batch_size
    x_batches = []
    y_batches = []
    for b in range(num_batches-1):
        x_batches.append(torch.Tensor(X[b*batch_size:(b+1)*batch_size,:,:]))
        y_batches.append(torch.Tensor(Y[b*batch_size:(b+1)*batch_size,:]))
    x_batches.append(torch.Tensor(X[(b+1)*batch_size:,:,:]))
    y_batches.append(torch.Tensor(X[(b+1)*batch_size:,:]))    
    
    return x_batches, y_batches


class LiDARDataset(torch.utils.data.Dataset):
  '''
  Prepare the dataset for regression
  '''

  def __init__(self, X:np.ndarray, y:np.ndarray, x_mean:np.float, x_std:np.float, stanarize:bool=True):
    
    if not torch.is_tensor(X) and not torch.is_tensor(y):
        if stanarize:
            # X = StandardScaler().fit_transform(X)
            X = (X - x_mean)/x_std
        
        self.X = torch.Tensor(X[:,None, :])
        self.y = torch.Tensor(y[:,None])

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]


# NB! All samples should be standardized with whole data mean and std!

def load_LiDARDataset(path_x:str, path_y:str, batch_size:int, train_test_split:float=0.7, train_val_split:float=0.2, shuffle:bool=True, test_as_tensor:bool=False):
    '''
    Load training set, validation set and test set from paths.
    '''
    X = np.loadtxt(path_x)
    # y = np.loadtxt(path_y)

    y = calculate_total_risk(path_y)

    x_mean, x_std = X.mean(), X.std()

    data_size  = len(X)
    train_size = int(train_test_split * data_size)
    val_size   = int(train_val_split * train_size)
    test_size  = data_size - train_size
    train_size = train_size - val_size
    # Training set
    X_train = X[:train_size,:]
    y_train = y[:train_size]
    data_train = LiDARDataset(X_train, y_train, x_mean, x_std)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    # Validation set
    X_val = X[train_size:train_size+val_size,:]
    y_val = y[train_size:train_size+val_size]
    data_val = LiDARDataset(X_val, y_val, x_mean, x_std)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    # Test set
    X_test = X[-test_size:,:]
    y_test = y[-test_size:]
    data_test = LiDARDataset(X_test, y_test, x_mean, x_std)
    
    if test_as_tensor:
        return dataloader_train, dataloader_val, data_test    
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return dataloader_train, dataloader_val, dataloader_test

# transform_train = transforms.Compose([
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])

def calculate_total_risk(path_y:str, mode:str='max') -> np.ndarray:
    # Y has a list of CRI at each row -> number of risks varies at each row -> read with predefined number of cols -> pandas
    # Placeholder, can consider more sophisticated ways to calculate the total risk when more then one obstacle is present.

    Y = pd.read_csv(path_y, delimiter=r"\s+", header=None, names=[i for i in range(5)])

    if mode=='sum':
        y = Y.sum(axis=1)
        print(y.shape)
    
    elif mode == 'max':
        y = Y.max(axis=1)

    else:
        y = np.mean(Y, axis=1)

    y = y.to_numpy(copy=True)
    return y
