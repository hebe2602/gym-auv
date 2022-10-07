from math import ceil
from pyexpat.model import XML_CTYPE_EMPTY
import torch
# from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


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
