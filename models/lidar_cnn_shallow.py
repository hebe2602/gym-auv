import torch
import torch.nn as nn
import numpy as np

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import gym

class LidarCNN_shallow(nn.Module):
 
    def __init__(self,
                 n_sensors:int=180, 
                 output_channels:list=[1], 
                 kernel_size:int=45, 
                 padding:int=15,
                 stride:int=15):
        super().__init__()
        self.n_sensors       = n_sensors
        self.kernel_size     = kernel_size
        self.padding         = padding 
        self.stride          = stride
        self.output_channels = output_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = self.kernel_size,
                stride       = self.stride,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        len_flat =12
        self.linear = nn.Sequential(
            nn.Linear(len_flat, 1),
            nn.ReLU()
        )
   


    def forward(self, x):
        
        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.linear:
            x = layer(x)

        return x