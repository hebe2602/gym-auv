import torch
import torch.nn as nn
import numpy as np

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import gym

class LidarCNN_shallow(nn.Module):
 
    def __init__(self, n_sensors:int, output_channels:list, kernel_size:int=5):
        super().__init__()
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.padding = 15 # (self.kernel_size - 1) // 2
        self.output_channels = output_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = self.kernel_size,
                stride       = 15,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Flatten()
 
        )
        # Output of feature_extractor is [N, C_out, L/num_maxpool]
        len_flat =12# int(np.ceil(self.n_sensors/2**4) * self.output_channels[-1])
        self.linear = nn.Sequential(
            nn.Linear(len_flat, 1),
            # nn.ReLU(),
            # nn.Linear(6, 1),
            nn.ReLU()
        )
   


    def forward(self, x):
        
        for layer in self.feature_extractor:
            x = layer(x)
        print(x.shape)
        for layer in self.linear:
            x = layer(x)
        print(x.shape)
        exit()
        return x