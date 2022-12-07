import torch.nn as nn
import numpy as np

class LidarCNN_2D(nn.Module):
 
    def __init__(self, n_sensors:int, output_channels:list, kernel_size:int=5):
        super().__init__()
        self.n_sensors       = n_sensors
        self.kernel_size     = kernel_size
        self.padding         = (self.kernel_size - 1) // 2
        self.output_channels = output_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
            nn.Conv2d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
            nn.Conv2d(
                in_channels  = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
            
            nn.Flatten()
        )
        # Output of feature_extractor is [N, C_out, L/num_maxpool]
        len_flat = 23 * self.output_channels[-1]
        self.regressor = nn.Sequential(
            nn.Linear(len_flat, 40),
            nn.ReLU(),
            # nn.Linear(16, 4),
            # nn.ReLU(),
            nn.Linear(40, 1),
            # nn.Sigmoid()
            nn.ReLU()
        )
   
    def forward(self, x):
        # print(x.shape)
        for layer in self.feature_extractor:
            x = layer(x)
            # print(x.shape)
     
        for layer in self.regressor:
            x = layer(x)
            # print(x.shape)

        return x
    