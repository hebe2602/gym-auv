import torch
import torch.nn as nn
import numpy as np


class LidarCNN(nn.Module):
 
    def __init__(self, n_sensors:int, output_channels:list, kernel_size:int=5):
        super().__init__()
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.padding = 2#(self.kernel_size - 1) // 2
        self.output_channels = output_channels
        # self.pool_kernel_size = 2
        # self.pool_stride = 2

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular',
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2),
            nn.Conv1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2),
            nn.Conv1d(
                in_channels  = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2),
            nn.Conv1d(
                in_channels  = self.output_channels[2],
                out_channels = self.output_channels[3],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2),
            nn.Flatten()
        )
        # Output of feature_extractor is [N, C_out, L/num_maxpool]
        len_flatt = 11 * self.output_channels[-1]
        self.regressor = nn.Sequential(
            nn.Linear(len_flatt, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def forward(self, x):
 
        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.regressor:
            x = layer(x)

        # print('Output size:', x.shape) 
        return x
    
    def _init_weights(self):
        layers = [*self.feature_extractor, *self.regressor]
        
        # Original xavier_uniform parameter initialization
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: 
                    nn.init.xavier_uniform_(param)

        # if subnet_init == "gaussian": # Our weight and bias initialization based on the Focal loss paper
        #     for layer in layers:
        #         if isinstance(layer, nn.Conv2d):
        #             nn.init.normal_(layer.weight, mean=0.0,std=0.01)
        #             nn.init.zeros_(layer.bias)
        #     p = 0.99        
        #     bias = np.log(p * (self.num_classes-1)/(1-p))
        #     nn.init.constant_(self.classification_heads[-1].bias[:self.num_boxes[0]], bias)


class BasicModel(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = 2#(self.kernel_size - 1) // 2
        self.layer = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, padding=2, padding_mode='circular')
        

    def forward(self, x):
        out = self.layer(x)
        return out

