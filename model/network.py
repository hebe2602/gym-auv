import torch
import torch.nn as nn
import numpy as np


class LidarCNN(nn.Module):
 
    def __init__(self, n_sensors:int, output_channels:list, kernel_size:int=5):
        super().__init__()
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
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
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
            nn.Conv1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 4,
                         stride      = 4,
                         ceil_mode   = True),
            
            nn.Flatten()
        )
        # Output of feature_extractor is [N, C_out, L/num_maxpool]
        len_flat = 23 * self.output_channels[-1]
        self.regressor = nn.Sequential(
            nn.Linear(len_flat, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            # nn.Sigmoid()
            nn.ReLU()
        )


    def forward(self, x):
 
        for layer in self.feature_extractor:
            x = layer(x)

        # print('shape output:', x.shape)
        for layer in self.regressor:
            x = layer(x)

        return x
    
    # def _init_weights(self, param_init):
    #     layers = [*self.feature_extractor, *self.regressor]
        
    #     if param_init =='xavier':
    #         for layer in layers:
    #             for param in layer.parameters():
    #                 if param.dim() > 1: 
    #                     nn.init.xavier_uniform_(param)

    #     if param_init == "gaussian":
    #         for layer in layers:
    #             if isinstance(layer, nn.Conv2d):
    #                 nn.init.normal_(layer.weight, mean=0.0,std=0.01)
    #                 nn.init.zeros_(layer.bias)
          

class LidarCNN_LSTM(nn.Module):
 
    def __init__(self, n_sensors:int, output_channels:list, kernel_size:int=5, param_init:str='xavier'):
        super().__init__()
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.output_channels = output_channels
        # self.pool_kernel_size = 2
        # self.pool_stride = 2
        self.batch_size  = 20
        self.lstm_layers = 1
        self.hidden_size = 8
        self.hidden_state = torch.zeros(1,16)
        self.cell_state = torch.zeros(1,16)

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
            nn.Conv1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 4,
                         stride      = 4,
                         ceil_mode   = True),
            nn.Flatten()
        )
        # Output of feature_extractor is [N, C_out, L/num_maxpool]
        len_flat = 23 * self.output_channels[-1]

        self.regressor = nn.Sequential(
            nn.LSTM(input_size=len_flat, hidden_size=16, num_layers=1),#, batch_first=True),
            # nn.LSTM(input_size=8, hidden_size=8, num_layers=1),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            # nn.Sigmoid()
            nn.ReLU()
        )
        

    def forward(self, x):
 
        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.regressor:
            if isinstance(layer, nn.LSTM):
                x, (hn, cn) = layer(x, (self.hidden_state, self.cell_state))
                self.hidden_state = hn.detach()
                self.cell_state   = cn.detach()
            else:
                x = layer(x)
        return x
 


class LidarCNN_best(nn.Module):
 
    def __init__(self, n_sensors:int, output_channels:list, kernel_size:int=5):
        super().__init__()
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.output_channels = output_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
            nn.Conv1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 4,
                         stride      = 4,
                         ceil_mode   = True),

            nn.Flatten()
        )
        # Output of feature_extractor is [N, C_out, L/num_maxpool]
        len_flat = 23 * self.output_channels[-1]
        self.regressor = nn.Sequential(
            nn.Linear(len_flat, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            # nn.Sigmoid()
            nn.ReLU()
        )


    def forward(self, x):
 
        for layer in self.feature_extractor:
            x = layer(x)

        # print('shape output:', x.shape)
        for layer in self.regressor:
            x = layer(x)

        return x
    
