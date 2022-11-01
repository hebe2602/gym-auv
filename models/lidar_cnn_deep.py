import torch
import torch.nn as nn
import numpy as np

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import gym

class LidarCNN_2_deep(nn.Module):
 
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
                kernel_size  = 45,
                stride       = 15,
                padding      = 15,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels  = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Flatten()
        )


        len_flat = 12 
        self.linear = nn.Sequential(
            nn.Linear(len_flat, 1),
            nn.ReLU(),    
        )
  

    def forward(self, x):
    
        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.linear_1:
            x = layer(x)

        return x



class LidarCNN_deep(nn.Module):
 
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
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
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
                         stride      = 2,
                         ceil_mode   = True),
            nn.Conv1d(
                in_channels  = self.output_channels[2],
                out_channels = self.output_channels[3],
                kernel_size  = self.kernel_size,
                stride       = 1,
                padding      = self.padding,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2,
                         stride      = 2,
                         ceil_mode   = True),
            nn.Flatten()
        )
        # Output of feature_extractor is [N, C_out, L/num_maxpool]
        len_flat = int(np.ceil(self.n_sensors/2**4) * self.output_channels[-1])
        self.linear_1 = nn.Sequential(
            nn.Linear(len_flat, 40),
            nn.ReLU(),
            nn.Linear(40, 8),
            
        )
        self.linear_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU()
        )


    def forward(self, x):
        
        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.linear_1:
            x = layer(x)
        
        for layer in self.linear_2:
            x = layer(x)
        
        return x



 
# class LidarCNN_single_output_feature(nn.Module):
 
#     def __init__(self, n_sensors:int, output_channels:list, kernel_size:int=5):
#         super().__init__()
#         self.n_sensors = n_sensors
#         self.kernel_size = kernel_size
#         self.padding = (self.kernel_size - 1) // 2
#         self.output_channels = output_channels

#         self.feature_extractor = nn.Sequential(
#             nn.Conv1d(
#                 in_channels  = 1,
#                 out_channels = self.output_channels[0],
#                 kernel_size  = self.kernel_size,
#                 stride       = 1,
#                 padding      = self.padding,
#                 padding_mode = 'circular'
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size = 2,
#                          stride      = 2,
#                          ceil_mode   = True),
#             nn.Conv1d(
#                 in_channels  = self.output_channels[0],
#                 out_channels = self.output_channels[1],
#                 kernel_size  = self.kernel_size,
#                 stride       = 1,
#                 padding      = self.padding,
#                 padding_mode = 'circular'
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size = 2,
#                          stride      = 2,
#                          ceil_mode   = True),
#             nn.Conv1d(
#                 in_channels  = self.output_channels[1],
#                 out_channels = self.output_channels[2],
#                 kernel_size  = self.kernel_size,
#                 stride       = 1,
#                 padding      = self.padding,
#                 padding_mode = 'circular'
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size = 2,
#                          stride      = 2,
#                          ceil_mode   = True),
#             nn.Conv1d(
#                 in_channels  = self.output_channels[2],
#                 out_channels = self.output_channels[3],
#                 kernel_size  = self.kernel_size,
#                 stride       = 1,
#                 padding      = self.padding,
#                 padding_mode = 'circular'
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size = 2,
#                          stride      = 2,
#                          ceil_mode   = True),
#             nn.Flatten()
#         )
#         # Output of feature_extractor is [N, C_out, L/num_maxpool]
#         len_flat = int(np.ceil(self.n_sensors/2**4) * self.output_channels[-1])
#         self.regressor = nn.Sequential(
#             nn.Linear(len_flat, 40),
#             nn.ReLU(),
#             nn.Linear(40, 1),
#             # nn.Sigmoid()
#             nn.ReLU()
#         )


#     def forward(self, x):
        
#         for layer in self.feature_extractor:
#             x = layer(x)

#         for layer in self.regressor:
#             x = layer(x)

#         return x

