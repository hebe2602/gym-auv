import gym
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarCNN_deep_pretrained(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space:gym.spaces.Box,
                 output_channels:list = [2,4,4,6],
                 kernel_size:int      = 9,
                 n_sensors:int        = 180, 
                 features_dim:int     = 8):

        super(LidarCNN_deep_pretrained, self).__init__(observation_space, features_dim=features_dim)
        
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


    def forward(self, x):

        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.linear_1:
            x = layer(x)

        return x

class LidarCNN_shallow_pretrained(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space:gym.spaces.Box,
                 n_sensors:int=180, 
                 output_channels:list=[1], 
                 kernel_size:int=45, 
                 padding:int=15,
                 stride:int=15,
                 features_dim:int=12):

        super(LidarCNN_shallow_pretrained, self).__init__(observation_space, features_dim=features_dim)
        self.n_sensors       = n_sensors
        self.kernel_size     = kernel_size
        self.padding         = padding 
        self.stride          = stride
        self.output_channels = output_channels

        self.mean = 143.3607156355717  # mean found during training
        self.std  = 23.58293602126056  # standard deviation found during traning

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
    
    def forward(self, x):
        x = (x - self.mean)/self.std
        for layer in self.feature_extractor:
            x = layer(x)

        return x

class NavigatioNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 6):
        super(NavigatioNN, self).__init__(observation_space, features_dim=features_dim)

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        shape = observations.shape
        observations = observations[:,0,:].reshape(shape[0], shape[-1])
        return self.passthrough(observations)

class PerceptionNavigationExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (1, 3, N_sensors)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, sensor_dim : int = 180, features_dim: int = 8, kernel_overlap : float = 0.05):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(PerceptionNavigationExtractor, self).__init__(observation_space, features_dim=1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "perception":

                # cnn = LidarCNN_deep_pretrained(observation_space=subspace, 
                #                           n_sensors=sensor_dim, 
                #                           output_channels=[2,4,4,6], 
                #                           kernel_size=9,
                #                           features_dim=features_dim) #8

                cnn = LidarCNN_shallow_pretrained(observation_space=subspace, 
                                                  n_sensors=sensor_dim, 
                                                  output_channels=[1], 
                                                  kernel_size=45,
                                                  padding=15,
                                                  stride=15,
                                                  features_dim=features_dim) #12

                print('Loading pretrained LidarCNN')                          
                pretrained_dict = th.load('gym_auv/utils/model_shallow_pretrained.json')
                model_dict = cnn.state_dict()
                
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # 1. filter out unnecessary keys
                model_dict.update(pretrained_dict) # 2. overwrite entries in the existing state dict
                cnn.load_state_dict(pretrained_dict) # 3. load the new state dict

                print(cnn)
                for param in cnn.parameters():
                    param.requires_grad = False # Freeze parameters. They will not be updated during backpropagation
                
                extractors[key] = cnn
                
                total_concat_size += features_dim  # extractors[key].n_flatten
            elif key == "navigation":
                # Pass navigation features straight through to the MlpPolicy.
                extractors[key] = NavigatioNN(subspace, features_dim=subspace.shape[-1]) #nn.Identity()
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        #     print('Appended' ,key)
        # print(encoded_tensor_list)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
      
        return th.cat(encoded_tensor_list, dim=1)

