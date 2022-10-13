import gym
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarCNN_pretrained(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space:gym.spaces.Box,
                 output_channels:list = [4,4],
                 kernel_size:int      = 9,
                 n_sensors:int        = 180, 
                 features_dim:int     =   1):

        super(LidarCNN_pretrained, self).__init__(observation_space, features_dim=features_dim)
        
        self.mean = 146.29372782863646  # mean found during training
        self.std  = 19.269065686163174  # standard deviation found during traning

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
        # Output of feature_extractor is [N, C_out, L/(2^num_maxpool2x1)]
        len_flat = int(np.ceil(self.n_sensors/2**3) * self.output_channels[-1])
        self.regressor = nn.Sequential(
            nn.Linear(len_flat, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU()
        )

    def forward(self, x):
        # Normalica sensor measurements before forward pass
        x = (x - self.mean)/self.std

        for layer in self.feature_extractor:
            x = layer(x)
        for layer in self.regressor:
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

    def __init__(self, observation_space: gym.spaces.Dict, sensor_dim : int = 180, features_dim: int = 1, kernel_overlap : float = 0.05):
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

                cnn = LidarCNN_pretrained(observation_space=subspace, 
                                          n_sensors=sensor_dim, 
                                          output_channels=[4,4], 
                                          kernel_size=9,
                                          features_dim=features_dim)

                print('Loading pretrained LidarCNN')
                cnn.load_state_dict(th.load('gym_auv/utils/cnn_1_pretrained.json'))
                
                for param in cnn.parameters():
                    param.requires_grad = False # Freeze parameters. They will not be updated during backpropagation
                
                extractors[key] = cnn
                
                total_concat_size += features_dim 

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
            print('Appended' ,key)
        print(encoded_tensor_list)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
