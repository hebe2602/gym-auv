import gym
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ContrastiveNN_pretrained(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space:gym.spaces.Box,
                 output_channels:list = [2,4,4,6],
                 kernel_size:int      = 9,
                 n_sensors:int        = 180, 
                 features_dim:int     = 12,
                 only_f=False):

        super(ContrastiveNN_pretrained, self).__init__(observation_space, features_dim=features_dim)
        
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.output_channels = output_channels
        self.only_f = only_f

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
        len_flat = int(12 * self.output_channels[-1])
        
        #projection head
        self.linear_1 = nn.Sequential(
            nn.Linear(len_flat, 40),
            nn.ReLU(),
            nn.Linear(40, 12)
            
        )
        self.mlp_projection_head = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
        )


    def forward(self, x):

        for layer in self.feature_extractor:
            x = layer(x)

        for layer in self.linear_1:
            x = layer(x)

        if self.only_f: #Only encoder network(feature extractor f(.)) used for as feature extractor in gym-auv
            return x
        
        for layer in self.mlp_projection_head:
            x = layer(x)

        return x
    
class CCNN_2(BaseFeaturesExtractor):
 
    def __init__(self,
                 observation_space:gym.spaces.Box, 
                 n_sensors:int=180, 
                 output_channels:list=[3,2,1], 
                 kernel_size:int=45,
                 features_dim:int     = 12):

        super(CCNN_2, self).__init__(observation_space, features_dim=features_dim)
        
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
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
            # nn.ReLU(),
            nn.Flatten()
        )


  

    def forward(self, x):
    
        for layer in self.feature_extractor:
            x = layer(x)


        return x
    
class CCNN_2_p(BaseFeaturesExtractor):
 
    def __init__(self,
                 observation_space:gym.spaces.Box, 
                 n_sensors:int=180, 
                 output_channels:list=[3,2,1], 
                 kernel_size:int=45,
                 features_dim:int     = 12,
                 only_f=False):

        super(CCNN_2_p, self).__init__(observation_space, features_dim=features_dim)

        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.only_f = only_f

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
            # nn.ReLU(),
            nn.Flatten()
        )
        """
        self.mlp_projection_head = nn.Sequential(
            nn.Linear(12, 12),  # Adjust the input dimension to match the output of feature_extractor
            nn.ReLU(),
            nn.Linear(12, 12)  # Output dimension suitable for contrastive loss
        )
        
        
        self.mlp_projection_head = nn.Sequential(
            nn.Linear(12, 12),  # Adjust the input dimension to match the output of feature_extractor
            nn.ReLU(),
            nn.Linear(12, 4)  # Output dimension suitable for contrastive loss
        )
        
        self.mlp_projection_head = nn.Sequential(
            nn.Linear(12, 24),  # Adjust the input dimension to match the output of feature_extractor
            nn.ReLU(),
            nn.Linear(24, 12)  # Output dimension suitable for contrastive loss
        )

        
        
         """
        self.mlp_projection_head = nn.Sequential(
            nn.Linear(12, 24),  # Adjust the input dimension to match the output of feature_extractor
            nn.ReLU(),
            nn.Linear(24, 12)  # Output dimension suitable for contrastive loss
        )
        
        

       
        


  

    def forward(self, x):
    
        for layer in self.feature_extractor:
            x = layer(x)

        if self.only_f: #Only encoder network(feature extractor f(.)) used for as feature extractor in gym-auv
            return x
        
        for layer in self.mlp_projection_head:
            x = layer(x)
        return x
    

class ContrastiveNN_shallow(BaseFeaturesExtractor):
 
    def __init__(self,
                 observation_space:gym.spaces.Box,
                 n_sensors:int=180, 
                 output_channels:list=[1], 
                 kernel_size:int=45, 
                 padding:int=15,
                 stride:int=15,
                 features_dim:int     = 12,
                 only_f=False):
     
        super(ContrastiveNN_shallow, self).__init__(observation_space, features_dim=features_dim)
        self.n_sensors       = n_sensors
        self.kernel_size     = kernel_size
        self.padding         = padding 
        self.stride          = stride
        self.output_channels = output_channels
        self.only_f = only_f

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
        """
        len_flat =12
        self.linear = nn.Sequential(
            nn.Linear(len_flat, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        
        self.linear = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
       

        self.linear = nn.Sequential(
            nn.Linear(12, 8),  # Adjust the input dimension to match the output of feature_extractor
            nn.ReLU(),
            nn.Linear(8, 4)  # Output dimension suitable for contrastive loss
        )
        """
        self.linear = nn.Sequential(
            nn.Linear(12, 24),  # Adjust the input dimension to match the output of feature_extractor
            nn.ReLU(),
            nn.Linear(24, 12)  # Output dimension suitable for contrastive loss
        )
        """
        self.linear = nn.Sequential(
            nn.Linear(12, 12),  # Adjust the input dimension to match the output of feature_extractor
            nn.ReLU(),
            nn.Linear(12, 4)  # Output dimension suitable for contrastive loss
        )
        """


    def forward(self, x):
        
        for layer in self.feature_extractor:
            x = layer(x)
        if self.only_f: #Only encoder network(feature extractor f(.)) used for as feature extractor in gym-auv
            return x
        for layer in self.linear:
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

    def __init__(self, observation_space: gym.spaces.Dict, sensor_dim : int = 180, features_dim: int = 12, kernel_overlap : float = 0.05):
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
                """
                #cnn_path='gym_auv/utils/contrastive_learner.json'
                #cnn_path='gym_auv/utils/contrastive_learner_rot_noise_b256_e40.json'
                #cnn_path = 'gym_auv/utils/contrastive_learner_random_rot.json'
                #cnn_path = 'gym_auv/utils/contrastive_learner_b256_e20_agaussian_noiseß.json'
                #cnn_path = 'gym_auv/utils/ContrastiveNN_b32_e40_anoise_x.json'
                #cnn_path = 'gym_auv/utils/ContrastiveNN_b256_e40_anoise_x_no_val.json'
                cnn_path = 'gym_auv/utils/contrastive_model_only_f_b256_e50_aanoise02_x5_no_val_projh212_norm_l1.json'
                cnn = ContrastiveNN_pretrained(observation_space=subspace, 
                                           n_sensors=sensor_dim, 
                                           output_channels=[2,4,4,6], 
                                           kernel_size=9,
                                           features_dim=features_dim,
                                           only_f=True) #12
                
               
                """
                

                

                
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b128_e100_anoise_(mean:2,std:rand:5-10)_x2.json'
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b128_e100_anoise_(mean:2,std:rand:5-10)_rot+-10_x4.json'
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b128_e100_anoise_(mean:2,std:rand:5-10)_rot+-10_x4_norm.json'
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b32_e40_anoise_x.json'
                
                
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b256_e60_aanoise_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b128_e40_aanoise05_x_no_val_projh212_norm.json'
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b256_e50_aanoise05_x5_no_val_projh212_norm_l1.json' #forrige true
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b256_e20_aanoise_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b128_e20_aanoise02_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b256_e40_aanoise02_x5_no_val_projh212_norm_l1.json'
                cnn_path = 'gym_auv/utils/ccnn_2_projectionhead_only_f_b256_e40_aanoise05_x5_no_val_projh212_norm_l1.json'
                cnn = CCNN_2_p(observation_space=subspace,
                            n_sensors=sensor_dim, 
                            output_channels=[3,2,1], 
                            kernel_size=45,
                            features_dim=features_dim,
                            only_f=True) 
                
                
                """
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e40_anoise_x_no_val_projh4.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b16_e20_anoise_x_no_val_projh12.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b16_e20_anoise_x_no_val_projh12_norm.json'
                cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e20_anoise_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e20_anoise_rot_rand15_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e20_anoisedet_x2_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b512_e20_aanoise_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b128_e20_aanoise_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b64_e20_aanoise_x_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b128_e20_aanoise_x2_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b128_e20_aanoise_x5_no_val_projh12_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b128_e20_aanoise05_x_no_val_projh4_norm.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e20_anoise_rot_rand15_x_no_val_projh12_norm copy.json'

                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e20_aanoise02_x_no_val_projh212_norm_l1.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e20_aanoise08_x_no_val_projh212_norm_l1.json'
                #cnn_path = 'gym_auv/utils/contrastive_model_shallow_only_f_b256_e20_aanoise04_x_no_val_projh212_norm_l1.json'
                cnn = ContrastiveNN_shallow(observation_space=subspace, 
                                            n_sensors=sensor_dim, 
                                            output_channels=[1], 
                                            kernel_size=45,
                                            padding=15,
                                            stride=15,
                                            features_dim=features_dim,
                                            only_f=True) 
                
                """
                pretrained_dict = th.load(cnn_path)
                model_dict = cnn.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # 1. filter out unnecessary keys
                print('\nLoading pretrained LidarCNN from', cnn_path)
                print('Pretraind layers:', [k for k in pretrained_dict.keys()])

                model_dict.update(pretrained_dict) # 2. overwrite entries in the existing state dict
                cnn.load_state_dict(pretrained_dict) # 3. load the new state dict

                for param in cnn.parameters():
                    param.requires_grad = False# Freeze parameters. They will not be updated during backpropagation
                
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
