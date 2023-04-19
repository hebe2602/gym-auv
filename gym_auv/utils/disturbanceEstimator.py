import numpy as np
import constants as const


# This file contains code for the disturbanceEstimator object, which is used to estimate the environmental disturbances affecting the ship
# The implementation is based on the paper found at https://arxiv.org/abs/2211.08360, created by Daniel Menges and Adil Rasheed


# Parameters
Gamma_1 = 1.0             # Adaptive gain for surge force disturbance estimate
Gamma_2 = 1.0             # Adaptive gain for sway force disturbance estimate
Gamma_3 = 1.0             # Adaptive gain for yaw moment disturbance estimate

# Constants
M_inv = const.M_inv       # Inverse of 3-DOF cybership II model mass matrix
k_11 = M_inv[0,0]
k_22 = M_inv[1,1]
k_23 = M_inv[1,2]
k_32 = M_inv[2,1]
k_33 = M_inv[2,2]

rho = 1 - (k_23*k_32 / (k_22*k_33))

T = np.array([[Gamma_1*rho/k_11, 0, 0],
              [0, Gamma_2/k_22, -Gamma_2*k_23/(k_22*k_33)],
              [0, -Gamma_3*k_32/(k_22*k_33), Gamma_3/k_33]])

# Initializations
T_d_init = np.array([0.0, 0.0, 0.0])             # Initialization of disturbance force estimate T_d
zeta_init = np.array([0.0, 0.0, 0.0])            # Initialization of observer variable zeta

class disturbanceEstimator():
    """
    Class that defines environmental disturbance estimator
    """
    def __init__(self, config:dict) -> None:
        
        self.ts = config['t_step_size']
        self.zeta = zeta_init
        self.T_d = T_d_init

    def update(self, state, state_dot):
        """
        Updates disturbance estimate T_d and observer variable zeta
        """

        nu = state[3:]
        nu_dot = state_dot[3:]

        # Update T_d
        self.T_d = self.zeta + T*nu

        # Update zeta
        zeta_dot = -T*nu_dot * (self.zeta + T*nu)
        self.zeta = self.zeta + self.ts * zeta_dot
    
    def get(self):
        """
        Returns current estimate of ship disturbance force
        """
        return self.T_d