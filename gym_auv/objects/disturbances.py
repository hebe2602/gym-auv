import numpy as np


# This file contains the code and functions for generating the randomized environmental disturbances that affect the ship. The disturbances consist
# of a slowly varying ocean current modelled as a random walk process, and generalized surge force, sway force, and yaw moment disturbances modelled
# as processes with a slow-varying random walk component and a white noise component  





def random_from_interval(abs_max):
    rand = -abs_max + 2*abs_max*np.random.random()
    return rand


# Parameters
max_current_velocity = 0.09                      # Maximum velocity of water current
max_current_velocity_w = 0.02                   # Max value for white noise w that is integrated to generate random walk current velocity signal
max_current_direction_w = 0.01                  # Max value for white noise w that is integrated to generate random walk current direction signal
max_ext_disturbance_Fu = 0.25                    # Max value for external surge force disturbance signal
max_ext_disturbance_Fv = 0.25                    # Max value for external sway force disturbance signal
max_ext_disturbance_Tr = 0.003                    # Max value for external yaw moment disturbance signal
max_ext_disturbance_Fu_w1 = 0.025                 # Max value for white noise w1 that is integrated to generate random walk componenet for surge force disturbance signal
max_ext_disturbance_Fu_w2 = 0.01                 # Max value for w2 that generates white noise component for surge force disturbance signal
max_ext_disturbance_Fv_w1 = 0.025                 # Max value for white noise w1 that is integrated to generate random walk component for sway force disturbance signal
max_ext_disturbance_Fv_w2 = 0.01                 # Max value for w2 that generates white noise component sway force disturbance signal
max_ext_disturbance_Tr_w1 = 0.0003                 # Max value for white noise w1 that is integrated to generate random walk component for yaw moment disturbance signal
max_ext_disturbance_Tr_w2 = 0.00003                 # Max value for w2 that generates white noise component for yaw moment disturbance signal

def generate_disturbances(config:dict):
    """
    Function for generating vector sequences of environmental disturbance values

    Parameters:
    config: Dictionary with configuration values

    Returns:
    current_velocities: numpy matrix containing a random walk sequence of ocean current velocities
    disturbance_forces: numpy matrix containing a sequence of disturbance forces generated using 
    a random walk component and a white noise component
    """

    # Parameters
    N = config['max_timesteps']
    ts = config['t_step_size']
    curr_vel_max = max_current_velocity
    curr_vel_max_w = max_current_velocity_w
    curr_angle_max_w = max_current_direction_w
    Fu_max = max_ext_disturbance_Fu
    Fu_max_w1 = max_ext_disturbance_Fu_w1
    Fu_max_w2 = max_ext_disturbance_Fu_w2
    Fv_max = max_ext_disturbance_Fv
    Fv_max_w1 = max_ext_disturbance_Fv_w1
    Fv_max_w2 = max_ext_disturbance_Fv_w2
    Tr_max = max_ext_disturbance_Tr
    Tr_max_w1 = max_ext_disturbance_Tr_w1
    Tr_max_w2 = max_ext_disturbance_Tr_w2

    # Initialize disturbance values
    curr_vel = np.zeros(N+1)
    curr_angle = np.zeros(N+1)
    Fu = np.zeros(N+1)
    Fv = np.zeros(N+1)
    Tr = np.zeros(N+1)

    current_velocities = np.zeros((3,N))
    disturbance_forces = np.zeros((3,N))

    curr_vel[0] = random_from_interval(curr_vel_max)  
    curr_angle[0] = random_from_interval(np.pi)             
    Fu[0] = random_from_interval(Fu_max)             
    Fv[0] = random_from_interval(Fv_max)
    Tr[0] = random_from_interval(Tr_max)
    

    for i in range(N):
        
        # Propagating random walk current velocity and current angle signals
        vel_w = random_from_interval(curr_vel_max_w)
        angle_w = random_from_interval(curr_angle_max_w)

        curr_vel[i+1] = np.clip(curr_vel[i] + ts*vel_w, -curr_vel_max, curr_vel_max)
        curr_angle[i+1] = np.clip(curr_angle[i] + ts*angle_w, -np.pi, np.pi)

        # Propagating random walk component of disturbance force signals
        Fu_w1 = random_from_interval(Fu_max_w1)
        Fv_w1 = random_from_interval(Fv_max_w1)
        Tr_w1 = random_from_interval(Tr_max_w1)

        Fu[i+1] = np.clip(Fu[i] + ts*Fu_w1, -Fu_max, Fu_max)
        Fv[i+1] = np.clip(Fv[i] + ts*Fv_w1, -Fv_max, Fv_max)
        Tr[i+1] = np.clip(Tr[i] + ts*Tr_w1, -Tr_max, Tr_max)
        
        # Updating current velocities matrix with decomposed NED current velocities
        current_velocities[:,i] = np.array([curr_vel[i]*np.cos(curr_angle[i]),
                                      curr_vel[i]*np.sin(curr_angle[i]),
                                      0.0])
        
        # Adding pure white noise components and updating disturbance force vector
        Fu_w2 = random_from_interval(Fu_max_w2)
        Fv_w2 = random_from_interval(Fv_max_w2)
        Tr_w2 = random_from_interval(Tr_max_w2)
        disturbance_forces[:,i] = np.array([np.clip(Fu[i] + Fu_w2, -Fu_max, Fu_max),
                                         np.clip(Fv[i] + Fv_w2, -Fv_max, Fv_max),
                                         np.clip(Tr[i] + Tr_w2, -Tr_max, Tr_max)])

    return current_velocities, disturbance_forces
    


            


