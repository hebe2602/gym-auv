import numpy as np

def random_from_interval(abs_max):
    rand = -abs_max + 2*abs_max*np.random.random()
    return rand


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

    # Parameters (See __init__.py) for description of values
    N = config['max_timesteps']
    ts = config['t_step_size']
    curr_vel_max = config['max_current_velocity']
    curr_vel_max_w = config['max_current_velocity_w']
    curr_angle_max_w = config['max_current_direction_w']
    Fu_max = config['max_ext_disturbance_Fu']
    Fu_max_w1 = config['max_ext_disturbance_Fu_w1']
    Fu_max_w2 = config['max_ext_disturbance_Fu_w2']
    Fv_max = config['max_ext_disturbance_Fv']
    Fv_max_w1 = config['max_ext_disturbance_Fv_w1']
    Fv_max_w2 = config['max_ext_disturbance_Fv_w2']
    Tr_max = config['max_ext_disturbance_Tr']
    Tr_max_w1 = config['max_ext_disturbance_Tr_w1']
    Tr_max_w2 = config['max_ext_disturbance_Tr_w2']

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
    


            


