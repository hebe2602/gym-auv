import numpy as np

def random_from_interval(abs_max):
    rand = -abs_max + 2*abs_max*np.random.random()
    return rand


class disturbances():
    
    def __init__(self, config:dict) -> None:
        # Initialize private attributes (See __init__.py) for description of values
        self._ts = config['t_step_size']
        self._curr_vel_max = config['max_current_velocity']
        self._curr_vel_max_w = config['max_current_velocity_w']
        self._curr_angle_max_w = config['max_current_direction_w']
        self._Fu_max = config['max_ext_disturbance_Fu']
        self._Fu_max_w1 = config['max_ext_disturbance_Fu_w1']
        self._Fu_max_w2 = config['max_ext_disturbance_Fu_w2']
        self._Fv_max = config['max_ext_disturbance_Fv']
        self._Fv_max_w1 = config['max_ext_disturbance_Fv_w1']
        self._Fv_max_w2 = config['max_ext_disturbance_Fv_w2']
        self._Tr_max = config['max_ext_disturbance_Tr']
        self._Tr_max_w1 = config['max_ext_disturbance_Tr_w1']
        self._Tr_max_w2 = config['max_ext_disturbance_Tr_w2']

        # Initialize disturbance values
        self.curr_vel = random_from_interval(self._curr_vel_max)  # current velocity
        self.curr_angle = random_from_interval(np.pi)             # Current angle
        self.Fu = random_from_interval(self._Fu_max)              # Disturbance forces
        self.Fv = random_from_interval(self._Fv_max)
        self.Tr = random_from_interval(self._Tr_max)
        
    def propagate_current(self):
        """
        Propagate brownian motion water current velocity and water current angle signals
        """
        ts = self._ts
        vel_w = random_from_interval(self._curr_vel_max_w)
        angle_w = random_from_interval(self._curr_angle_max_w)

        self.curr_vel = np.clip(self.curr_vel + ts*vel_w, -self._curr_vel_max, self._curr_vel_max)
        self.curr_angle = np.clip(self.curr_angle + ts*angle_w, -np.pi, np.pi)
    
    def propagate_disturbance_forces(self):
        """
        Propagate brownian motion disturbance force signals
        """
        ts = self._ts
        Fu_w1 = random_from_interval(self._Fu_max_w1)
        Fv_w1 = random_from_interval(self._Fv_max_w1)
        Tr_w1 = random_from_interval(self._Tr_max_w1)

        self.Fu = np.clip(self.Fu + ts*Fu_w1, -self._Fu_max, self._Fu_max)
        self.Fv = np.clip(self.Fv + ts*Fv_w1, -self._Fv_max, self._Fv_max)
        self.Tr = np.clip(self.Tr + ts*Tr_w1, -self._Tr_max, self._Tr_max)

    def Get(self):
        """
        Get current velocity component vector, current angle and disturbance forces (with pure white noise component added)
        """

        current_vector = np.array([self.curr_vel*np.cos(self.curr_angle), self.curr_vel*np.sin(self.curr_angle), 0.0])

        Fu_w2 = random_from_interval(self._Fu_max_w2)
        Fv_w2 = random_from_interval(self._Fv_max_w2)
        Tr_w2 = random_from_interval(self._Tr_max_w2)

        Fu = np.clip(self.Fu + Fu_w2, -self._Fu_max, self._Fu_max)
        Fv = np.clip(self.Fv + Fv_w2, -self._Fv_max, self._Fv_max)
        Tr = np.clip(self.Tr + Tr_w2, -self._Tr_max, self._Tr_max)
        F = np.array([Fu, Fv, Tr])

        return current_vector, F
            


