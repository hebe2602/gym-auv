from acados_template import AcadosOcp, AcadosOcpSolver
from gym_auv.utils.ship_model import export_ship_PSF_model
import numpy as np
import time
from casadi import log
from shapely.geometry.polygon import Polygon

 

class TestFilter:
   """
   A filter that adds or subtracts the speed input in play mode.
   """

   def __init__(self, add_input):
        self.add_input = add_input

   def filter(self, input):
      input[0] += self.add_input
      return input


class SafetyFilter:
      """
      Saftey filter class - sets up a predictive safety filter using the acados solver.
      """
      def __init__(self, env, rank, model_type, max_detected_rays_per_sector = 5, PSF_max_detect_distance = 30):
            """
            Initialize the filter with the ship dynamics model, constraints and solver options.
            """

            ocp = AcadosOcp()
            self.rank = rank
            ocp.code_export_directory = 'c_generated_code/c_generated_code_' + str(rank)
            self.env = env
            self.diff_u = 0.0
            self.n_static_obst = 0
            self.n_moving_obst = 0
            if hasattr(env, 'n_static_obst'):
                  self.n_static_obst = env.n_static_obst
            if hasattr(env, 'n_moving_obst'):
                  self.n_moving_obst = env.n_moving_obst
            self.n_obst = self.n_static_obst + self.n_moving_obst
            self.n_sectors = env.config["n_sectors"]
            self.n_sensors_per_sector = env.config["n_sensors_per_sector"]
            self.max_detected_rays_per_sector = max_detected_rays_per_sector
            self.max_detected_rays = max_detected_rays_per_sector*self.n_sectors
            self.PSF_max_detect_distance = PSF_max_detect_distance
            self.detected_ray_point_avoidance_radius = 8.0
            self.lidar_detection = env.config["lidar_obstacle_detection"]
            self.infeasible_solution = False


            # set model
            model = export_ship_PSF_model(model_type=model_type, max_detected_rays=self.max_detected_rays,
                                           n_obstacles=self.n_obst, lidar_detection=self.lidar_detection)
            ocp.model = model

            self.N = 50
            self.T_s = 0.5
            nx = model.x.size()[0]
            nu = model.u.size()[0]
            ny = nu
            nb = 3
            if self.lidar_detection:
                  nh = self.max_detected_rays
                  nh_e = self.max_detected_rays + 1
            else:
                  nh = self.n_obst
                  nh_e = self.n_obst + 1
            T_f = self.N*self.T_s

            # set dimensions
            ocp.dims.N = self.N
            

            # set cost type
            ocp.cost.cost_type_0 = 'LINEAR_LS'

            Vx_0 = np.zeros((ny,nx))
            ocp.cost.Vx_0 = Vx_0
            Vu_0 = np.eye(ny)
            ocp.cost.Vu_0 = Vu_0
            
            # Bounds on inputs
            F_u_max = 2.0
            F_r_max = 0.2
            
            # cost weights on inputs. Higher penalization on surge-thrust deviation to encourage movement

            gamma_F_u = 1.0
            gamma_F_r = 1.0e-2

            # Normalizing weight on F_u and F_r
            F_r_normalization = (1.0/F_r_max)**2
            F_u_normalization = (1.0/F_u_max)**2
            W_0 = np.eye(ny)
            W_0[0,0] = gamma_F_u*F_u_normalization
            W_0[-1,-1] = gamma_F_r*F_r_normalization

            # Set linear least squares cost matrix
            ocp.cost.W_0 = W_0
            
            u0 = np.array([0,0]) #.reshape(2,1)
            yref_0 = u0
            ocp.cost.yref_0 = yref_0

            #set slack variables cost
            ocp.cost.Zl = 0*np.ones((nb + nh,))
            ocp.cost.Zu = 0*np.ones((nb + nh,))
            ocp.cost.zl = 100*np.ones((nb + nh,))
            ocp.cost.zu = 100*np.ones((nb + nh,))
            ocp.cost.Zl_e = 0*np.ones((nh_e,))
            ocp.cost.Zu_e = 0*np.ones((nh_e,))
            ocp.cost.zl_e = 100*np.ones((nh_e,))
            ocp.cost.zu_e = 100*np.ones((nh_e,))


            #state constraints
            uv_max = 2.0
            r_max = 0.15

            ocp.constraints.lbx = np.array([-uv_max,-uv_max,-r_max])
            ocp.constraints.ubx = np.array([+uv_max,+uv_max,+r_max])
            ocp.constraints.idxbx = np.array([3,4,5])
            ocp.constraints.idxsbx = np.array([0,1,2])

            #input constraints
            ocp.constraints.lbu = np.array([0.0,-F_r_max])
            ocp.constraints.ubu = np.array([+F_u_max,+F_r_max])
            ocp.constraints.idxbu = np.array([0,1])
            

            #Safety zone for rendering
            env.vessel.safety_zone = Polygon([(-1, -1), 
                                                (-1, 1), 
                                                (1, 1), 
                                                (1, -1), 
                                                (-1, -1), 
                                                ])
            
            
            #Terminal set for rendering 
            # env.vessel.terminal_set = Polygon([ (-ocp.constraints.lbx_e[0], -ocp.constraints.lbx_e[1]), 
            #                                     (-ocp.constraints.lbx_e[0], ocp.constraints.lbx_e[1]), 
            #                                     (ocp.constraints.lbx_e[0], ocp.constraints.lbx_e[1]), 
            #                                     (ocp.constraints.lbx_e[0], -ocp.constraints.lbx_e[1]), 
            #                                     (-ocp.constraints.lbx_e[0], -ocp.constraints.lbx_e[1]), 
            #                                     ])
            env.vessel.terminal_set = env.vessel.safety_zone


            #Safe trajectory for rendering
            env.vessel.safe_trajectory = np.ndarray((self.N+1,nx))



            if self.lidar_detection:
                  #obstacle constraint
                  # Initialize all obstacle values to 1. Initializing to 0 would result in numerical error in solver because
                  # derivative of sqrt(x) is undefined for x = 0
                  p0 = 999*np.ones((3*self.max_detected_rays))
                  
                  #Set initial parameter values corresponding to obstacle radius to -1. Deactivates constraints
                  p0[-self.max_detected_rays:] = -50
                  
            else:
                  #obstacle constraints
                  self.obstacles = env.obstacles
                  p0 = np.zeros((3*self.n_obst))

                  #Moving obstacles
                  for i in range(self.n_moving_obst):
                        p0[3*i:3*i+2] = self.obstacles[i].position
                        p0[3*i+2] = self.obstacles[i].width + 2.0
                  
                  #Static obstacles
                  for i in range(self.n_moving_obst,self.n_obst):
                        p0[3*i:3*i+2] = self.obstacles[i].position
                        p0[3*i+2] = self.obstacles[i].radius


            self.p = p0
            ocp.parameter_values = self.p
            ocp.constraints.lh = np.zeros((nh,))
            ocp.constraints.uh = 9999*np.ones((nh,))
            ocp.constraints.lh_e = np.zeros((nh_e,))
            ocp.constraints.lh_e[-1] = -1
            ocp.constraints.uh_e = 9999*np.ones((nh_e,))
            ocp.constraints.uh_e[-1] = 1
            ocp.constraints.idxsh = np.array(range(nh))
            ocp.constraints.idxsh_e = np.array(range(nh_e))


            #initial state
            ocp.constraints.x0 = np.array(env.vessel._state)


            # set options
            ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
            # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
            # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
            ocp.solver_options.integrator_type = 'IRK'
            # ocp.solver_options.print_level = 1
            ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 3
            ocp.solver_options.sim_method_newton_iter = 3
            ocp.solver_options.nlp_solver_step_length = 1.0
            ocp.solver_options.nlp_solver_tol_eq = 1e-6
            ocp.solver_options.nlp_solver_tol_stat = 1e-6

            # set prediction horizon
            ocp.solver_options.tf = T_f


            json_file = 'acados_ocp/acados_ocp_' + str(rank) + '.json'

            self.ocp_solver = AcadosOcpSolver(ocp, json_file = json_file)



      def update_obstacles_from_lidar(self, ray_dists, ray_angles, state):
            """Return updated obstacle parameter vector based on most recent sensor measurements"""
            
            
            obs_param_updated = 999*np.ones(3 * self.max_detected_rays)
            
            # Last "max_detected_rays" elements of obstacle vector correspond to radius, initialize to -1, same as deactivating constraints
            obs_param_updated[-self.max_detected_rays:] = -50.0

            # Find indices of sensor_dists less than max detection distance for safety filter
            detect_idxs = np.asarray(ray_dists <= self.PSF_max_detect_distance).nonzero()

            # If no ray detections within PSF_max_distance, return
            if len(detect_idxs[0]) == 0:
                  self.p = obs_param_updated
                  return

            # Add state[2] (heading) to get NED angles of detected rays
            ray_angles_NED = ray_angles + state[2]

            # Create 2D array with ray distances, corresponding sensor angles and indices
            detect_rays = np.row_stack([ray_dists[detect_idxs], ray_angles_NED[detect_idxs]])
            
            # Sort detected rays on distance, we only use the "max_detected_rays_per_sector" number of closest detected rays per sector
            closest_detected_rays = []

            for i in range(self.n_sectors):
                  start = i*self.n_sensors_per_sector
                  end = start + self.n_sensors_per_sector
                  
                  # Get detected rays within sector
                  sector_detect_rays = detect_rays[:, np.asarray((detect_idxs[0] >= start) & (detect_idxs[0] < end))]
                  
                  # If no detected rays within sector, continue to next iteration
                  if len(sector_detect_rays) == 0:
                        continue

                  # Sort detected rays within sector based on distance
                  sorted_sector_detect_rays = sector_detect_rays[:,np.argsort(sector_detect_rays[0])]

                  # Add (max_detected_rays_per_sector) distances and angles tuple to list
                  closest_detected_rays.extend(zip(sorted_sector_detect_rays[0,:self.max_detected_rays_per_sector], sorted_sector_detect_rays[1,:self.max_detected_rays_per_sector]))
 
            # index for assigning subsequent ray-detection obstacles to subsequent sets of parameter elements
            obs_param_idx = 0

            for ray in closest_detected_rays:
                  
                  dist = ray[0]
                  angle = ray[1]
                  # Set obstacle x-value
                  obs_param_updated[obs_param_idx] = state[0] + dist*np.cos(angle)

                  # Set obstacle y-value
                  obs_param_updated[obs_param_idx + self.max_detected_rays] = state[1] + dist*np.sin(angle)

                  # Set obstacle r-value
                  obs_param_updated[obs_param_idx + 2*self.max_detected_rays] = self.detected_ray_point_avoidance_radius

                  obs_param_idx += 1
                  
            self.p = obs_param_updated
            #print('Updated obstacle parameters: ', self.p[:-3])




      def filter(self, u, state):
            """
            Solve the filter for the current input. 

            Returns the calculated optimal input u0. 
            """      
            self.ocp_solver.cost_set(0,"yref",u)


            #print('Current state: ', state)
            # curr_pred = self.ocp_solver.get(1,'x')
            # print('Diff between current state and PSF prediction: ', state - curr_pred)
            status = self.ocp_solver.solve()

            #self.ocp_solver.print_statistics() # encapsulates: stat = self.ocp_solver.get_stats("statistics")

            for j in range(self.N+1):
                  self.env.vessel.safe_trajectory[j,:] = self.ocp_solver.get(j,'x')

            if status != 0:
                  # for i in range(self.N):
                  #      print(i, ': x: ', self.ocp_solver.get(i,'x'), ', u: ', self.ocp_solver.get(i,'u'))
                  #raise Exception(f'acados returned status {status}.')
                  # If acados returns error, return original input and disable safety filter
                  self.env.history['infeasible_solution'] = np.array([1])
                  print('Infeasible solution, returning original input and disabling safety filter.')
                  #self.env.vessel._use_safety_filter = False
                  print('Current state: ', state)
                  print('vessel_position', self.env.vessel.position)
                  print('obstacles', [obs.position for obs in self.env.obstacles])
                  print('path', self.env.path._points, self.env.path.length)
                  self.infeasible_solution = True
                  self.diff_u = np.zeros(2)
                  return u
            else:
                  new_u = self.ocp_solver.get(0, "u")

            self.diff_u = new_u - u
            #print('Initial u: ',u, ', new u: ', new_u, ', diff: ', self.diff_u)
            return new_u

      def update(self, state, nav_state):
            """
            Update the current state. 
            """
            #print(self.ocp_solver.get(1,'x') - state)
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)

            #Terminal set parameters
            ctp_heading = nav_state['look_ahead_heading_error']
            ctp = nav_state['closest_point']
            ctp_x = ctp[0]
            ctp_y = ctp[1]
            self.p[-3:] = np.array([ctp_x,ctp_y,ctp_heading])

            if self.lidar_detection:
                  for i in range(self.N + 1):
                        self.ocp_solver.set(i,'p',self.p)

            #Set moving obstacle parameters
            else:     
                  moving_obst_dist = [np.linalg.norm(state[:2] - moving_obst.position) for moving_obst in self.obstacles[:self.n_moving_obst]]
                  close_obs_idxs = np.where(np.array(moving_obst_dist) < 100)[0]
                  
                  pred_obst_pos = [self.obstacles[i].position for i in range(self.n_moving_obst)]
                  for i in range(self.N + 1):
                        for j in range(self.n_moving_obst):

                              #Update obstacle parameters for close obstacles
                              if j in close_obs_idxs:
                                    self.p[3*j:3*j+2] = pred_obst_pos[j]
                                    self.p[3*j+2] = self.obstacles[j].width + 2.0

                                    #Predict future position
                                    index = int(np.floor(self.obstacles[j].waypoint_counter))
                                    obst_speed = self.obstacles[j].trajectory_velocities[index]
                                    pred_obst_pos[j] = pred_obst_pos[j] + [self.T_s*obst_speed[k] for k in range(2)]

                              #Make obstacles far away negligible
                              else:
                                    self.p[3*j+2] = -5.0
                  
                        self.ocp_solver.set(i,'p',self.p)
      def __del__(self):
            del self.ocp_solver
            

