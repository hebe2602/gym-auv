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
      def __init__(self, env, rank, model_type):
            """
            Initialize the filter with the ship dynamics model, constraints and solver options.
            """

            ocp = AcadosOcp()
            ocp.code_export_directory = 'c_generated_code/c_generated_code_' + str(rank)
            self.env = env
            self.diff_u = 0.0
            n_obstacles = env.n_obstacles

            # set model
            model = export_ship_PSF_model(model_type=model_type, n_obstacles=n_obstacles)
            ocp.model = model

            self.N = 50
            T_s = 0.5
            nx = model.x.size()[0]
            nu = model.u.size()[0]
            ny = nu
            nh = n_obstacles
            nb = 3
            nh_e = n_obstacles# + 1
            T_f = self.N*T_s

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
            F_r_max = 0.15
            
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
            r_max = 0.2

            ocp.constraints.lbx = np.array([-uv_max,-uv_max,-r_max])
            ocp.constraints.ubx = np.array([+uv_max,+uv_max,+r_max])
            ocp.constraints.idxbx = np.array([3,4,5])
            ocp.constraints.idxsbx = np.array([0,1,2])

            #input constraints
            ocp.constraints.lbu = np.array([-0.2,-F_r_max])
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


            #obstacle constraint
            p0 = np.zeros((3 + 3*n_obstacles))
            for i in range(n_obstacles):
                  p0[3*i:3*i+2] = env.obstacles[i].position
                  p0[3*i+2] = env.obstacles[i].radius

            self.p = p0
            ocp.parameter_values = self.p
            ocp.constraints.lh = np.zeros((n_obstacles,))
            ocp.constraints.uh = 500*np.ones((n_obstacles,))
            ocp.constraints.lh_e = np.zeros((nh_e,))
            #ocp.constraints.lh_e[-1] = -1
            ocp.constraints.uh_e = 500*np.ones((nh_e,))
            #ocp.constraints.uh_e[-1] = 1
            ocp.constraints.idxsh = np.array(range(nh_e))
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
            

      
      def filter(self, u, state):
            """
            Solve the filter for the current input. 

            Returns the calculated optimal input u0. 
            """      

            self.ocp_solver.cost_set(0,"yref",u)

            # print('Current state: ', state)
            # curr_pred = self.ocp_solver.get(1,'x')
            # print('Diff between current state and PSF prediction: ', state - curr_pred)

            status = self.ocp_solver.solve()
            #self.ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

            for j in range(self.N+1):
                  self.env.vessel.safe_trajectory[j,:] = self.ocp_solver.get(j,'x')

            if status != 0:
                  for i in range(self.N):
                       print(i, ': x: ', self.ocp_solver.get(i,'x'), ', u: ', self.ocp_solver.get(i,'u'))
                  raise Exception(f'acados returned status {status}.')

            new_u = self.ocp_solver.get(0, "u")
            self.diff_u = new_u - u
            print('Initial u: ',u, ', new u: ', new_u)
            return new_u

      def update(self, state, nav_state):
            """
            Update the current state. 
            """
            
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)
      
            ctp_heading = nav_state['look_ahead_heading_error']
            ctp = nav_state['closest_point']
            ctp_x = ctp[0]
            ctp_y = ctp[1]
            self.p[-3:] = np.array([ctp_x,ctp_y,ctp_heading])
            # for i in range(self.N + 1):
            #      self.ocp_solver.set(i,'p',p)
            self.ocp_solver.set(self.N,'p',self.p)
            
            


