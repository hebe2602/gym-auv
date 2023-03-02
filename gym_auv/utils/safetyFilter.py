from acados_template import AcadosOcp, AcadosOcpSolver
from acadostesting_v02_export.models.ship_model import export_ship_model
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
      def __init__(self, env):
            """
            Initialize the filter with the ship dynamics model, constraints and solver options.
            """

            ocp = AcadosOcp()
            self.env = env

            # set model
            model = export_ship_model(model_type='simplified')
            ocp.model = model

            T_s = 0.5
            nx = model.x.size()[0]
            nu = model.u.size()[0]
            ny = nu
            self.N = 50
            T_f = self.N*T_s

            # set dimensions
            ocp.dims.N = self.N


            # set cost
            ocp.cost.cost_type_0 = 'LINEAR_LS'

            Vx_0 = np.zeros((ny,nx))
            ocp.cost.Vx_0 = Vx_0
            Vu_0 = np.eye(ny)
            ocp.cost.Vu_0 = Vu_0

            F_u_max = 2.0
            F_r_max = 0.15

            W_0 = np.eye(ny)
            W_0[-1,-1] = F_u_max/F_r_max
            ocp.cost.W_0 = W_0
            
            u0 = np.array([0,0]) #.reshape(2,1)
            yref_0 = u0
            ocp.cost.yref_0 = yref_0

            #set slack variables cost
            ocp.cost.Zl = 0*np.ones((4,))
            ocp.cost.Zu = 0*np.ones((4,))
            ocp.cost.zl = 100*np.ones((4,))
            ocp.cost.zu = 100*np.ones((4,))
            ocp.cost.Zl_e = 0*np.ones((2,))
            ocp.cost.Zu_e = 0*np.ones((2,))
            ocp.cost.zl_e = 100*np.ones((2,))
            ocp.cost.zu_e = 100*np.ones((2,))


            #state constraints
            uv_max = 2.0
            r_max = 0.2

            ocp.constraints.lbx = np.array([-uv_max,-uv_max,-r_max])
            ocp.constraints.ubx = np.array([+uv_max,+uv_max,+r_max])
            ocp.constraints.idxbx = np.array([3,4,5])
            ocp.constraints.idxsbx = np.array([0,1,2])

            #input constraints
            ocp.constraints.lbu = np.array([-F_u_max,-F_r_max])
            ocp.constraints.ubu = np.array([+F_u_max,+F_r_max])
            ocp.constraints.idxbu = np.array([0,1])

            
            #terminal set
            #goal = env.path.end
            #xy_max_goal = 100
            #ocp.constraints.lbx_e = np.array([goal[0]-xy_max_goal,goal[1]-xy_max_goal,-uv_max,-uv_max,-r_max])
            #ocp.constraints.ubx_e = np.array([goal[0]+xy_max_goal,goal[1]+xy_max_goal,+uv_max,+uv_max,+r_max])

            # ocp.constraints.lbx_e = np.array([-xy_max,-xy_max,-uv_max,-uv_max,-r_max])
            # ocp.constraints.ubx_e = np.array([+xy_max,+xy_max,+uv_max,+uv_max,+r_max])
            # ocp.constraints.idxbx_e = np.array([0,1,3,4,5])
            
            # ocp.constraints.idxsbx_e = np.array([0,1,2,3,4])
            

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


            #Initialize obstacle and track params
            p0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
            p0[:2] = env.obstacles[0].position
            p0[2] = env.obstacles[0].radius


            # initialize track params
            
            ocp.parameter_values = p0
            
            ocp.constraints.lh = np.array([0.0])
            ocp.constraints.uh = np.array([500.0])
            ocp.constraints.lh_e = np.array([0.0,-1.0])
            ocp.constraints.uh_e = np.array([500.0,1.0])
            ocp.constraints.idxsh = np.array([0])
            ocp.constraints.idxsh_e = np.array([0,1])


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

            self.ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

      
      def filter(self, u, state):
            """
            Solve the filter for the current input. 

            Returns the calculated optimal input u0. 
            """      

            self.ocp_solver.cost_set(0,"yref",u)
            print('Current state: ', state)
            curr_pred = self.ocp_solver.get(1,'x')
            print('Diff between current state and PSF prediction: ', state - curr_pred)

            status = self.ocp_solver.solve()
            self.ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

            for j in range(self.N+1):
                  self.env.vessel.safe_trajectory[j,:] = self.ocp_solver.get(j,'x')

            if status != 0:
                  for i in range(self.N):
                       print(i, ': x: ', self.ocp_solver.get(i,'x'), ', u: ', self.ocp_solver.get(i,'u'))
                  raise Exception(f'acados returned status {status}.')
            
            return self.ocp_solver.get(0, "u")

      def update(self, state, obstacles, nav_state):
            """
            Update the current state. 
            """
            
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)
            
            obs_x = obstacles[0].position[0]
            obs_y = obstacles[0].position[1]
            obs_r = obstacles[0].radius
            ctp_heading = nav_state['target_heading']
            ctp = nav_state['closest_point']
            ctp_x = ctp[0]
            ctp_y = ctp[1]
            p = np.array([obs_x,obs_y,obs_r,ctp_x,ctp_y,ctp_heading])
            # for i in range(self.N + 1):
            #      self.ocp_solver.set(i,'p',p)
            self.ocp_solver.set(self.N,'p',p)
            
            


