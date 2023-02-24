from acados_template import AcadosOcp, AcadosOcpSolver
from acadostesting_v02_export.models.ship_model import export_ship_model
import numpy as np
import time
from casadi import log



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

            # set model
            model = export_ship_model()
            ocp.model = model

            T_s = 0.5
            nx = model.x.size()[0]
            nu = model.u.size()[0]
            ny = nu
            N = 50
            T_f = N*T_s

            # set dimensions
            ocp.dims.N = N


            # set cost
            # the 'EXTERNAL' cost type can be used to define general cost terms
            # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
            ocp.cost.cost_type_0 = 'LINEAR_LS'
            u0 = np.array([0,0]) #.reshape(2,1)
            ocp.parameter_values = u0

            Vx_0 = np.zeros((ny,nx))
            ocp.cost.Vx_0 = Vx_0
            Vu_0 = np.eye(ny)
            ocp.cost.Vu_0 = Vu_0

            F_u_max = 2.0
            F_r_max = 0.15

            W_0 = np.eye(ny)
            W_0[-1,-1] = F_u_max/F_r_max
            ocp.cost.W_0 = W_0
            
            yref_0 = u0
            ocp.cost.yref_0 = yref_0

            ocp.cost.Zl = 0*np.ones((nx-1,))
            ocp.cost.Zu = 0*np.ones((nx-1,))
            ocp.cost.zl = 100*np.ones((nx-1,))
            ocp.cost.zu = 100*np.ones((nx-1,))
            ocp.cost.Zl_e = 0*np.ones((nx-1,))
            ocp.cost.Zu_e = 0*np.ones((nx-1,))
            ocp.cost.zl_e = 100*np.ones((nx-1,))
            ocp.cost.zu_e = 100*np.ones((nx-1,))
            # set constraints

            xy_max = 100.0
            uv_max = 2.0
            r_max = 0.2

            ocp.constraints.lbx = np.array([-xy_max,-xy_max,-uv_max,-uv_max,-r_max])
            ocp.constraints.ubx = np.array([+xy_max,+xy_max,+uv_max,+uv_max,+r_max])
            ocp.constraints.idxbx = np.array([0,1,3,4,5])
            ocp.constraints.lbx_e = 0.25*np.array([-xy_max,-xy_max,-uv_max,-uv_max,-r_max])
            ocp.constraints.ubx_e = 0.25*np.array([+xy_max,+xy_max,+uv_max,+uv_max,+r_max])
            ocp.constraints.idxbx_e = np.array([0,1,3,4,5])
            ocp.constraints.idxsbx = np.array([0,1,2,3,4])
            ocp.constraints.idxsbx_e = np.array([0,1,2,3,4])
            
            # (x - x_obj)**2 + (y - y_obj)**2 >= radius_obj

            ocp.constraints.lbu = np.array([0,-F_r_max])
            ocp.constraints.ubu = np.array([+F_u_max,+F_r_max])
            ocp.constraints.idxbu = np.array([0,1])


            # x = model.x
            # obst = env.obstacles[0]
            # pos = obst.position
            # r = obst.radius

            # circle = (x[0] - pos[0])**2 + (x[1] - pos[1])**2 - r**2
            
            # ocp.constraints.lh = np.array([0.0])
            # ocp.constraints.uh = np.array([100000.0])
            # ocp.model.con_h_expr = circle
            # ocp.constraints.lh_e = ocp.constraints.lh
            # ocp.constraints.uh_e = ocp.constraints.uh 

            # ocp.model.con_h_expr_e = ocp.model.con_h_expr


            ocp.constraints.x0 = np.array(env.vessel._state)
            #ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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

            if status != 0:
                  for i in range(50):
                       print(i, ': x: ', self.ocp_solver.get(i,'x'), ', u: ', self.ocp_solver.get(i,'u'))
                  raise Exception(f'acados returned status {status}.')
            
            return self.ocp_solver.get(0, "u")

      def update(self, state):
            """
            Update the current state. 
            """
            # uv_max = 2.0
            # xy_max_e = 4
            # N = 50
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)
            # self.ocp_solver.constraints_set(N,'lbx',np.array([state[0] + 5 - xy_max_e, state[1] + 5 - xy_max_e, -uv_max, -uv_max]))
            # self.ocp_solver.constraints_set(N,'ubx',np.array([state[0] + 5 + xy_max_e, state[1] + 5 + xy_max_e, +uv_max, +uv_max]))
            # st_curr = self.ocp_solver.get(0,'x')
            # x_curr = st_curr[0]
            # y_curr = st_curr[1]
            #print('Dist to obs: ', np.sqrt((x_curr-env.obstacles[0].position[0])**2 + (y_curr-env.obstacles[0].position[1])**2))


