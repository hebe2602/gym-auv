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

            # create ocp object to formulate the OCP
            ocp = AcadosOcp()

            # set model
            model = export_ship_model(model_type='simplified')
            ocp.model = model

            self._T_s = 0.5
            nx = model.x.size()[0]
            nu = model.u.size()[0]
            ny = nu
            self._N = 50
            T_f = self._N*self._T_s

            # set dimensions
            ocp.dims.N = self._N


            # set cost
            # the 'EXTERNAL' cost type can be used to define general cost terms
            # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
            ocp.cost.cost_type_0 = 'LINEAR_LS'
            u0 = np.array([0,0]) #.reshape(2,1)

            Vx_0 = np.zeros((ny,nx))
            ocp.cost.Vx_0 = Vx_0
            Vu_0 = np.eye(ny)
            ocp.cost.Vu_0 = Vu_0

            F_u_max = 1.0
            F_r_max = 0.15

            W_0 = 1e-1*np.eye(ny)
            W_0[-1,-1] = F_u_max/F_r_max
            ocp.cost.W_0 = W_0
            
            yref_0 = u0
            ocp.cost.yref_0 = yref_0
            
            # ocp.cost.Zl = 100*np.ones((1,))
            # ocp.cost.Zu = 0*np.ones((1,))
            # ocp.cost.zl = 0*np.ones((1,))
            # ocp.cost.zu = 0*np.ones((1,))
            # # set constraints
            x_max = 20.0
            y_max = 20.0
            u_max = 2.0
            u_min = -2.0
            v_max = 2.0
            v_min = -2.0

            ocp.constraints.lbx = np.array([-x_max,-y_max,u_min,v_min])
            ocp.constraints.ubx = np.array([+x_max,+y_max,+u_max,+v_max])
            ocp.constraints.idxbx = np.array([0,1,3,4])
            ocp.constraints.lbx_e = 0.5*np.array([-x_max,-y_max,u_min,v_min])
            ocp.constraints.ubx_e = 0.5*np.array([+x_max,+y_max,+u_max,+v_max])
            ocp.constraints.idxbx_e = np.array([0,1,3,4])
            #ocp.constraints.idxsbx_e = np.array([0,1,3,4])


            ocp.constraints.lbu = np.array([-F_u_max,-F_r_max])
            ocp.constraints.ubu = np.array([+F_u_max,+F_r_max])
            ocp.constraints.idxbu = np.array([0,1])

            ocp.constraints.x0 = np.array(env.vessel._state)
            #ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


            # obstacle constraint
            obst = env.obstacles[0]
            r_obs = obst.radius + 5
            ocp.parameter_values = np.array([obst.position[0],obst.position[1],obst.radius])
      
            # ocp.model.con_h_expr = log((state[0] - pos[0])**2 + (state[1] - pos[1])**2)
            # ocp.constraints.lh = np.array([-r_obs])
            # ocp.constraints.uh = np.array([9999])

            # ocp.constraints.lh_e = ocp.constraints.lh
            # ocp.constraints.uh_e = ocp.constraints.uh

            #ocp.constraints.idxsh = np.array([0])

            # set options
            ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
            # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
            # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
            ocp.solver_options.integrator_type = 'IRK'
            # ocp.solver_options.print_level = 1
            ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 3
            ocp.solver_options.sim_method_newton_iter = 3
            ocp.solver_options.nlp_solver_tol_eq = 1e-2
            ocp.solver_options.nlp_solver_tol_stat = 1e-2
            ocp.solver_options.nlp_solver_step_length = 1.0
            ocp.solver_options.nlp_solver_ext_qp_res

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


