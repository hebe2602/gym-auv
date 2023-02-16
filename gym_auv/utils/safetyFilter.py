from acados_template import AcadosOcp, AcadosOcpSolver
from acadostesting_v02_export.models.ship_model import export_ship_model
import numpy as np
import time



class TestFilter:
   """
   A filter that adds or subtracts the speed input in play mode.
   """

   def __init__(self, add_input):
        self.add_input = add_input

   def filter(self, input):
      input[0] += self.add_input
      return input


class SafteyFilter:
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
            model = export_ship_model()
            ocp.model = model

            T_s = 1.0
            #nx = model.x.size()[0]
            #nu = model.u.size()[0]
            N = 50
            T_f = N*T_s

            # set dimensions
            ocp.dims.N = N

            # set cost
            R_mat = 2*np.diag([1,1])
            # the 'EXTERNAL' cost type can be used to define general cost terms
            # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
            # ocp.cost.cost_type = 'EXTERNAL'
            ocp.cost.cost_type_e = 'EXTERNAL'
            ocp.cost.cost_type_0 = 'EXTERNAL'
            u0 = np.array([0,0]).reshape(2,1)
            ocp.parameter_values = u0

            # ocp.model.cost_expr_ext_cost = 0
            # ocp.model.cost_expr_ext_cost_e = 0
            ocp.model.cost_expr_ext_cost_0 = (model.u.T - model.p.T) @ R_mat @ (model.u - model.p)
            ocp.model.cost_expr_ext_cost_e = 0
            #ocp.cost.Zl_e = 100*np.ones((nx-1,))
            #ocp.cost.Zu_e = 100*np.ones((nx-1,))
            #ocp.cost.zl_e = 0*np.ones((nx-1,))
            #ocp.cost.zu_e = 0*np.ones((nx-1,))

            # set constraints
            F_u_max = 5*1.0
            F_r_max = 5*0.5
            xy_max = 2000.0
            uvr_max = 2.0

            ocp.constraints.lbx = np.array([-xy_max,-xy_max,-uvr_max,-uvr_max,-uvr_max])
            ocp.constraints.ubx = np.array([+xy_max,+xy_max,+uvr_max,+uvr_max,+uvr_max])
            ocp.constraints.idxbx = np.array([0,1,3,4,5])
            ocp.constraints.lbx_e = 0.25*np.array([-xy_max,-xy_max,-uvr_max,-uvr_max,-uvr_max])
            ocp.constraints.ubx_e = 0.25*np.array([+xy_max,+xy_max,+uvr_max,+uvr_max,+uvr_max])
            ocp.constraints.idxbx_e = np.array([0,1,3,4,5])
            #ocp.constraints.idxsbx_e = np.array([0,1,3,4,5])


            ocp.constraints.lbu = np.array([-F_u_max,-F_r_max])
            ocp.constraints.ubu = np.array([+F_u_max,+F_r_max])
            ocp.constraints.idxbu = np.array([0,1])

            ocp.constraints.x0 = np.array(env.vessel._state)
            #ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


            # obstacle constraint
            state = ocp.model.x
            obst = env.obstacles[0]
            pos = obst.position
            r = obst.radius
      
            ocp.model.con_h_expr = (state[0] - pos[0])**2 + (state[1] - pos[1])**2 - 2*r**2
            ocp.constraints.lh = np.array([0.0])
            ocp.constraints.uh = np.array([100000.0])

            ocp.constraints.lh_e = ocp.constraints.lh
            ocp.constraints.uh_e = ocp.constraints.uh

            ocp.model.con_h_expr_e = ocp.model.con_h_expr



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

            # set prediction horizon
            ocp.solver_options.tf = T_f

            self.ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

      
      def filter(self, u):
            """
            Solve the filter for the current input. 

            Returns the calculated optimal input u0. 
            """      

            self.ocp_solver.set(0,"p",u)
            status = self.ocp_solver.solve()
            self.ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

            if status != 0:
                  raise Exception(f'acados returned status {status}.')
            
            return self.ocp_solver.get(0, "u")

      def update(self, state):
            """
            Update the current state. 
            """
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)


