from acados_template import AcadosOcp, AcadosOcpSolver
from models.ship_model import export_ship_model
import numpy as np
import time

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_ship_model()
    ocp.model = model

    T_s = 1
    nx = model.x.size()[0]
    nu = model.u.size()[0]
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
    F_u_max = 1.0
    F_r_max = 0.5

    xy_max = 20.0
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

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
    
    
    simN = 1000
    simX = np.ndarray((simN+1, nx))
    simU = np.ndarray((simN, nu))

    u1_rand = -F_u_max + np.random.rand(simN,1)*2*F_u_max
    u2_rand = -F_r_max + np.random.rand(simN,1)*2*F_r_max
    u_rand = np.column_stack([u1_rand,u2_rand])
    u1_alt = np.linspace(0.0,F_u_max,num=simN).reshape(-1,1)
    u2_alt = np.linspace(0.0,F_r_max,num=simN).reshape(-1,1)
    u_alt = np.column_stack([u1_alt,u2_alt])

    stime = time.time()
    for i in range(simN):
        print(i)
        print(simX[i-1,:])
        ocp_solver.set(0,"p",u_alt[i,:])
        status = ocp_solver.solve()
        ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            raise Exception(f'acados returned status {status}.')
        
        x0 = ocp_solver.get(0, "x")
        u0 = ocp_solver.get(0, "u")
        for j in range(nx):
            simX[i, j] = x0[j]
        for j in range(nu):
            simU[i, j] = u0[j]
        

        #Update initial condition:
        x0 = ocp_solver.get(1, "x")
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)
    
    ttime = time.time() - stime

    simX = simX[:i,:]
    simU = simU[:i,:]
    print(i)
    print('Max x-value was:', max(abs(simX[:,0])))
    print('Max y-value was:', max(abs(simX[:,1])))

    print(ttime)


if __name__ == '__main__':
    main()