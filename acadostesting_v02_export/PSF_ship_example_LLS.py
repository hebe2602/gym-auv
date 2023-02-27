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

    T_s = 1.0
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
    u0 = np.array([0,0]).reshape(2,1)
    ocp.parameter_values = u0

    Vx_0 = np.zeros((ny,nx))
    ocp.cost.Vx_0 = Vx_0
    Vu_0 = np.eye(ny)
    ocp.cost.Vu_0 = Vu_0

    ocp.cost.W_0 = np.eye(ny)
    
    yref_0 = u0
    ocp.cost.yref_0 = yref_0

    # ocp.cost.Zl_e = 100*np.ones((nx-1,))
    # ocp.cost.Zu_e = 100*np.ones((nx-1,))
    # ocp.cost.zl_e = 0*np.ones((nx-1,))
    # ocp.cost.zu_e = 0*np.ones((nx-1,))
    # set constraints
    F_u_max = 1.0
    F_r_max = 0.5

    xy_max = 20.0
    uv_max = 2.0

    ocp.constraints.lbx = np.array([-xy_max,-xy_max,-uv_max,-uv_max])
    ocp.constraints.ubx = np.array([+xy_max,+xy_max,+uv_max,+uv_max])
    ocp.constraints.idxbx = np.array([0,1,3,4])
    ocp.constraints.lbx_e = 0.25*np.array([-xy_max,-xy_max,-uv_max,-uv_max])
    ocp.constraints.ubx_e = 0.25*np.array([+xy_max,+xy_max,+uv_max,+uv_max])
    ocp.constraints.idxbx_e = np.array([0,1,3,4])
    # ocp.constraints.idxsbx_e = np.array([0,1,3,4,5])

    # (x - x_obj)**2 + (y - y_obj)**2 >= radius_obj

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
    ocp.solver_options.nlp_solver_step_length = 1.0
    ocp.solver_options.nlp_solver_tol_eq = 1e-2

    # set prediction horizon
    ocp.solver_options.tf = T_f

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
    
    
    simN = 1000
    simX = np.ndarray((simN+1, nx))
    simU = np.ndarray((simN, nu))
    simX_pred = np.ndarray((N+1,nx,simN+1))
    stimes = np.ndarray((simN,))

    u1_rand = -F_u_max + np.random.rand(simN,1)*2*F_u_max
    u2_rand = -F_r_max + np.random.rand(simN,1)*2*F_r_max
    u_rand = np.column_stack([u1_rand,u2_rand])
    u1_alt = np.linspace(0.0,F_u_max,num=simN).reshape(-1,1)
    u2_alt = np.zeros((simN,)).reshape(-1,1)#np.linspace(0.0,F_r_max,num=simN).reshape(-1,1)
    u_alt = np.column_stack([u1_alt,u2_alt])

    for i in range(simN):
        print(i)
        #print(simX[i-1,:])
        ocp_solver.cost_set(0,'yref',u_alt[i,:])
        stime = time.time()
        status = ocp_solver.solve()
        stimes[i] = time.time() - stime
        ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            #raise Exception(f'acados returned status {status}.')
            break

        for j in range(N+1):
            simX_pred[j,:,i] = ocp_solver.get(j,'x')

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
    
    print(i)
    simX = simX[:i,:]
    simX_pred = simX_pred[:,:,:i]
    stimes = stimes[:i]
    np.savetxt('./data/simX.txt',simX)
    simX_pred = simX_pred.reshape(-1,nx)
    np.savetxt('./data/simX_pred.txt',simX_pred)

    print('Max x-value was: ', max(abs(simX[:,0])))
    print('Max y-value was: ', max(abs(simX[:,1])))
    print('Max phi-value was: ', max(abs(simX[:,2])))
    print('Max u-value was: ', max(abs(simX[:,3])))
    print('Max v-value was: ', max(abs(simX[:,4])))
    print('Max r-value was: ', max(abs(simX[:,5])))
    print('Average difference between primary and modified control input was: F_u: ', np.average(abs(simU[:i,0] - u_rand[:i,0])), ', F_r: ', np.average(abs(simU[:i,1] - u_rand[:i,1])))
    print('Average solve time was: ', np.average(stimes))


if __name__ == '__main__':
    main()