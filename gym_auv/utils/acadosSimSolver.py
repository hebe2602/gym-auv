from acados_template import AcadosSim, AcadosSimSolver
from gym_auv.utils.ship_model import export_ship_ODE_model


def export_cybership_II_ode_simulator(Ts, model_type = 'simplified'):
    """Creates ODE simulator for 3-DOF Cybership II model using Acados"""
    
    sim = AcadosSim()

    # Import acados ship dynamics model
    sim.model = export_ship_ODE_model(model_type)

    # set simulation time
    sim.solver_options.T = Ts
    # set options
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 3 # for implicit integrator
 
    # Create integrator object
    acados_integrator = AcadosSimSolver(sim)

    return acados_integrator






