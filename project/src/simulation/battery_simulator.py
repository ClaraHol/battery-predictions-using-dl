
import pybamm
import numpy as np
import traceback


def simulator(params, current, t_sim, V_cut_lb, V_cut_ub):
    # This is the base level simulator called by the SBI models during trainning
    # params: The physical state of the battery
    # current: The current used during the cycle in ampere
    # t_sim: The endtime of the simulation, simulation can still be stopped early if we leave the battery operational bounds
    # output: The simulation solution

    options = {'SEI': 'constant',
        'SEI film resistance': 'distributed'}
    model = pybamm.lithium_ion.DFN(options=options)
    params["Current function [A]"] = current
    parameter_values = pybamm.ParameterValues(params)


    solver = pybamm.IDAKLUSolver(rtol= 1e-3, atol= 1e-3)
    var_pts = {
   "x_n": 20,  # negative electrode thickness direction mesh size
   "x_s": 10,  # separator thickness direction mesh size
   "x_p": 20,  # positive electrode thickness direction mesh size
   "r_n": 20,  # negative particle radius direction mesh size
   "r_p": 20,  # positive particle radius direction mesh size
   }

    sim = pybamm.Simulation(model, parameter_values=parameter_values,
                        solver=solver, var_pts= var_pts)
    t_eval = np.linspace(0, t_sim, num=1000) # The paper used t_sim*3

    try:
        sim.solve(t_eval=t_eval)
        # sim.solve()
        return sim.solution
    except Exception as e:
        result = f"Extreme case error!"
        del sim, model, parameter_values, solver
        gc.collect()
        return result