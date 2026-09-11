
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

    # experiment = pybamm.Experiment([
    #     f"Discharge at {abs(current)} A for {t_sim} seconds or until {V_cut_lb} or {V_cut_ub} V",
    # ])

    experiment = pybamm.Experiment([
    pybamm.step.current(
        abs(current),
        duration=t_sim,
        termination=[f"{V_cut_lb} V", f"{V_cut_ub} V"]
    )
    ])

    solver = pybamm.IDAKLUSolver(rtol= 1e-3, atol= 1e-3)
    var_pts = {
   "x_n": 20,  # negative electrode thickness direction mesh size
   "x_s": 10,  # separator thickness direction mesh size
   "x_p": 20,  # positive electrode thickness direction mesh size
   "r_n": 20,  # negative particle radius direction mesh size
   "r_p": 20,  # positive particle radius direction mesh size
   }

    sim = pybamm.Simulation(model, experiment= experiment, parameter_values=parameter_values,
                        solver=solver, var_pts= var_pts)
    t_eval = np.linspace(0, t_sim*3, num=1000)

    try:
    #   sim.solve(t_eval=t_eval)
        sim.solve()
        return sim.solution
    except Exception as e:
        print('Extreme case error!', repr(e))
        traceback.print_exc()
        return 'Extreme case error!'