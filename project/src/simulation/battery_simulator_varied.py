import gc

import numpy as np
import pybamm


def simulator_v2(params, current, t_sim, soc_start):
    # This is the base level simulator called by the SBI models during trainning
    # params: The physical state of the battery
    # current: The current used during the cycle in ampere
    # t_sim: The endtime of the simulation, simulation can still be stopped early if we leave the battery operational bounds
    # output: The simulation solution

    options = {"SEI": "constant", "SEI film resistance": "distributed"}
    model = pybamm.lithium_ion.DFN(options=options)
    params["Current function [A]"] = current
    params["Open-circuit voltage at 0% SOC [V]"] = params[
    "Lower voltage cut-off [V]"
    ]

    params["Open-circuit voltage at 100% SOC [V]"] = params[
        "Upper voltage cut-off [V]"
    ]

    parameter_values = pybamm.ParameterValues(params)
    

    solver = pybamm.IDAKLUSolver(rtol=1e-3, atol=1e-3, options={"dt_min": 1e-6})
    var_pts = {
        "x_n": 20,  # negative electrode thickness direction mesh size
        "x_s": 10,  # separator thickness direction mesh size
        "x_p": 20,  # positive electrode thickness direction mesh size
        "r_n": 20,  # negative particle radius direction mesh size
        "r_p": 20,  # positive particle radius direction mesh size
    }

    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=solver, var_pts=var_pts
    )
    t_eval = np.linspace(0, t_sim, num=1000)  # The paper used t_sim*3

    try:
        sim.solve(t_eval=t_eval, initial_soc = soc_start)
        # sim.solve()
        
        return sim.solution
    except Exception as e:

        result = "Extreme case error!"
        del sim, model, parameter_values, solver
        gc.collect()
        return result
