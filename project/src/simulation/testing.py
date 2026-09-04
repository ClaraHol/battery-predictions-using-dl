import numpy as np
import pybamm


#### Construct a pybamm model

def base_simulator(t_sim: int, parameter_values=None):
    options = {'SEI': 'constant',
            'SEI film resistance': 'distributed'}
    model = pybamm.lithium_ion.DFN(options= options)

    solver = pybamm.IDAKLUSolver(rtol=1e-3, atol=1e-3, on_failure="warn")

    # This is the mesh size, higher values means greater resolution in different parts of the model.
    var_pts = {
   "x_n": 20,  # negative electrode thickness direction mesh size
   "x_s": 10,  # separator thickness direction mesh size
   "x_p": 20,  # positive electrode thickness direction mesh size
   "r_n": 20,  # negative particle radius direction mesh size
   "r_p": 20,  # positive particle radius direction mesh size
   }
    print("reached this point")
    t_eval = np.linspace(0, t_sim*3, num=100)
    sim = pybamm.Simulation(model, parameter_values=parameter_values,solver=solver, var_pts=var_pts)

    sim.solve(t_eval=t_eval)
    return sim


sim = base_simulator(30, parameter_values= pybamm.ParameterValues("Chen2020"))

print(type(sim.solution))
print(vars(sim.solution))

sim.plot()