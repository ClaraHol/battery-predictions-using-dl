import pybamm
import numpy as np
# ==============================================================================
# CODE ATTRIBUTION
# ------------------------------------------------------------------------------
# Associated Paper: Discovery Learning predicts battery cycle life from minimal experiments
# Source Capsule:    Code Ocean (Capsule ID: 9957480, Version: v1)
# Capsule URL:       https://codeocean.com/capsule/9957480/tree/v1
# Primary Authors:   Attribution to original authors/creators of Capsule 9957480
# Platform/DOI:      Code Ocean Reproducible Research Capsule
# License:           Refer to capsule source license (typically MIT or CC-BY-4.0)
# Date Accessed:     September 9, 2026
#
# ==============================================================================
##### pre-given general parameters
# Specific parameters
capa_used = 1.011024 # Discharge or charge capacity at the target cycle number (i.e. cycle 1) [Ah], take the minimal value of all similar cells.
# Nominal capacity
capa_nom = 1.1 # rated capacity [Ah], useless but necessary
# Cut-off voltages (refered to the manufacture specification) 
V_cut_lb = 2
V_cut_ub = 3.6
##### Design parameters
# Total corss-sectional area of the electrode [m^2]
area = 794e-4
# Thichness [m]
L_n = 36e-6 
L_p = 81e-6 
L_s = 18e-6 
# Porosity [-]
epse_n = 0.25
epse_p = 0.26
epse_s = 0.304 # https://doi.org/10.1016/j.jpowsour.2023.233009
# Particle radius [m]
Rp_n = 3e-6
Rp_p = 0.5e-6
######################################################################
##### pre-calculated parameters
# Maximum solid-phase lithium concentration
Cs_max_n = 30675
Cs_max_p = 22835
F = 96485 #(Faraday constant)

##### Obtained from literature
c_e_typ = 1200
sigma_s_n = 100
sigma_s_p = 10
t_0_plus = 0.38
b = 2.2


# If the operation is discharging, we should consider theta_n1 
### SNL-A is charging!!!
######################################################################
# If the operation is discharging, we should consider theta_n1 
theta_n_min = 0
theta_n_max = 0.944
theta_p_min = 0.022
theta_p_max = 0.991

theta_n_lb = theta_n_min 
theta_n_ub = theta_n_min + 0.4
theta_p_lb = theta_p_max - 0.4
theta_p_ub = theta_p_max 

epss_n_lb = 0.4
epss_n_ub = 0.9
epss_p_lb = 0.4
epss_p_ub = 0.9

# The parameter bounds with log10
params_log={
    'D_s_n [m2/s]': [np.log10(1e-16),np.log10(1e-10)],
    'D_s_p [m2/s]': [np.log10(1e-16),np.log10(1e-10)],
    'k_n [(A/m2)(mol/m3)^(-1.5)]': [np.log10(1e-8), np.log10(1e-2)],
    'k_p [(A/m2)(mol/m3)^(-1.5)]': [np.log10(1e-8), np.log10(1e-2)],
    'D_e': [0.3, 6],
    'sigma_e [S/m]': [0.3, 2.8],
    'R_f [ohm*m^2]': [np.log10(1e-5), np.log10(1e-1)], # lumped film resistance
    'eps_s_n':[epss_n_lb, epss_n_ub],
    'eps_s_p':[epss_p_lb, epss_p_ub],
    'theta_n_init':[theta_n_lb, theta_n_ub],
    'theta_p_init':[theta_p_lb, theta_p_ub]
}
######################################################################
# The flag for the log10
params_log_flag={
    'D_s_n [m2/s]': 1,
    'D_s_p [m2/s]': 1,
    'k_n [(A/m2)(mol/m3)^(-1.5)]': 1,
    'k_p [(A/m2)(mol/m3)^(-1.5)]': 1,
    'D_e': 0,
    'sigma_e': 0,
    'R_f [ohm*m^2]': 1, # lumped film resistance
    'eps_s_n':0,
    'eps_s_p':0,
    'theta_n_init':0,
    'theta_p_init':0
}
######################################################################

# anode OCP function
def anode_ocp(sto):  #p32-523-anode, 0-0.93984
    a1 = -1.39730049e-01
    a2 = 1.00040810e+00
    a3 = -1.39544960e+01
    b1 = 8.28605657e-01
    b2 = -1.81616246e+02
    c1 = -4.40620809e-11  
    c2 = 5.83507053e+01 
    c3 = -3.35105711e+01
    d1 = -2.02278879e-02  
    d2 = 1.12969614e+01 
    d3 = -6.24595086e+00  
    e1 = 1.80798094e-01
    e2 = 2.45153920e+01 
    e3 = -1.88151849e+00

    u_eq = (
             a1 + a2* pybamm.exp(a3*sto) + b1*pybamm.exp(b2*sto) + c1*pybamm.exp(c2*sto+c3) 
           +d1*pybamm.arctan(d2*sto+d3)
            +e1*pybamm.arctan(e2*sto+e3) 
    )
    return u_eq
# cathode OCP function
def cathode_ocp(sto): # LFP_charge 0.0220~0.9753 
    a1 = 7.19612504e-13
    a2 = 6.79528383e+00
    a3 = 3.74081730e+00
    a4 = -9.57171127e-06
    a5 = 1.41157457e+01
    a6 = 4.62787551e-01
    b1 = -2.35141036e+00
    b2 = 1.76079252e+01
    b3 = 4.96690501e-02
    c1 = 6.45307340e+00
    c2 = 3.86259985e+00
    c3 = -1.04172443e+00
    d1 = 3.25618645e+00
    d2 = 2.31778129e+00
    d3 = -4.61265810e-01
    e1 = 2.57607900e+01
    e2 = -1.49171110e+00
    e3 = -1.64577417e+00
    f1 = -1.56849244e+01
    f2 = -8.97941008e+00
    f3 = -1.04457073e+00
    g1 = 1.53788436e+00
    g2 = -2.83351899e+00
    g3 = -4.06195603e-01
    u_eq = (a1 * pybamm.exp(a2*sto) + a3+ a4 * pybamm.exp(a5*sto)+ a6 * (sto-0.55)**2
+ b1 * pybamm.tanh(b2 * (sto + b3))
+ c1 * pybamm.tanh(c2 * (sto + c3))
+ d1 * pybamm.tanh(d2 * (sto + d3))
+ e1 * pybamm.tanh(e2 * (sto + e3))
+ f1 * pybamm.tanh(f2 * (sto + f3))
+ g1 * pybamm.tanh(g2 * (sto + g3)))
    return u_eq

def params_setting(params):
    theta_n_init = params[9]
    theta_p_init = params[10]
    
    def anode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
        m_ref = params[2]
        return m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    def cathode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
        m_ref = params[3]
        return m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    def anode_diffusivity(sto, T):
        m_ref = params[0]
        coeff = (1.5 - sto) ** 1.5
        return m_ref * coeff
    def cathode_diffusivity(sto,T):
        m_ref = params[1]
        coeff = (1.5 - sto) ** 1.5
        return m_ref * coeff
    def electrolyte_conductivity(c_e, T):
        sigma_e = params[5] * (0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000))
        return sigma_e
    def electrolyte_diffusivity(c_e, T):
        D_c_e = params[4]*(8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10)
        return D_c_e
    
    params_dict={
    ##### General specification
    'Nominal cell capacity [A.h]':capa_nom, # useless but necessary
    'Number of cells connected in series to make a battery': 1,
    ##### Design parameters
    'Electrode height [m]': area,
    'Electrode width [m]': 1,
    'Number of electrodes connected in parallel to make a cell': 1,
    'Negative electrode thickness [m]': L_n,
    'Positive electrode thickness [m]': L_p,
    'Separator thickness [m]': L_s,
    'Negative electrode porosity': epse_n,
    'Positive electrode porosity': epse_p,
    'Separator porosity': epse_s,
    'Negative particle radius [m]': Rp_n,
    'Positive particle radius [m]': Rp_p,
    'Negative electrode active material volume fraction': params[7], 
    'Positive electrode active material volume fraction': params[8],
    'Negative electrode Bruggeman coefficient (electrolyte)': b, # assumed
    'Positive electrode Bruggeman coefficient (electrolyte)': b, # assumed
    'Separator Bruggeman coefficient (electrolyte)': b, # assumed
    'Negative electrode Bruggeman coefficient (electrode)': b, # assumed
    'Positive electrode Bruggeman coefficient (electrode)': b, # assumed
    ##### Material properties - electrode
    'Negative electrode exchange-current density [A.m-2]': anode_exchange_current_density,
    'Positive electrode exchange-current density [A.m-2]': cathode_exchange_current_density,
    'Negative electrode diffusivity [m2.s-1]': anode_diffusivity,
    'Positive electrode diffusivity [m2.s-1]': cathode_diffusivity,
    'Negative electrode OCP [V]': anode_ocp,
    'Positive electrode OCP [V]': cathode_ocp,
    'Maximum concentration in negative electrode [mol.m-3]': Cs_max_n,
    'Maximum concentration in positive electrode [mol.m-3]': Cs_max_p,

    ######################################################################
    'Negative electrode conductivity [S.m-1]': sigma_s_n, 
    'Positive electrode conductivity [S.m-1]': sigma_s_p, 
    'Negative electrode charge transfer coefficient': 0.5, # assumed
    'Positive electrode charge transfer coefficient': 0.5, # assumed
    'Negative electrode OCP entropic change [V.K-1]': 0, # assumed, useless but necessary
    'Positive electrode OCP entropic change [V.K-1]': 0,# assumed, useless but necessary
    ##### Material properties - electrolyte
    'Electrolyte conductivity [S.m-1]': electrolyte_conductivity,
    'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity,
    'Cation transference number': t_0_plus, # WangCY-Nature
    'Thermodynamic factor': 2, # WangCY-Nature
    ######################################################################
    'SEI resistivity [Ohm.m]': params[6]/(5e-9),
    'Initial inner SEI thickness [m]': 2.5e-9,#2.5e-09
    'Initial outer SEI thickness [m]': 2.5e-9,#2.5e-09
    'Inner SEI partial molar volume [m3.mol-1]':0.1, ### useless but necessary
    'Outer SEI partial molar volume [m3.mol-1]':0.1, ### useless but necessary
    'Ratio of lithium moles to SEI moles': 0.1, ### useless but necessary
    ######################################################################
    ##### Initial parameters 
    'Initial concentration in electrolyte [mol.m-3]': c_e_typ, 
    'Typical electrolyte concentration [mol.m-3]': c_e_typ, 
    'Initial concentration in negative electrode [mol.m-3]': Cs_max_n*theta_n_init,
    'Initial concentration in positive electrode [mol.m-3]': Cs_max_p*theta_p_init,
    ##### Operating conditions 
    'Current function [A]': capa_nom, # This will be overwirte in 'battery_simulator'
    'Typical current [A]': capa_nom, # Useless but necessary
    'Lower voltage cut-off [V]': V_cut_lb,
    'Upper voltage cut-off [V]': V_cut_ub,
    'Ambient temperature [K]': 298.15, # Useless but necessary
    'Initial temperature [K]': 298.15, # Useless but necessary
    'Reference temperature [K]': 298.15, # Useless but necessary
    ##### Currrent collector
    'Negative current collector conductivity [S.m-1]': 58411000.0, # useless but necessary
    'Positive current collector conductivity [S.m-1]': 36914000.0, # useless but necessary
    ##### Other parameters
    'Negative electrode electrons in reaction': 1, # useless but necessary
    'Positive electrode electrons in reaction': 1, # useless but necessary
    }
    return params_dict