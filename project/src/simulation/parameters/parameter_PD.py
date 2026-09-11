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
capa_used = 82.73 # Discharge or charge capacity at the target cycle number (i.e. cycle 1) [Ah], take the minimal value of all similar cells.
# Nominal capacity
capa_nom = 83.456 # rated capacity [Ah], useless but necessary
# Cut-off voltages (refered to the manufacture specification) 
V_cut_lb = 2.5
V_cut_ub = 4.15
##### Design parameters
# Total corss-sectional area of the electrode [m^2]
area = 0
# Thichness [m]
L_n = 0
L_p = 0
L_s = 0
# Porosity [-]
epse_n = 0
epse_p = 0
epse_s = 0
# Particle radius [m]
Rp_n = 0
Rp_p = 0
##### pre-calculated parameters
# Maximum solid-phase lithium concentration
Cs_max_n = 62820 # 
Cs_max_p = 50790 # 
F = 96485 #(Faraday constant)

##### Obtained from literature
c_e_typ = 1100
sigma_s_n = 100
sigma_s_p = 10
t_0_plus = 0.415
b = 2.2

######################################################################
# If the operation is discharging, we should consider theta_n1 
theta_n_min = 0.04
theta_n_max = 0.95
theta_p_min = 0.15
theta_p_max = 0.92

theta_n_lb = theta_n_max - 0.4
theta_n_ub = theta_n_max
theta_p_lb = theta_p_min
theta_p_ub = 0.385

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
    'theta_p_init':[0.14, theta_p_ub]
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

########### NCM 90505 

def cathode_ocp(sto): # p84-cathode-0.15 0.92

    a1 =-1.98781559e+01
    a2 =-9.79695956e-01
    b1 =7.36974099e+00
    b2 =-4.35967754e+00

    b3 =-4.36343389e-01
    c1 =7.59705492e+00
    c2 =4.22061759e+00
    c3 =-4.33022835e-01

    d1 =-6.83825023e-02
    d2 =2.14231668e+01
    d3 =-2.81293277e-01
    e1 =-2.33446722e-03

    e2 = 1.48125800e+02
    e3 =-7.64036416e-01
    f1 =2.42245328e+01
    f2 =-1.90333635e+02
    
    f3 =-9.38545773e-01
    g1 =9.94978343e-03
    g2 =3.37923004e+01
    g3 =-8.72105806e-01

    u_eq = ( a1 + a2*sto
            + b1 *  pybamm.tanh(b2 * (sto + b3))
            + c1 *  pybamm.tanh(c2 * (sto + c3))
            + d1 *  pybamm.tanh(d2 * (sto + d3))
            + e1 *  pybamm.tanh(e2 * (sto + e3))
            + f1 *  pybamm.tanh(f2 * (sto + f3))
            + g1 *  pybamm.tanh(g2 * (sto + g3)) 
           
            )
    return u_eq

##### Si-C 10%
def anode_ocp(sto): # p84-anode-0.04 0.95
    
    a1 =-6.54748051e+02
    a2 =1.13667330e+03
    b1 =1.21061628e+03
    b2 =-3.42342373e+00

    b3 =-3.47992034e-01
    c1 =4.40773065e+02
    c2 =3.77260866e+00
    c3 =-3.62661043e-01

    d1 =1.66522041e+03
    d2 =2.83046289e+00
    d3 =-3.13356639e-01
    e1 =-8.24237567e+02

    e2 = 6.36593514e+01
    e3 =2.84750561e-02
    f1 =1.89192646e+03
    f2 =-5.98384605e-01
    
    f3 =-1.60326824e+00
    g1 =1.25043051e+03
    g2 =-2.20872151e+00
    g3 =-2.53315867e-01

    u_eq = (a1 + a2 * sto+
            + b1 * pybamm.tanh(b2 * (sto + b3))
            + c1 *  pybamm.tanh(c2 * (sto + c3))
            + d1 *  pybamm.tanh(d2 * (sto + d3))
            + e1 *  pybamm.tanh(e2 * (sto + e3))
            + f1 *  pybamm.tanh(f2 * (sto + f3))
            + g1 *  pybamm.tanh(g2 * (sto + g3))
           
            )
    return u_eq

###########

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