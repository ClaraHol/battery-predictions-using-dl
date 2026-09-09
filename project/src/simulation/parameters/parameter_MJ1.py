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
capa_used = 3.241967541 # Discharge or charge capacity at the target cycle number (i.e. cycle 1) [Ah], take the minimal value of all similar cells.
# Nominal capacity
capa_nom = 3.5 # rated capacity [Ah], useless but necessary
# Cut-off voltages (refered to the manufacture specification) 
V_cut_lb = 2.5
V_cut_ub = 4.2 
##### Design parameters
# Total corss-sectional area of the electrode [m^2]
area = 760e-4
# Thichness [m]
L_n = 85e-6 
L_p = 72e-6 
L_s = 15e-6 # Average taken from 2 papers
# Porosity [-]
epse_n = 0.27
epse_p = 0.36
epse_s = 0.393 # This is (0.38+0.38+0.42)/3
# Particle radius [m]
Rp_n = 7.5e-6   # half particle size
Rp_p = 3e-6  # half particle size
######################################################################
##### pre-calculated parameters
# Maximum solid-phase lithium concentration
Cs_max_n = 42102
Cs_max_p = 50975
F = 96485 #(Faraday constant)

##### Obtained from literature
c_e_typ = 1200
sigma_s_n = 100
sigma_s_p = 10
t_0_plus = 0.38 #'Wang chao yang Nature'
b = 2.2 # related to tortuosity, coming from 'Wang chao yang Nature, and fast-charging Nature Energy'
######################################################################
# If the operation is discharging, we should consider theta_n1 
theta_n_min = 0.002 
theta_n_max = 0.873
theta_p_min = 0.193
theta_p_max = 0.955

theta_n_lb = theta_n_max - 0.4
theta_n_ub = theta_n_max
theta_p_lb = theta_p_min
theta_p_ub = 0.3755

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
    'theta_p_init':[0, theta_p_ub]  
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

# anode OCP function   Si-C Using TUM Strum, Power Source
def anode_ocp(sto):  # SiC  anode, 0.002286 0.8665
    a1 = -2.09901437e+00
    a2=6.35384968e+00
    a3=-2.38613565e+02
    a4=2.52002108e+00

    a5=6.37300924e+00
    a6=2.94020752e-01
    a7=-6.26748326e-06
    a8=1.47679135e+01

    b1=1.14019969e+00
    b2=-2.57294592e+01
    b3=2.39521886e-02
    c1=1.95048580e+03

    c2=-3.24696837e+00
    c3=-1.40908634e+00
    d1=3.03950221e-02
    d2=-2.57729087e+01

    d3=-1.78695078e-01
    e1=2.54498737e+02
    e2=-4.45357904e+00
    e3=2.11082141e+00

    f1=4.73190072e+02
    f2=-1.16490726e+01
    f3=1.12036418e+00
    g1=3.53277403e+02

    g2=-3.68393711e+00
    g3=3.44782030e+00
    h1=6.29679262e+02
    h2=-1.82156225e+02

    h3=2.17594241e-02
    h4=-1.43068638e-02
    h5=6.71986256e+01
    h6=-4.69781321e-01
    
    u_eq = (a1 * pybamm.exp(a2*sto) + a3+ a4 * pybamm.exp(a5*sto)+ a6 * (sto-0.55)**2 + a7 * pybamm.exp(a8*sto)
            + b1 * pybamm.tanh(b2 * (sto + b3))
            + c1 * pybamm.tanh(c2 * (sto + c3))
            + d1 * pybamm.tanh(d2 * (sto + d3))
            + e1 * pybamm.tanh(e2 * (sto + e3))
            + f1 * pybamm.tanh(f2 * (sto + f3))
            + g1 * pybamm.tanh(g2 * (sto + g3))
            + h1 * pybamm.tanh(h2 * (sto + h3))
            + h4 * pybamm.arctan(h5 * (sto + h6))
       
            )
    return u_eq
# cathode OCP function
def cathode_ocp(sto):#p76-cathode 0.199 0.955
    a1 = -2.18493356e+02
    a2 = 1.19712762e+02 
    b1 = 1.18752552e+02
    b2  =-1.80371401e+00
    b3  =-4.87054294e-01
    c1   =6.75715460e-02
    c2  =1.29537935e+01
    c3  =-4.07723172e-01
    d1  =1.26951322e-01
    d2  =1.66532922e+01
    d3  =-2.38124428e-01
    e1  =-3.73311025e+00
    e2  =4.07914776e+00
    e3  =-7.99705211e-01
    f1  =3.07216575e-01
    f2 =-7.61133341e+01
    f3  =-9.49909874e-01
    g1  =3.72655720e+02
    g2  =1.16916449e+00
    g3 =-5.17062307e-01

    u_eq = (
    a1*sto+a2
    +b1* pybamm.tanh(b2*(sto+b3))
    +c1* pybamm.tanh(c2*(sto+c3))
    +d1* pybamm.tanh(d2*(sto+d3))
    +e1* pybamm.tanh(e2*(sto+e3))
    +f1* pybamm.tanh(f2*(sto+f3))
    +g1* pybamm.tanh(g2*(sto+g3))
    )
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