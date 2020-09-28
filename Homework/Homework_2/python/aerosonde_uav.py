import numpy as np

## Aircraft Parameters (Aerosonde UAV)
m = 13.5 #kg
J = np.matrix([[0.8244,0,-0.1204],[0,1.135,0],[-0.1204,0,1.759]]) #kg*m^2
S = 0.55 #m^2
b = 2.8956 #m
c = 0.18994 #m
S_prop = 0.2027 #m^2
rho = 1.2682 #kg/m^3
k = 80.0
k_Tp = 0.0
k_Omega = 0.0
e = 0.9

## Longitudinal Aero Coefficients
C_L0 = 0.28
C_D0 = 0.03
C_m0 = -0.02338
C_L_alpha = 3.45
C_D_alpha = 0.30
C_m_alpha = -0.38
C_L_q = 0.0
C_D_q = 0.0
C_m_q = -3.6
C_L_delev = -0.36
C_D_delev = 0
C_m_delev = -0.5
C_prop = 1.0
M = 50.0
alpha_0 = 0.4712
epsilon = 0.1592
C_D_p = 0.0437
C_n_drud = -0.032

## Lateral/Directional Aero Coefficients
C_Y0 = 0.0
C_l0 = 0.0
C_n0 = 0.0
C_Y_beta = -0.98
C_l_beta = -0.12
C_n_beta = 0.25
C_Y_p = 0.0
C_l_p = -0.26
C_n_p = 0.022
C_Y_r = 0.0
C_l_r = 0.14
C_n_r = -0.35
C_Y_dail = 0.0
C_l_dail = 0.08
C_n_dail = 0.06
C_Y_drud = -0.17
C_l_drud = 0.105
