import numpy as np
import matplotlib.pyplot as plt
import integrators as intg

## Aircraft Parameters (Aerosonde UAV)
m = 13.5 #kg
J = np.matrix([[0.8244,0,-0.1204],[0,1.135,0],[-0.1204,0,1.759]]) #kg*m^2
J = np.matrix([[0.8244,0,0],[0,1.135,0],[0,0,1.759]]) #kg*m^2
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

#  xdot = f(t, x, u), 
#   t: time
#                              0   1   2  3  4  5    6      7    8  9 10 11
#   x: state vector in book, [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
#       u, v, w ground speeds 
#               0  1  2  3  4  5
#   u: input, [fx,fy,fz, l, m, n]

## Helper Functions
def sind(angle):
    #Compute sin of angle in degrees as composition of np functions
    return np.sin(np.deg2rad(angle))
def cosd(angle):
    #Compute cos of angle in degrees as composition of np functions
    return np.cos(np.deg2rad(angle))
def tand(angle):
    #Compute tan of angle in degrees as composition of np functions
    return np.tan(np.deg2rad(angle))

## Define EOM's as diff eq's
def f_pos(t, x, u):
    # Equation 3.14
    R = np.matrix([[cosd(x[7])*cosd(x[8]),\
                    sind(x[6])*sind(x[7])*cosd(x[8])-cosd(x[6])*sind(x[8]),\
                    cosd(x[6])*sind(x[7])*cosd(x[8])+sind(x[6])*sind(x[8])],\
                   [cosd(x[7])*sind(x[8]),\
                    sind(x[6])*sind(x[7])*sind(x[8])+cosd(x[6])*cosd(x[8]),\
                    cosd(x[6])*sind(x[7])*sind(x[8])-sind(x[6])*cosd(x[8])],\
                   [-sind(x[7]),\
                    sind(x[6])*cosd(x[7]),\
                    cosd(x[6])*cosd(x[7])]])
    substate = np.array([[x[3]],[x[4]],[x[5]]])
    mat_mult_result = np.matmul(R,substate)
    return np.array([float(mat_mult_result[0]),float(mat_mult_result[1]),\
                     float(mat_mult_result[2]),0,0,0,0,0,0,0,0,0])

def f_vel(t, x, u):
    # Equation 3.15
    f_over_m = np.array([[u[0]/m],[u[1]/m],[u[2]/m]])
    first_term = np.array([[x[11]*x[4]-x[10]*x[5]],\
                           [x[9]*x[5]-x[11]*x[3]],\
                           [x[10]*x[3]-x[9]*x[4]]])
    summation = np.add(first_term,f_over_m)
    return np.array([0,0,0,float(summation[0]),float(summation[1]),\
                     float(summation[2]),0,0,0,0,0,0])

def f_euler(t,x,u):
    # Equation 3.16
    R = np.matrix([[1,\
                    sind(x[6])*tand(x[7]),\
                    cosd(x[6])*tand(x[7])],\
                   [0,\
                    cosd(x[6]),\
                    -sind(x[6])],\
                   [0,\
                    sind(x[6])/cosd(x[7]),\
                    cosd(x[6])/cosd(x[7])]])
    substate = np.array([[x[9]],[x[10]],[x[11]]])
    mat_mul_result = np.matmul(R,substate)
    return np.array([0,0,0,0,0,0,float(mat_mul_result[0]),\
                     float(mat_mul_result[1]),float(mat_mul_result[2]),0,0,0])

def f_rate(t,x,u):
    # Equations 3.13
    gamma = J[0,0]*J[2,2]-(-J[0,2])**2
    gamma_1 = ((-J[0,2])*(J[0,0]-J[1,1]+J[2,2]))/gamma
    gamma_2 = (J[2,2]*(J[2,2]-J[1,1])+(-J[0,2])**2)/gamma
    gamma_3 = J[2,2]/gamma
    gamma_4 = -J[0,2]/gamma
    gamma_5 = (J[2,2]-J[0,0])/J[1,1]
    gamma_6 = -J[0,2]/J[1,1]
    gamma_7 = ((J[0,0]-J[1,1])*J[0,0]+(-J[0,2])**2)/gamma
    gamma_8 = J[0,0]/gamma
    #Equation 3.17
    first_term = np.array([[gamma_1*x[9]*x[10]-gamma_2*x[10]*x[11]],\
                           [gamma_5*x[9]*x[11]-gamma_6*(x[9]**2-x[11]**2)],\
                           [gamma_7*x[9]*x[10]-gamma_1*x[10]*x[11]]])
    second_term = np.array([[gamma_3*u[3]+gamma_4*u[5]],\
                            [u[4]/J[1,1]],\
                            [gamma_4*u[3]+gamma_8*u[5]]])
    summation = np.add(first_term,second_term)
    return np.array([0,0,0,0,0,0,0,0,0,float(summation[0]),\
                     float(summation[1]),float(summation[2])])


## Sim
# Sim Parameters
t0 = 0; tf = 20; dt = 0.01; n = int(np.floor(tf/dt));
#Initial State and Initial Input
x0 = np.array([0.0,0.0,0.0,10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
u0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

# Define State Integrators as RK4 integrators on EOMs
inertial_pos_intg = intg.RungeKutta4(dt,f_pos)
velocity_intg = intg.RungeKutta4(dt,f_vel)
euler_angles_intg = intg.RungeKutta4(dt,f_euler)
rates_intg = intg.RungeKutta4(dt,f_rate)

x = x0
u = u0
t = t0
t_history = [0]
x_history = [x]

# Sim Loop
for i in range(n):
    # Step sim
    inertial_pos = inertial_pos_intg.step(t,x,u)
    velocity = velocity_intg.step(t,x,u)
    euler_angles = euler_angles_intg.step(t,x,u)
    rates = rates_intg.step(t,x,u)
    
    # Step time
    t = (i+1) * dt
    
    # Reconstruct State Vector
    x = np.concatenate((inertial_pos[:3],velocity[3:6],euler_angles[6:9],rates[9:]))

    # Append State and Time to History
    t_history.append(t)
    x_history.append(x)
    
print(x)
