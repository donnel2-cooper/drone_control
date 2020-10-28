import sys
sys.path.append('..')
import numpy as np
import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Quaternion
import mav_dynamics
import mavsim_python_chap5_model_coef as chap5
import matplotlib.pyplot as plt

# Define sim parameters
dt = 0.01
uav = mav_dynamics.mavDynamics(dt)

T0 = 0
Tf = 20

n = int(np.floor((Tf-T0)/dt))

state = uav._state
alpha = [uav._alpha*180/np.pi]
beta = [uav._beta*180/np.pi]
phi = [MAV.phi0*180/np.pi]
theta = [MAV.theta0*180/np.pi]
psi = [MAV.psi0*180/np.pi]
gamma = [uav.true_state.gamma*180/np.pi]
t_history = [T0]

for ii in range(n):
    delta = chap5.u_trim
    wind = np.array([[0.], [0.], [0.], [0.], [0.], [0.]])
    uav.update(delta,wind)
    alpha.append(uav._alpha*180/np.pi)
    beta.append(uav._beta*180/np.pi)
    t_history.append(ii*dt)
    phi_t,theta_t,psi_t = Quaternion2Euler(uav._state[6:10])
    phi.append(phi_t*180/np.pi)
    theta.append(theta_t*180/np.pi)
    psi.append(psi_t*180/np.pi)
    gamma.append(uav.true_state.gamma*180/np.pi)
    state = np.concatenate((state,uav._state),axis = 1)

# close all figures
print(uav._state)
plt.close('all')

plt.figure()
plt.plot(t_history,alpha,'r',label = 'aoa') 
plt.xlabel('t')
plt.ylabel('angle [deg]')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_history,beta,'r',label = 'aos') 
plt.xlabel('t')
plt.ylabel('angle [deg]')
plt.legend()
plt.show()


# Pos vs Time
plt.figure()
plt.plot(t_history,state[0,:],'r',label = 'pn') 
plt.plot(t_history,state[1,:],'b',label = 'pe') 
plt.plot(t_history,state[2,:],'g',label = 'pd') 
plt.xlabel('t [s]')
plt.ylabel('Inertial Positions [m]')
plt.legend()
plt.show()

# Vel vs Time
plt.figure()
plt.plot(t_history,state[3,:],'r',label = 'u') 
plt.plot(t_history,state[4,:],'b',label = 'v') 
plt.plot(t_history,state[5,:],'g',label = 'w') 
plt.xlabel('t [s]')
plt.ylabel('Air Speed [m/s]')
plt.legend()
plt.show()

# Euler Angles vs Time
plt.figure()
plt.plot(t_history,phi,'r',label = 'phi') 
plt.plot(t_history,theta,'b',label = 'theta') 
plt.plot(t_history,psi,'g',label = 'psi') 
plt.xlabel('t [s]')
plt.ylabel('Euler Angle [deg]')
plt.legend()
plt.show()

# Ang Rates vs Time
plt.figure()
plt.plot(t_history,state[10,:],'r',label = 'p') 
plt.plot(t_history,state[11,:],'b',label = 'q') 
plt.plot(t_history,state[12,:],'g',label = 'r') 
plt.xlabel('t [s]')
plt.ylabel('Angular rates [deg/s]')
plt.legend()
plt.show()


# [[MAV.pn0],   # (0)
#  [MAV.pe0],   # (1)
#  [MAV.pd0],   # (2)
#  [MAV.u0],    # (3)
#  [MAV.v0],    # (4)
#  [MAV.w0],    # (5)
#  [MAV.e0],    # (6)
#  [MAV.e1],    # (7)
#  [MAV.e2],    # (8)
#  [MAV.e3],    # (9)
#  [MAV.p0],    # (10)
#  [MAV.q0],    # (11)
#  [MAV.r0]]    # (12)