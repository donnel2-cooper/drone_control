import sys
sys.path.append('..')
import numpy as np
import parameters.aerosonde_parameters as MAV
import tools.signals as sigs
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Quaternion
import mav_dynamics
import mavsim_python_chap5_model_coef as chap5
import matplotlib.pyplot as plt

# Define sim parameters
dt = 0.01
uav = mav_dynamics.mavDynamics(dt)

T0 = 0
Tf = 5

n = int(np.floor((Tf-T0)/dt))

alpha = [uav._alpha*180/np.pi]
beta = [uav._beta*180/np.pi]
phi = [MAV.phi0*180/np.pi]
theta = [MAV.theta0*180/np.pi]
psi = [MAV.psi0*180/np.pi]
gamma = [uav.true_state.gamma*180/np.pi]
t_history = [T0]

rud_doublet = sigs.signals(amplitude=0.2,frequency=1.0,start_time=1,duration=1,dc_offset = chap5.u_trim[2])
for ii in range(n):
    delta_r = rud_doublet.doublet(dt*ii)
    delta = np.array([chap5.u_trim[0],chap5.u_trim[1],delta_r,chap5.u_trim[3]])
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

plt.figure()
plt.plot(t_history,phi,'r',label = 'phi') 
plt.plot(t_history,theta,'b',label = 'theta') 
plt.plot(t_history,psi,'g',label = 'psi') 
plt.xlabel('t')
plt.ylabel('angle [deg]')
plt.legend()
plt.show()


