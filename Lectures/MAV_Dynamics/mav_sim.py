import sys
sys.path.append('..')
import numpy as np
import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Quaternion
import mav_dynamics
import mavsim_python_chap5_model_coef as chap5

# Define sim parameters
dt = 0.01
uav = mav_dynamics.mavDynamics(dt)

T0 = 0
Tf = 0.05

n = int(np.floor((Tf-T0)/dt))

print(uav._state)
print('\n')
for ii in range(n):
    print(ii)
    delta = chap5.u_trim
    wind = np.array([[0.], [0.], [0.], [0.], [0.], [0.]])
    uav.update(delta,wind)

print(uav._state)
print(delta)