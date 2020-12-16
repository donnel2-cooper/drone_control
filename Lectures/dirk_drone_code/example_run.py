import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from rotations import *
import mavsim_python_parameters_aerosonde_parameters as P
import simulation_parameters as SIM
import autopilot


plt.close('all')

# Sim time
sim_time = 4

# Trim state from Beard (Ch05)
#  pn, pe, pd, u, v, w,  e0, e1, e2, e3, p, q, r
x_trim = np.array([[-0.000000, -0.000000, -100.000000, 24.971443, 0.000000, 1.194576, 0.993827, 0.000000, 0.110938, 0.000000, 0.000000, 0.000000, 0.000000]]).flatten()
#x_trim = np.array([[-0.000000, -0.000000, -100.000000, 24.971443, 0.000000, 1.194576, 0.9961586592674767, -0.0010209355036630732, 0.025005381108653475, 0.08391272622506223, 0.000000, 0.000000, 0.000000]]).flatten()
#  delta_e, delta_a, delta_r, delta_t
delta_trim = np.array([[-0.118662], [.009775], [-0.001611],[0.857721]])
#delta_trim = np.array([[-1.24779080e-01], [-2.00606541e-03], [1.21662241e-04],[6.76753679e-01]])

# Init drone
drone = Drone()

#Init Autopilot
ctrl = autopilot.autopilot(SIM.ts_simulation)
throttle = float(delta_trim[3])

# Set drone rigid body state to x_trim
drone.state.rigid_body = x_trim
# Set drone input to delta_trim 
delta = delta_trim

# Drone Sim
t_history = [0]
x_history = [drone.state.rigid_body]
delta_history = [delta]

Va_command = 25
h_command = 100
chi_command = np.radians(45)

while drone.state.time <= 500: #SIM.end_time:
    drone.update(delta)
    command = np.array([chi_command,h_command,Va_command])
    delta = ctrl.update(command,drone.state) + np.array([[0],[0],[0],[throttle]])
    throttle = throttle+float(delta[3])
    t_history.append(drone.state.time)
    x_history.append(drone.state.rigid_body)
    delta_history.append(delta)

# Convert to numpy array
x_history = np.asarray(x_history)

plt.figure()
plt.plot(t_history, x_history[:, :3])
plt.legend(['x1', 'x2', 'x3'])
plt.show()


plt.figure()
plt.plot(t_history, x_history[:, 3:6])
plt.legend(['u', 'v', 'w'])
plt.show()

# Convert quat to euler
nsteps = x_history.shape[0]
euler_angles = np.zeros((nsteps, 3))
for i in range(nsteps):         
    e = x_history[i, 6:10]
    euler_angles[i, :] = quat2euler(e) # phi, theta, psi = 

plt.figure()
plt.plot(t_history, (180/np.pi)*euler_angles)
plt.legend(['phi', 'theta', 'psi'])
plt.show()

plt.figure()
plt.plot(t_history, x_history[:, 10:])
plt.legend(['p', 'q', 'r'])
plt.show()