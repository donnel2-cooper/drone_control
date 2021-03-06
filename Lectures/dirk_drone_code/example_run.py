import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from rotations import *
import mavsim_python_parameters_aerosonde_parameters as P
import simulation_parameters as SIM
import autopilot
import signals as sigs


plt.close('all')

# Sim time
sim_time = 4

# Trim state from Beard (Ch05)
#  pn, pe, pd, u, v, w,  e0, e1, e2, e3, p, q, r
# x_trim = np.array([[-0.000000, -0.000000, -100.000000, 24.971443, 0.000000, 1.194576, 0.993827, 0.000000, 0.110938, 0.000000, 0.000000, 0.000000, 0.000000]]).flatten()
x_trim = np.array([[-0.000000, -0.000000, -100.000000, 24.971443, 0.000000, 1.194576, 0.9996879148480606, -3.27011737092009e-05, 0.025002935823056502, 7.4734980921807e-05, 0.000000, 0.000000, 0.000000]]).flatten()
#  delta_e, delta_a, delta_r, delta_t
# delta_trim = np.array([[-0.118662], [.009775], [-0.001611],[0.857721]])
delta_trim = np.array([[-1.24779080e-01], [-2.00606541e-03], [1.21662241e-04],[6.76753679e-01]])

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
t_history = []
x_history = []
delta_history = []
chi_history = []

Va_command = 25
h_command = 100
chi_command = np.radians(0)


while drone.state.time <= 50: #SIM.end_time:
    # if drone.state.time>100:
    #     chi_command = np.radians(5)
    drone.update(delta)
    command = np.array([chi_command,h_command,Va_command])
    delta = ctrl.update(command,drone.state)
    delta[3] = float(delta[3]) + float(delta_trim[3])
    

while drone.state.time <= 250:
    drone.update(delta)
    command = np.array([chi_command,h_command,Va_command])
    delta = ctrl.update(command,drone.state)
    delta[3] = float(delta[3]) + float(delta_trim[3])
    t_history.append(drone.state.time-50)
    x_history.append(drone.state.rigid_body)
    delta_history.append(delta)
    chi_history.append(drone.state.chi)

# Convert to numpy array
x_history = np.asarray(x_history)

plt.figure()
plt.plot(t_history, x_history[:, :3])
plt.legend(['x1', 'x2', 'x3'])
plt.show()

plt.figure()
plt.plot(t_history, -x_history[:,2])
plt.legend(['h [m]'])
plt.ylim((95,105))
plt.show()


plt.figure()
plt.plot(t_history, x_history[:, 3:6])
plt.legend(['u', 'v', 'w'])
plt.show()

delev = []
dail = []
drud = []
dthrot = []

# Convert quat to euler
nsteps = x_history.shape[0]
euler_angles = np.zeros((nsteps, 3))
for i in range(nsteps):         
    e = x_history[i, 6:10]
    euler_angles[i, :] = quat2euler(e) # phi, theta, psi = 
    delev.append(float(delta_history[i][0]))
    dail.append(float(delta_history[i][1]))
    drud.append(float(delta_history[i][2]))
    dthrot.append(float(delta_history[i][3]))

plt.figure()
plt.plot(t_history, (180/np.pi)*euler_angles)
plt.legend(['phi [deg]', 'theta [deg]', 'psi [deg]'])
plt.show()

plt.figure()
plt.plot(t_history, np.degrees(x_history[:, 10:]))
plt.legend(['p', 'q', 'r'])
plt.show()

plt.figure()
plt.plot(t_history, delev)
plt.plot(t_history, dail)
plt.plot(t_history, drud)
plt.plot(t_history, dthrot)
plt.legend(['Elev [deg]','Ail [deg]','Rud [deg]','Throt [deg]'])
plt.show()

plt.figure()
plt.plot(t_history, np.degrees(chi_history))
plt.legend(['chi [deg]'])
plt.show()