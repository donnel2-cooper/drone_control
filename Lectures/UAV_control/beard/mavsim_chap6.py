"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
# import parameters.simulation_parameters as SIM

from data_viewer import dataViewer
from mav_dynamics import mavDynamics
from wind_simulation import windSimulation
from autopilot import autopilot
from tools.signals import signals
import matplotlib.pyplot as plt
import copy
import time


# initialize the simulation time
# sim_time = SIM.start_time

sim_time = 0
end_time = 20
ts_simulation = 0.01


# initialize elements of the architecture
# wind = windSimulation(SIM.ts_simulation)
# mav = mavDynamics(SIM.ts_simulation)
# ctrl = autopilot(SIM.ts_simulation)

wind = windSimulation(ts_simulation)
mav = mavDynamics(ts_simulation)
ctrl = autopilot(ts_simulation)

# autopilot commands
from message_types.msg_autopilot import msgAutopilot
commands = msgAutopilot()


Va_command = signals(dc_offset=25.0,
                     #amplitude=3.0,
                     amplitude=0.0,
                     start_time=2.0,
                     frequency=0.01)
h_command = signals(dc_offset=100.0,
                    #amplitude=10.0,
                    amplitude=0.0,
                    start_time=0.0,
                    frequency=0.02)
chi_command = signals(dc_offset=np.radians(0),
                      #amplitude=np.radians(45),
                      amplitude=np.radians(0),
                      start_time=5.0,
                      frequency=0.015)



t_history = []
state_history = []
delta_history = []

Va_command_hist = []
h_command_hist = []
chi_command_hist = []

t0 = time.time()
# main simulation loop
# while sim_time < SIM.end_time:
while sim_time < end_time:
    
    # -------autopilot commands-------------
    # commands.airspeed_command = Va_command.square(sim_time)
    # commands.course_command = chi_command.square(sim_time)
    # commands.altitude_command = h_command.square(sim_time)
    commands.airspeed_command = Va_command.step(sim_time)
    commands.course_command = chi_command.step(sim_time)
    commands.altitude_command = h_command.step(sim_time)

    
    Va_command_hist.append(copy.copy(commands.airspeed_command))
    chi_command_hist.append(copy.copy(commands.course_command))
    h_command_hist.append(copy.copy(commands.altitude_command))

    # -------controller-------------
    estimated_state = mav.true_state  # uses true states in the control
    delta, commanded_state = ctrl.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------store simulation resutls-------------
    t_history.append(sim_time)
    state_history.append(copy.copy(estimated_state))
    delta_history.append(copy.copy(delta))
    # print(estimated_state.pn, estimated_state.beta)
    
    # -------increment time-------------
    # sim_time += SIM.ts_simulation
    sim_time += ts_simulation

tf = time.time()
print(f"Simulation finished in {round(tf-t0,2)} seconds.")

#state_history = np.array(state_history)
#delta_history = np.array(delta_history)


# beta = []

# for i in range(0,len(state_history)):
#     print(state_history[i].pn)

alpha = [state.alpha for state in state_history]
plt.figure()
plt.plot(t_history,alpha,'r',label = 'aoa') 
plt.xlabel('t')
plt.ylabel('angle [deg]')
plt.legend()
plt.show()


beta = [state.beta for state in state_history]
plt.figure()
plt.plot(t_history,beta,'r',label = 'aos') 
plt.xlabel('t')
plt.ylabel('angle [deg]')
plt.legend()
plt.show()


# Pos vs Time

pn = [state.pn for state in state_history]
pe = [state.pe for state in state_history]
h = [state.h for state in state_history]

plt.figure()
plt.plot(t_history,pn,'r',label = 'pn') 
plt.plot(t_history,pe,'b',label = 'pe') 
plt.plot(t_history,h,'g',label = 'height') 
plt.xlabel('t [s]')
plt.ylabel('Inertial Positions [m]')
plt.legend()
plt.show()


# pn = [state.pn for state in state_history]
# pe = [state.pe for state in state_history]
# h = [state.h for state in state_history]

plt.figure()
plt.plot(t_history,Va_command_hist,'r',label = 'Va cmd') 
plt.plot(t_history,h_command_hist,'b',label = 'H cmd') 
plt.plot(t_history,chi_command_hist,'g',label = 'chi cmd') 
plt.xlabel('t [s]')
plt.ylabel('Cmd')
plt.legend()
plt.show()


# # Vel vs Time
# plt.figure()
# plt.plot(t_history,state[3,:],'r',label = 'u') 
# plt.plot(t_history,state[4,:],'b',label = 'v') 
# plt.plot(t_history,state[5,:],'g',label = 'w') 
# plt.xlabel('t [s]')
# plt.ylabel('Air Speed [m/s]')
# plt.legend()
# plt.show()

# # Euler Angles vs Time
# plt.figure()
# plt.plot(t_history,phi,'r',label = 'phi') 
# plt.plot(t_history,theta,'b',label = 'theta') 
# plt.plot(t_history,psi,'g',label = 'psi') 
# plt.xlabel('t [s]')
# plt.ylabel('Euler Angle [deg]')
# plt.legend()
# plt.show()

# # Ang Rates vs Time
# plt.figure()
# plt.plot(t_history,state[10,:],'r',label = 'p') 
# plt.plot(t_history,state[11,:],'b',label = 'q') 
# plt.plot(t_history,state[12,:],'g',label = 'r') 
# plt.xlabel('t [s]')
# plt.ylabel('Angular rates [deg/s]')
# plt.legend()
# plt.show()