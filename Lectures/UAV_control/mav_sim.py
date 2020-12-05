import sys
sys.path.append('..')
import numpy as np
import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Quaternion
import mav_dynamics
import mavsim_python_chap5_model_coef as chap5
import matplotlib.pyplot as plt
from pid import PIDControl 
import tools.transfer_function as tf

class Controller:
    def __init__(self, kp, ki, kd, limit, sigma, Ts, flag=True):
        self.kp = kp # Proportional control gain
        self.ki = ki # Integral control gain
        self.kd = kd # Derivative control gain
        self.limit = limit # The output saturates at this limit
        self.sigma = sigma # dirty derivative bandwidth is 1/sigma 
        self.beta = (2.0*sigma-Ts)/(2.0*sigma+Ts)
        self.Ts = Ts # sample rate 
        self.flag = flag

    def update(self, r, y):
        PID_object = PIDControl(self.kp, self.ki, self.kd, self.limit, self.sigma, self.Ts, self.flag)        
        return PID_object.PID(r,y)

# Define sim parameters
dt = 0.001
uav = mav_dynamics.mavDynamics(dt)

T0 = 0
Tf = 10

n = int(np.floor((Tf-T0)/dt))

delta = chap5.u_trim

state = uav._state
alpha = [uav._alpha*180/np.pi]
beta = [uav._beta*180/np.pi]
phi = [MAV.phi0*180/np.pi]
theta = [MAV.theta0*180/np.pi]
psi = [MAV.psi0*180/np.pi]
gamma = [uav.true_state.gamma*180/np.pi]
chi = [uav.true_state.chi*180/np.pi]
t_history = [T0]
r_fb_array = [uav.true_state.r]

dail_cmd_array = [delta[1]]
drud_cmd_array = [delta[2]]

kd_q = 1;
kd_p = 1;
T_r = 1;
kr = 5;

#lat-dir controllers
course_controller = Controller(0.4,0.3,0,50,0.1,dt,True)
bank_controller = Controller(1.4,0,0,50,0.1,dt,False)
num = np.array([[T_r,0]])
den = np.array([[T_r,1]])
yaw_damper = tf.transfer_function(num,den,dt)

#lon/throttle controllers
h_controller = Controller(1,1,0,20,0.1,dt,True)
theta_controller = Controller(5,0,0,10,0.1,dt,False)
airspeed_controller = Controller(1,1,0,1,0.1,dt,True)


course_ref = uav.true_state.chi
r_ref = 0
airspeed_ref = uav.true_state.Va
h_ref = uav.true_state.h


for ii in range(n):
    #Calculate Pitch Control
    theta_cmd = h_controller.update(h_ref,uav.true_state.h)
    q_cmd = theta_controller.update(theta_cmd,uav.true_state.theta)
    delev_cmd = q_cmd-uav.true_state.q*kd_q
    
    #Calculate Roll Control
    bank_cmd = course_controller.update(course_ref,uav.true_state.chi)
    p_cmd = bank_controller.update(bank_cmd,uav.true_state.phi)
    dail_cmd = p_cmd-uav.true_state.p*kd_p
    
    #Calculate Yaw Control
    r_fb = yaw_damper.update(uav.true_state.r)*kr
    drud_cmd = -(r_ref-r_fb)
    
    #Calculate Airspeed Control
    dthrot_cmd = airspeed_controller.update(airspeed_ref,uav.true_state.Va)+chap5.u_trim[3]
    
    #Make delta
    delta = np.array([delev_cmd, dail_cmd, drud_cmd, dthrot_cmd])
    
    #Update State
    wind = np.array([[0.], [0.], [0.], [0.], [0.], [0.]])
    uav.update(delta,wind)
    uav._update_true_state()
    alpha.append(uav._alpha*180/np.pi)
    beta.append(uav._beta*180/np.pi)
    t_history.append(ii*dt)
    phi_t,theta_t,psi_t = Quaternion2Euler(uav._state[6:10])
    phi.append(phi_t*180/np.pi)
    theta.append(theta_t*180/np.pi)
    psi.append(psi_t*180/np.pi)
    gamma.append(uav.true_state.gamma*180/np.pi)
    chi.append(uav.true_state.chi*180/np.pi)
    dail_cmd_array.append(dail_cmd)
    drud_cmd_array.append(drud_cmd)
    r_fb_array.append(r_fb)
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

# Chi vs Time
plt.figure()
plt.plot(t_history,chi,'r',label = 'chi') 
plt.xlabel('t [s]')
plt.ylabel('Chi [deg]')
plt.legend()
plt.show()

# u vs Time
plt.figure()
plt.plot(t_history,dail_cmd_array,'',label = 'u_ail') 
plt.plot(t_history,drud_cmd_array,'',label = 'u_rud') 
plt.xlabel('t [s]')
plt.ylabel('u [deg]')
plt.legend()
plt.show()

# r and r_fb
plt.figure()
plt.plot(t_history,state[12,:],'',label = 'r') 
plt.plot(t_history,r_fb_array,'',label = 'r_fb') 
plt.xlabel('t [s]')
plt.ylabel('r [deg/s]')
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





# """
# mavsim_python
#     - Chapter 6 assignment for Beard & McLain, PUP, 2012
#     - Last Update:
#         2/5/2019 - RWB
#         2/24/2020 - RWB
# """
# import sys
# sys.path.append('..')
# import numpy as np
# import parameters.simulation_parameters as SIM

# from chap2.mav_viewer import mavViewer
# from chap3.data_viewer import dataViewer
# from chap4.mav_dynamics import mavDynamics
# from chap4.wind_simulation import windSimulation
# from chap6.autopilot import autopilot
# from tools.signals import signals

# # initialize the visualization
# VIDEO = False  # True==write video, False==don't write video
# mav_view = mavViewer()  # initialize the mav viewer
# data_view = dataViewer()  # initialize view of data plots
# if VIDEO is True:
#     from chap2.video_writer import videoWriter
#     video = videoWriter(video_name="chap6_video.avi",
#                         bounding_box=(0, 0, 1000, 1000),
#                         output_rate=SIM.ts_video)

# # initialize elements of the architecture
# wind = windSimulation(SIM.ts_simulation)
# mav = mavDynamics(SIM.ts_simulation)
# ctrl = autopilot(SIM.ts_simulation)

# # autopilot commands
# from message_types.msg_autopilot import msgAutopilot
# commands = msgAutopilot()
# Va_command = signals(dc_offset=25.0,
#                      amplitude=3.0,
#                      start_time=2.0,
#                      frequency=0.01)
# h_command = signals(dc_offset=100.0,
#                     amplitude=10.0,
#                     start_time=0.0,
#                     frequency=0.02)
# chi_command = signals(dc_offset=np.radians(180),
#                       amplitude=np.radians(45),
#                       start_time=5.0,
#                       frequency=0.015)

# # initialize the simulation time
# sim_time = SIM.start_time

# # main simulation loop
# print("Press Command-Q to exit...")
# while sim_time < SIM.end_time:

#     # -------autopilot commands-------------
#     commands.airspeed_command = Va_command.square(sim_time)
#     commands.course_command = chi_command.square(sim_time)
#     commands.altitude_command = h_command.square(sim_time)

#     # -------controller-------------
#     estimated_state = mav.true_state  # uses true states in the control
#     delta, commanded_state = ctrl.update(commands, estimated_state)

#     # -------physical system-------------
#     current_wind = wind.update()  # get the new wind vector
#     mav.update(delta, current_wind)  # propagate the MAV dynamics

#     # -------update viewer-------------
#     mav_view.update(mav.true_state)  # plot body of MAV
#     data_view.update(mav.true_state, # true states
#                      estimated_state, # estimated states
#                      commanded_state, # commanded states
#                      SIM.ts_simulation)
#     if VIDEO is True:
#         video.update(sim_time)

#     # -------increment time-------------
#     sim_time += SIM.ts_simulation

# if VIDEO is True:
#     video.close()



#
# e_course = course_cmd-course_fb
# bank_cmd = PID(e_course,kp,ki)
# e_bank = bank_cmd-bank_fb
# p_cmd = PID(e_bank,kp)
# dail_cmd = p_cmd-p_fb*kd
# p_new = G*dail_cmd
# bank_new = p_new*1/s
# course_new = g/Vg/s^bank_new
# course_fb = course_new
# bank_fb = bank_new
# p_fb = p_new
# ...




