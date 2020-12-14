import sys
sys.path.append('..')
import numpy as np
import model_coef as TF
import parameters.aerosonde_parameters as MAV

gravity = MAV.gravity  # gravity constant
rho = MAV.rho  # density of air
sigma = 0.1  # low pass filter gain for derivative
Va0 = TF.Va_trim

#----------roll loop-------------
# get transfer function data for delta_a to phi
wn_roll = 50
zeta_roll = 0.707
max_ail = np.radians(45)
max_roll = np.radians(30)
roll_kp = max_ail/max_roll
roll_kd = (2*zeta_roll*wn_roll-MAV.a_phi_1)/MAV.a_phi_2

Wx = 10

#----------course loop-------------
wn_course = wn_roll/Wx
zeta_course = 0.707
course_kp = 2*zeta_course*wn_course*MAV.Va0/9.81
course_ki = wn_course**2*MAV.Va0/9.81

#----------yaw damper-------------
yaw_damper_tau_r =1
yaw_damper_kp =1

#----------pitch loop-------------
wn_pitch =1
zeta_pitch = 0.707
pitch_kp =1
pitch_kd =1
K_theta_DC =1

#----------altitude loop-------------
wn_altitude = 2
zeta_altitude = 0.707
altitude_kp = 1
altitude_ki = 0
altitude_zone = 0   # moving saturation limit around current altitude

#---------airspeed hold using throttle---------------
wn_airspeed_throttle =1
zeta_airspeed_throttle = 0.707
airspeed_throttle_kp =1
airspeed_throttle_ki =1
