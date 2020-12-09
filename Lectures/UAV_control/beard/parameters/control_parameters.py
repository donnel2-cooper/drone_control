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
wn_roll =1
zeta_roll =0.1
roll_kp =1
roll_kd =1

#----------course loop-------------
wn_course =1
zeta_course =0.1
course_kp =1
course_ki =1

#----------yaw damper-------------
yaw_damper_tau_r =1
yaw_damper_kp =1

#----------pitch loop-------------
wn_pitch =1
zeta_pitch =0.1
pitch_kp =1
pitch_kd =1
K_theta_DC =1

#----------altitude loop-------------
wn_altitude =1
zeta_altitude =0.1
altitude_kp =1
altitude_ki =1
altitude_zone =1   # moving saturation limit around current altitude

#---------airspeed hold using throttle---------------
wn_airspeed_throttle =1
zeta_airspeed_throttle =0.1
airspeed_throttle_kp =1
airspeed_throttle_ki =1
