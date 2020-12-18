import sys
import numpy as np
import control_parameters as AP
import pid
import rotations
from control import matlab

class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = pid.PIDControl(kp=AP.roll_kp,kd=0,Ts=ts_control,limit=np.radians(45))
        self.course_from_roll = pid.PIDControl(kp=AP.course_kp,ki=AP.course_ki,Ts=ts_control,limit=np.radians(30))
        self.yaw_damper = pid.PIDControl(kp=AP.sideslip_kp,ki=AP.sideslip_ki,Ts=ts_control,limit=np.radians(45))

        # instantiate lateral controllers
        self.pitch_from_elevator = pid.PIDControl(kp=AP.pitch_kp,kd=0,limit=np.radians(45))
        self.altitude_from_pitch = pid.PIDControl(kp=AP.altitude_kp,ki=AP.altitude_ki,Ts=ts_control,limit=np.radians(30))
        self.airspeed_from_throttle = pid.PIDControl(kp=AP.airspeed_throttle_kp,ki=AP.airspeed_throttle_ki,Ts=ts_control,limit=1.5,flag=True)

    def update(self, cmd, state):

        # lateral autopilot
        phi_c = self.course_from_roll.PID(cmd[0],state.chi)
        delta_a_prime = self.roll_from_aileron.PID(phi_c, state.phi)
        delta_a = delta_a_prime-AP.roll_kd*state.p
        delta_r = self.yaw_damper.PID(0,state.beta)

        # longitudinal autopilot
        h_c = cmd[1]
        theta_c = self.altitude_from_pitch.PID(h_c, -state.pd)
        delta_e_prime = self.pitch_from_elevator.PID(theta_c, state.theta)
        delta_e = delta_e_prime-AP.pitch_kd*state.q
        delta_t = self.airspeed_from_throttle.PID(cmd[2], state.airspeed)

        # construct output
        delta = np.array([[delta_e],[delta_a],[delta_r],[delta_t]])
        return delta

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output