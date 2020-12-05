import numpy as np
import beam_parameters as P
from integrators import get_integrator
# import integrators as intg
from pid import PIDControl
import matplotlib.pyplot as plt

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
        self.PID_object = PIDControl(self.kp, self.ki, self.kd, self.limit, self.sigma, self.Ts, self.flag)        

    def update(self, r, y):
        return self.PID_object.PID(r,y)

class System:
    def __init__(self, J, d, b, omega, Ts):
        self.J = J # Moment of Inertia
        self.d = d # Distance between props and center of rotation
        self.b = b # Damping coefficient
        self.omega = omega # initial state
        self.Ts = Ts
        self.intg_omega = get_integrator(Ts,self.omega_dot)
        self.intg_theta = get_integrator(Ts,self.theta_dot)
        
    def omega_dot(self, t,omega,u):
        self.omega_dot = self.d/self.J*u - self.b/self.J*omega #Psuedo code EOM
        return self.omega_dot 

    def theta_dot(self,t,theta,u):
        return u

    def update_omega(self,t,y,u):
        return self.intg_omega.step(t,y,u)
    
    def update_theta(self,t,y,u):
        return self.intg_theta.step(t,y,u)
        
    
# Init system and feedback controller
system = System(P.J,P.d,P.b,P.x0, P.Ts)
PIcontroller = Controller(P.kp,P.ki,0,100000,P.sigma,P.Ts,flag=True)
Dcontroller = Controller(P.kd,0,0, P.umax, P.sigma, P.Ts,flag = True)
# Simulate step response
t_history = [0]
omega_history = [0]
theta_history = [0]
upi_history = [0]
ud_history = [0]

r = 1 # 0.78 rad = 45 deg
theta = 0
omega = 0
t = 0

for i in range(P.nsteps):
    upi = PIcontroller.update(r,theta)
    #for j in range(1,10):
    for j in range(1,2):
        ud = Dcontroller.update(upi, omega)
        omega = system.update_omega(t,omega,upi-ud)
    t += P.Ts
    theta = system.update_theta(t,theta,omega)
    t_history.append(t)
    omega_history.append(omega)
    theta_history.append(theta)
    upi_history.append(upi)
    ud_history.append(ud)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
ax1.plot(t_history,theta_history, label="θ") 
ax1.plot(t_history,np.ones(len(t_history)), label="Setpoint", linestyle = "dashed")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Position [rad]")
ax1.set_title(f"Position Control with kp={P.kp} ki={P.ki} kd={P.kd}")
ax1.legend(loc=4)

ax2.plot(t_history,omega_history, label="$\omega$") 
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Speed [rad/s]")
ax2.set_title(f"Speed with kp={P.kp} ki={P.ki} kd={P.kd}")
ax2.legend(loc=4)

ax3.plot(t_history,np.add(upi_history, ud_history), label="Total Actuation")
ax3.plot(t_history,upi_history, label="PI Input")
ax3.plot(t_history,ud_history, label="D Input")

ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Thurst [kg]")
ax3.set_title("Input Thrust")
ax3.legend(loc=4)

plt.savefig('Lectures/Motor_Workshop/Beam_Position_Control_Figs/'+f"cascade_kp_{P.kp}_ki_{P.ki}_kd_{P.kd}.png")
plt.show()



'''
# Plot response theta due to step change in r
plt.figure(figsize=(8,6))
plt.plot(t_history,omega_history, label="$\omega$")
plt.plot(t_history,theta_history, label="$\Theta$")
plt.legend()
plt.show()

print(f'Final angle = {theta_history[-1]}')

# Plot actuation signal
plt.figure(figsize=(8,6))
plt.plot(t_history,np.add(upi_history, ud_history), label="U total")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Actuation [V]")
plt.show()
'''