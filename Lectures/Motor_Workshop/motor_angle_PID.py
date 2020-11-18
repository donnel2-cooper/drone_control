import numpy as np
import parameters as P
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

    def update(self, r, y):
        PID_object = PIDControl(self.kp, self.ki, self.kd, self.limit, self.sigma, self.Ts, self.flag)        
        return PID_object.PID(r,y)

class System:
    def __init__(self, K, tau, omega, Ts):
        self.K = K # Motor gain
        self.tau = tau # Time Constant
        self.omega = omega # initial state
        self.Ts = Ts
        self.intg_omega = get_integrator(Ts,self.omega_dot)
        self.intg_theta = get_integrator(Ts,self.theta_dot)
        
    def omega_dot(self, t,omega,u):
        self.omega_dot = (self.K/self.tau)*u - omega/self.tau #Psuedo code EOM
        return self.omega_dot 

    def theta_dot(self,t,theta,u):
        return u

    def update_omega(self,t,y,u):
        return self.intg_omega.step(t,y,u)
    
    def update_theta(self,t,y,u):
        return self.intg_theta.step(t,y,u)
        
    
# Init system and feedback controller
system = System(P.K,P.tau,P.x0, P.Ts)
controller = Controller(P.kp,P.ki,P.kd,P.umax,P.sigma,P.Ts,flag=True)
# Simulate step response
t_history = [0]
omega_history = [0]
theta_history = [0]
u_history = [0]

r = 13
theta = 0
omega = 0.1
t = 0

for i in range(P.nsteps):
    
    u = controller.update(r,theta)
    
    t += P.Ts
    omega = system.update_omega(t,omega,u)
    theta = system.update_theta(t,theta,omega)
    t_history.append(t)
    omega_history.append(omega)
    theta_history.append(theta)
    u_history.append(u)


# Plot response theta due to step change in r
plt.figure(figsize=(8,6))
plt.plot(t_history,omega_history, label="$\omega$")
plt.plot(t_history,theta_history, label="$\Theta$")
plt.xlabel("Time [s]")
plt.legend()
plt.show()

print(f'Final angle = {theta_history[-1]}')


# Plot actuation signal
plt.figure(figsize=(8,6))
plt.plot(t_history,u_history, label="Actuation$")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Actuation [V]")
plt.show()