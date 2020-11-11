import numpy as np
import parameters as P
from integrators import get_integrator
import integrators as intg
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
    def __init__(self, K, tau, omega):
        self.K = K # Motor gain
        self.tau = tau # Time Constant
        self.omega = omega # initial state
        #self.Ts = Ts

    def update(self, u):
        self.omega_dot = (self.K/self.tau)*u - self.omega/self.tau #Psuedo code EOM
        return self.omega_dot

    def omega_dot(self, t,omega,u):
        self.omega_dot = (self.K/self.tau)*u - omega/self.tau #Psuedo code EOM
        return self.omega_dot        

# Init system and feedback controller
system = System(P.K,P.tau,P.x0)
controller = Controller(P.kp,P.ki,P.kd,P.umax,P.sigma,P.Ts,flag=True)

# Simulate step response
t_history = [0]
y_history = [0]
u_history = [0]


r = 1
y = 0
t = 0
for i in range(P.nsteps):
    u = controller.update(r, y) 
    t += P.Ts
    #y_dot = system.update(u) 
    
    y = get_integrator(P.Ts,system.omega_dot,integrator="RK4").step(t,y,u) #Psuedo code to update omega dot/omega

    t_history.append(t)
    y_history.append(y)
    u_history.append(u)

# Plot response y due to step change in r
plt.figure(figsize=(8,6))
plt.plot(t_history,y_history)
plt.show()


# Plot actuation signal