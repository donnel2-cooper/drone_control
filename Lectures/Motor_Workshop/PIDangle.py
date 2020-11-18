import numpy as np
import parameters as P
from integrators import get_integrator
# import integrators as intg
from pid import PIDControl
import matplotlib.pyplot as plt
import control 

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
        self.intg_state = get_integrator(Ts,self.state)

    def omega_dot(self, t,omega,u):
        self.alpha = (self.K/self.tau)*u - omega/self.tau #Psuedo code EOM
        return self.alpha 

    def state(self, t, x, u):
        state_vector = np.zeros(2)
        state_vector[0] = x[1]
        state_vector[1] = self.K/self.tau*u[0] - x[1]/self.tau
        return state_vector


    def theta_dot(self,t,theta,u):
        return u

    def update_omega(self,t,y,u):
        return self.intg_omega.step(t,y,u)
    
    def update_theta(self,t,y,u):
        return self.intg_theta.step(t,y,u)
        
    def update_state(self,t,y,u):
        return self.intg_state.step(t,y,u)
    
# Init system and feedback controller
system = System(P.K,P.tau,P.x0, P.Ts)
controller = Controller(P.kp,P.ki,P.kd,P.umax,P.sigma,P.Ts,flag=True)
controller2 = Controller(P.kp,P.ki,P.kd,P.umax,P.sigma,P.Ts,flag=True)
# Simulate step response
r = 1
theta = 0
omega = 0.1
t = 0

t_history = [0]
omega_history = [0]
theta_history = [0]
u_history = [0]
x_0 = np.array([theta, omega])
x_history = [x_0]
x = x_0




for i in range(P.nsteps):
    #Harry's
    """    
    u = controller.update(r,theta)
    omega = system.update_omega(t,omega,u)
    theta = system.update_theta(t,theta,omega)
    """
    #State space
    
    u2 = controller2.update(r,x)
    x = system.update_state(t,x,u2)
    
    t += P.Ts
    t_history.append(t)
    omega_history.append(omega)
    theta_history.append(theta)
    x_history.append(x)
    u_history.append(u2)

'''
# Control Library

s = control.TransferFunction.s
K = P.K
kp = P.kp
ki = P.ki
kd = P.kd
tau = P.tau
G_yr = K*(kp+s*kd)/(tau*s**2+s+K*(kp+s*kd))

T_step, y_step = control.step_response(G_yr,t_history)
plt.figure(figsize=(8,6))
plt.plot(T_step,y_step, label="$\omega$")
plt.xlabel("Time [s]")
plt.ylabel("Speed [rad/s]")
plt.legend()
plt.show()
'''
x_history = np.array(x_history)

# Plot response theta due to step change in r
plt.figure(figsize=(8,6))
plt.plot(t_history,x_history[:,0], label="$\omega$")
plt.plot(t_history,x_history[:,1], label="$\Theta$")
#plt.plot(T_step,y_step, label="$\omega$2")
plt.xlabel("Time [s]")
plt.legend()
plt.show()

print(f'Final angle = {theta_history[-1]}')

u_history.pop(0)
t_history.pop(0)
t_history.pop(0)

u_history = np.array(u_history)
# Plot actuation signal
plt.figure(figsize=(8,6))
plt.plot(t_history,u_history[1:,0], label="1$")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Actuation [V]")
plt.show()