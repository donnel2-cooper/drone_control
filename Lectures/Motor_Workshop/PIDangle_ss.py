import numpy as np
import parameters as P
from integrators import get_integrator
import integrators as intg
from pid import PIDControl
import matplotlib.pyplot as plt

def f_omega(t,omega,u):
    return (P.K*u-omega)/P.tau

class System:
    def __init__(self, K, tau, omega, Ts):
        self.K = K # Motor gain
        self.tau = tau # Time Constant
        self.omega = omega # initial state
        self.Ts = Ts
        
    def f(t,x,u):
        return np.array([x[1],-1/P.tau*x[1]+P.K/P.tau*u])

# Initial Conditions
r0 = 1
theta0 = 0
omega0 = 0
t0 = 0
x0 = np.array([theta0,omega0])

C = PIDControl(P.kp,P.ki,P.kd,P.umax,P.sigma,P.Ts,flag=True)
plant = System(P.K,P.tau,omega0,P.Ts)

integrator_RK4 = intg.get_integrator(P.Ts, System.f)

# Simulate step response
t_history = [t0]
x_history = [x0]
u_history = [0]

r = r0
x = np.array([theta0,omega0])
t = t0

for i in range(P.nsteps):
    u = C.PID(r,x[0])
    t += P.Ts
    x = integrator_RK4.step(t,x,u)
    t_history.append(t)
    x_history.append(x)
    u_history.append(u)


# Plot response theta due to step change in r
plt.figure(figsize=(8,6))
plt.plot(t_history,x_history, label="$\omega$")
plt.xlabel("Time [s]")
plt.ylabel("Speed [rad/s]")
plt.legend()
plt.show()

# print(f'Final speed = {omega_history[-1]}')


# Plot actuation signal
plt.figure(figsize=(8,6))
plt.plot(t_history,u_history, label="Actuation")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Actuation [V]")
plt.show()



