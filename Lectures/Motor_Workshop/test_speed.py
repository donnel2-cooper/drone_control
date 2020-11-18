import numpy as np
import parameters as P
from integrators import get_integrator
from pid import PIDControl
import matplotlib.pyplot as plt


PID_object = PIDControl(P.kp, P.ki, P.kd, P.umax, P.sigma, P.Ts)

def omega_dot(t,omega,u):
    alpha = (P.K/P.tau)*u - omega/P.tau #Psuedo code EOM
    return alpha     


intg = get_integrator(P.Ts, omega_dot)

class Controller:
    def __init__(self):
        pass
        
    def update(self, r, y):
        return PID_object.PID(r,y)



class System:
    def __init__(self):
        pass
   
    def update(self,t,y, u):
        return intg.step(t,y,u)



system = System()
controller = Controller()

theta_i = 0
w_i = 0

r = 1
y0 = np.array([theta_i])
t = 0

t_history = [0]
y_history = [y0]
u_history = [0]

y = y0

for i in range(P.nsteps):
    u = controller.update(r, y) 
    y = system.update(t, y, u) 
    t += P.Ts

    t_history.append(t)
    y_history.append(y)
    u_history.append(u)



t_ = np.asarray(t_history)
y_ = np.asarray(y_history)
# Plot response y due to step change in r
plt.figure()
plt.plot(t_history, y_[:,0])
plt.show()
# Plot actuation signal


t_ = np.asarray(t_history)
u_ = np.asarray(u_history)
# Plot response y due to step change in r
plt.figure()
plt.plot(t_history, u_)
plt.show()
# Plot actuation signal
