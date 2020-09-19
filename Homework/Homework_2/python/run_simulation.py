import numpy as np
import matplotlib.pyplot as plt
import integrators as intg

def f(t, x, u):
    return np.array([x[1], -x[0]+u])


t = 0; x = np.array([0, 1]); u = 0
dt = 0.1; tf = 30; n = int(np.floor(tf/dt))

integrator = intg.RK4(dt, f)



t_history = [0]
x_history = [x]
for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt

    t_history.append(t)
    x_history.append(x)

intg.__doc__
plt.figure()
plt.plot(t_history, x_history)
plt.show()