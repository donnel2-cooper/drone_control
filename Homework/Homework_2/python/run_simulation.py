import numpy as np
import matplotlib.pyplot as plt
import integrators as intg

# Parameters mass-spring system
m = 1
b = 0.25
k = 1


#  xdot = f(t, x, u), 
#   t: time
#   x: state vector
#   u: input
def f(t, x, u):
    # mass sprint system state space form:
    return np.array([x[1], -k*x[0]/m - b*x[1]/m])
        
# Initial conditions
t = 0 
x0 = np.array([0, 1])
u = 0

# Numerical solution
dt = 0.1; n = 100

integrator = intg.Heun(dt, f)
#integrator = intg.RungeKutta4(dt, f)

x = x0
t_history = [0]
x_history = [x]

for i in range(n):
    
    x = integrator.step(t, x, u)
    t = (i+1) * dt

    t_history.append(t)
    x_history.append(x)


# Convert numerical soln to numpy arrays
t_ = np.asarray(t_history)
x_ = np.asarray(x_history)

# Analytical solution
wn = np.sqrt(k/m)
z = b/(2*m*wn)
sigma = z*wn
wd = wn*np.sqrt(1 - z**2)
A = x0[0]
B = (x0[1] + sigma*x0[0])/wd
x_exact = np.exp(-sigma*t_)*(A*np.cos(wd*t_) + B*np.sin(wd*t_))


# Error
error = x_exact - x_[:,0]


# close all figures
plt.close('all')

plt.figure()
plt.plot(t_history, x_[:,0],'rx', label='Heun')
plt.plot(t_history, x_exact,'g--', label='exact')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_history,error,'g--')
plt.xlabel('t')
plt.ylabel('error')
plt.show()