import numpy as np
import matplotlib.pyplot as plt
import integrators as intg

# Define diff eq xdot = f(t,x,u)
def f(t, x, u):
    A = np.array([[   0,    1],
                  [-k/m, -b/m]])
    B = np.array([0, 1/m])
    state = np.array([x[0],x[1]])
    return np.dot(A,state)+np.dot(B, u)

def underdamped_mass_spring_damper(t,x,m,b,k):
    # System Dependant Solution Parameters (0<zeta<1)
    wn = np.sqrt(k/m);
    zeta = b/(2*wn*m);
    if zeta > 1:
        raise Exception('Overdamped System')
    sigma = zeta*wn
    wd = wn*np.sqrt(1-zeta**2)
    # Parameters that depend on initial state
    alpha = x[0]/2;
    beta = -(x[1]+sigma*x[0])/(2*wd);

    return np.array([2*np.exp(-sigma*t)*(alpha*np.cos(wd*t)-beta*np.sin(wd*t)), 
                     2*np.exp(-sigma*t)*((-sigma*alpha-beta*wd)*np.cos(wd*t)+(sigma*beta-alpha*wd)*np.sin(wd*t))])

# System Parameters
m = 1.0; #kg
b = 0.25; #kg/s
k = 1.0; #kg/s^2


#Simulation Parameters
t = 0; x = np.array([0.0,0.0]); u = 1
tf = 30; dt = 1e-2; n = int(np.floor(tf/dt));

integrator_euler = intg.Euler(dt, f);
integrator_heun = intg.Heun(dt, f);


t_history = [0]
x_history_euler = [x]
x_history_heun = [x]
x_anal_hist = [x]
err_euler_hist = [np.array([0.0,0.0])]
err_heun_hist = [np.array([0.0,0.0])]

# Init Arrays
x_euler = x
x_heun = x

#Loop to integrate xdot = f(t,x,u)
for i in range(n):
    # Integrated Solution
    x_euler = integrator_euler.step(t, x_euler, u)
    x_heun = integrator_heun.step(t, x_heun, u)
    # Analytic Solution
    x_anal = underdamped_mass_spring_damper(t,x,m,b,k);
    # Error in Integrated Solution
    err_euler = x_euler-x_anal
    err_heun = x_heun-x_anal
    # Step time
    t = (i+1) * dt
    # Record Values
    t_history.append(t)
    x_history_euler.append(x_euler)
    x_history_heun.append(x_heun)
    x_anal_hist.append(x_anal)
    err_euler_hist.append(err_euler)
    err_heun_hist.append(err_heun)

#intg.__doc__
#sz = 5
#plt.figure(figsize=(8, 8),dpi=800)
#fig, axs = plt.subplots(2, 1, constrained_layout=True)
#axs[0].plot(t_history, x_history_euler, '-', t_history, x_anal_hist)
#axs[0].set_title('Numerical vs. Analytical States')
#axs[0].set_xlabel('Time(s)')
#axs[0].set_ylabel('Amplitude')
#axs[0].legend(['Numerical Position', 'Numerical Velocity','Analytical Position', 'Analytical Velocity'],loc='upper right', prop={'size': sz})
#fig.suptitle('Heun, dt = '+str(dt)+'s', fontsize=16)

#axs[1].plot(t_history, err_euler_hist)
#axs[1].set_xlabel('Time(s)')
#axs[1].set_title('State Errors')
#axs[1].set_ylabel('Amplitude')
#axs[1].legend(['Position Error', 'Velocity Error'],loc='upper right', prop={'size': sz})
#plt.show()

intg.__doc__
sz = 5
plt.figure(figsize=(8, 8),dpi=800)
plt.plot(t_history, x_history_heun)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.legend(['Numerical Position', 'Numerical Velocity','Analytical Position', 'Analytical Velocity'],loc='upper right', prop={'size': sz})
plt.title('Step Response', fontsize=16)
plt.show()