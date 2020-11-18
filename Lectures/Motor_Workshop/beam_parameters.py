# DC motor 
J = 3760/1000/100**2 # g*cm^2
d = 12/100 # cm
b = 0.001 # damping coefficient
umax = 10/1000 # g

# Simulation
x0 = 0 # rad/s
Ts = 5e-3 # s
nsteps = 1000

# PID controller
kp = 0.1
ki = 10
kd = 0
sigma = 0.1