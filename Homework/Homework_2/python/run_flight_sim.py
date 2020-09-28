import numpy as np
import matplotlib.pyplot as plt
import flight_sim as fs

## Aircraft Parameters (Aerosonde UAV)
mass = 13.5 #kg
J = np.matrix([[0.8244,0,-0.1204],[0,1.135,0],[-0.1204,0,1.759]]) #kg*m^2

## Initialize arone as aircraft instance
drone = fs.aircraft(mass,J)

## Sim
# Sim Parameters
t0 = 0; tf = 20; dt = 0.01; n = int(np.floor(tf/dt));
#Initial State and Initial Input
x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
u0 = np.array([float(mass),0.0,0.0,0.0,0.0,0.0])

x = x0
u = u0
t = t0
t_history = [0]
x_history = [x]

# Sim Loop
for i in range(n):
    # Step sim
    x = drone.step(t,x,u,dt)
    
    # Step time
    t = (i+1) * dt

    # Append State and Time to History
    t_history.append(t)
    x_history.append(x)
    
print(x)
