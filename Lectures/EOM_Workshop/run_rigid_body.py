import rigid_body as rb
import numpy as np
import integrators as intg
import aerosonde_uav as uav

x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)

drone = rb.RigidBody(uav.m,uav.J,x0)

t = 0
u = np.array([0.0,0.0,0.0,0.0,0.0,uav.J[2,2]])
u = u.reshape(u.shape[0],1)

tf = 10; dt = 0.001; n = int(np.floor(tf/dt));
integrator = intg.Euler(dt, drone.f)

x = x0
t_history = [0]
x_history = [x]

for i in range(n):
    x = integrator.step(t, x, u)
    t = (i+1) * dt
    t_history.append(t)
    x_history.append(x)
    
    #Update drone state
    drone.x = x


print(x)
print(rb.get_euler_angles_from_rot(rb.rot_from_quat(x[6:10])))