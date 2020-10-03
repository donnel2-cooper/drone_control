import rigid_body as rb
import numpy as np
import integrators as intg
import aerosonde_uav as uav
import matplotlib.pyplot as plt

yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)

#x0 = np.array([0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = np.array([0.0,0.0,0.0,\
               0.0,0.0,0.0,\
               float(quat0[0]),float(quat0[1]),float(quat0[2]),float(quat0[3]),\
               np.deg2rad(10),np.deg2rad(10),np.deg2rad(0)])
x0 = x0.reshape(x0.shape[0],1)

drone = rb.RigidBody(uav.m,uav.J,x0)

t = 0
u = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
u = u.reshape(u.shape[0],1)

tf = 10; dt = 1e-2; n = int(np.floor(tf/dt));
integrator = intg.RungeKutta4(dt, drone.f)

x = x0
t_history = [0]
x_history = [x]

for i in range(n):
    x = integrator.step(t, drone.x, u)
    t = (i+1) * dt
    t_history.append(t)
    x_history.append(x)
    
    #Update drone state
    drone.x = x


print(x)
print(np.rad2deg(rb.get_euler_angles_from_rot(rb.rot_from_quat(x[6:10]))))
print(all(x == drone.x))

t_ = np.asarray(t_history)
x_ = np.asarray(x_history)