import rigid_body as rb
import numpy as np
import integrators as intg
import aerosonde_uav as uav

phi = 0
theta = 0
psi = 0


#print(rb.get_euler_angles_from_rot(rb.rot_from_quat(np.array([0,1,0,0]))))

x0 = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(uav.m,uav.J,x0)

#print(uav.m)
#print(uav.J)
t = 0
u = np.array([0,0,uav.m,0,0,0])
u = u.reshape(u.shape[0],1)
#print("u = ", u)
#print(drone.f(t,x0,u))

tf = 30; dt = 0.001; n = int(np.floor(tf/dt));
integrator = intg.RungeKutta4(dt, drone.f)

x = x0
t_history = [0]
x_history = [x]

for i in range(n):
    
    x = integrator.step(t, drone.f(t,x,u), u)
    t = (i+1) * dt

    t_history.append(t)
    x_history.append(x)
    


print(x)