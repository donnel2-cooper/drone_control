import numpy as np
import integrators as intg

## Helper Functions
def sind(angle):
    #Compute sin of angle in degrees as composition of np functions
    return np.sin(np.deg2rad(angle))
def cosd(angle):
    #Compute cos of angle in degrees as composition of np functions
    return np.cos(np.deg2rad(angle))
def tand(angle):
    #Compute tan of angle in degrees as composition of np functions
    return np.tan(np.deg2rad(angle))    

class aircraft():
    def __init__(self,mass,J):
        self.mass = mass
        self.J = J

    ## Define EOM's as diff eq's
    #  xdot = f(t, x, u), 
    #   t: time
    #                              0   1   2  3  4  5    6      7    8  9 10 11
    #   x: state vector in book, [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]
    #       pn, pe, pd iniertial frame positions
    #       u, v, w ground speeds 
    #       phi, theta, psi Euler angles (in degrees)
    #       p, q, r angular rates (in dps)
    #
    #               0  1  2  3  4  5
    #   u: input, [fx,fy,fz, l, m, n]
    def f_pos(self,t, x, u):
        # Equation 3.14
        R = np.matrix([[cosd(x[7])*cosd(x[8]),\
                        sind(x[6])*sind(x[7])*cosd(x[8])-cosd(x[6])*sind(x[8]),\
                        cosd(x[6])*sind(x[7])*cosd(x[8])+sind(x[6])*sind(x[8])],\
                       [cosd(x[7])*sind(x[8]),\
                        sind(x[6])*sind(x[7])*sind(x[8])+cosd(x[6])*cosd(x[8]),\
                        cosd(x[6])*sind(x[7])*sind(x[8])-sind(x[6])*cosd(x[8])],\
                       [-sind(x[7]),\
                        sind(x[6])*cosd(x[7]),\
                        cosd(x[6])*cosd(x[7])]])
        substate = np.array([[x[3]],[x[4]],[x[5]]])
        mat_mult_result = np.matmul(R,substate)
        return np.array([float(mat_mult_result[0]),float(mat_mult_result[1]),\
                         float(mat_mult_result[2]),0,0,0,0,0,0,0,0,0])
    
    def f_vel(self,t, x, u):
        # Equation 3.15
        f_over_m = np.array([[u[0]/self.mass],[u[1]/self.mass],[u[2]/self.mass]])
        first_term = np.array([[x[11]*x[4]-x[10]*x[5]],\
                               [x[9]*x[5]-x[11]*x[3]],\
                               [x[10]*x[3]-x[9]*x[4]]])
        summation = np.add(first_term,f_over_m)
        return np.array([0,0,0,float(summation[0]),float(summation[1]),\
                         float(summation[2]),0,0,0,0,0,0])
    
    def f_euler(self,t,x,u):
        # Equation 3.16
        R = np.matrix([[1,\
                        sind(x[6])*tand(x[7]),\
                        cosd(x[6])*tand(x[7])],\
                       [0,\
                        cosd(x[6]),\
                        -sind(x[6])],\
                       [0,\
                        sind(x[6])/cosd(x[7]),\
                        cosd(x[6])/cosd(x[7])]])
        substate = np.array([[x[9]],[x[10]],[x[11]]])
        mat_mul_result = np.matmul(R,substate)
        return np.array([0,0,0,0,0,0,float(mat_mul_result[0]),\
                         float(mat_mul_result[1]),float(mat_mul_result[2]),0,0,0])
    
    def f_rate(self,t,x,u):
        # Equations 3.13
        gamma = self.J[0,0]*self.J[2,2]-(-self.J[0,2])**2
        gamma_1 = ((-self.J[0,2])*(self.J[0,0]-self.J[1,1]+self.J[2,2]))/gamma
        gamma_2 = (self.J[2,2]*(self.J[2,2]-self.J[1,1])+(-self.J[0,2])**2)/gamma
        gamma_3 = self.J[2,2]/gamma
        gamma_4 = -self.J[0,2]/gamma
        gamma_5 = (self.J[2,2]-self.J[0,0])/self.J[1,1]
        gamma_6 = -self.J[0,2]/self.J[1,1]
        gamma_7 = ((self.J[0,0]-self.J[1,1])*self.J[0,0]+(-self.J[0,2])**2)/gamma
        gamma_8 = self.J[0,0]/gamma
        #Equation 3.17
        first_term = np.array([[gamma_1*x[9]*x[10]-gamma_2*x[10]*x[11]],\
                               [gamma_5*x[9]*x[11]-gamma_6*(x[9]**2-x[11]**2)],\
                               [gamma_7*x[9]*x[10]-gamma_1*x[10]*x[11]]])
        second_term = np.array([[gamma_3*u[3]+gamma_4*u[5]],\
                                [u[4]/self.J[1,1]],\
                                [gamma_4*u[3]+gamma_8*u[5]]])
        summation = np.add(first_term,second_term)
        return np.array([0,0,0,0,0,0,0,0,0,float(summation[0]),\
                         float(summation[1]),float(summation[2])])
            
    def step(self,t,x,u,dt = 0.01):
        # Integrate vehicle state forward one time step dt
        inertial_pos_intg = intg.RungeKutta4(dt,self.f_pos)
        velocity_intg = intg.RungeKutta4(dt,self.f_vel)
        euler_angles_intg = intg.RungeKutta4(dt,self.f_euler)
        rates_intg = intg.RungeKutta4(dt,self.f_rate)
        inertial_pos = inertial_pos_intg.step(t,x,u)
        velocity = velocity_intg.step(t,x,u)
        euler_angles = euler_angles_intg.step(t,x,u)
        rates = rates_intg.step(t,x,u)
        return np.concatenate((inertial_pos[:3],velocity[3:6],euler_angles[6:9],rates[9:]))
