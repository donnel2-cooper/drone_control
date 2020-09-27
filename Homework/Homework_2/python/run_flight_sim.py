# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:59:05 2020

@author: jackd
"""

import numpy as np
import matplotlib.pyplot as plt
import integrators as intg
import flight_sim as fs

#Initialize Sim Params
dt = 0.01

# Initialize Drone
initial_params = fs.ac_params(10,np.matrix(((1,0,0),(0,1,0),(0,0,1))))
initial_state = fs.ac_state(0,0,0,10,0,0,0,0,0,0,0,0)
drone = fs.aircraft(initial_params,initial_state)

for ii in range(100):
    print(drone.time_step(dt).pn)
    
    
print(drone.f_position(