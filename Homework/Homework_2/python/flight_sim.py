# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:17:46 2020

@author: jackd
"""
import numpy as np

class ac_params:
    def __init__(self, mass, inertia_matrix):
        self.mass = mass
        self.inertia_matrix = inertia_matrix

class ac_state:
    def __init__(self,pn,pe,pd,u,v,w,phi,theta,psi,p,q,r):
        self.pn = pn
        self.pe = pe
        self.pd = pd
        self.u = u
        self.v = v
        self.w = w
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.p = p
        self.q = q
        self.r = r
        
    def R_BI():
        #Define this later
        return np.matrix((0))
        

class aircraft:
    def __init__(self, params,state):
        self.params = params
        self.state = state
    def f_position(t,self.state,u):
        pos_states = np.array((self.state.pn,self.state.pe,self.state.pd))
        return pos_states
    def update_ac_params(self):
        # Placeholder
        return self.params
    def update_ac_state(t,ac_state,u):
        # Placeholder
        return ac_state
        
        
        
    def time_step(self, dt):
        #self.update_ac_params()
        #self.update_ac_state()
        self.state.pn = self.state.pn+self.state.u*dt
        return self.state
        