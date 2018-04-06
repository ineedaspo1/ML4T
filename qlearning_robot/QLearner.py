"""
Template for implementing QLearner  (c) 2018 T. Ruzmetov
"""

import numpy as np
import random as rand

class QLearner(object):

    """
    
    Briefly describe!

    --------------------------
    Actions: There are 4 possible actions
    0 - move north 
    1 - move east
    2 - move south
    3 - move west
    
    States: Mapping state from 2D to 1D
    state = icol * 10 + irow 
   
    """
    
    def __init__(self, num_states=100, num_actions = 4, alpha = 0.2, \
                 gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.rar = rar
        self.radr = radr
        
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        
        """
        self.s = s
        rand_action = rand.randint(0, self.num_actions-1)
        best_action = 2 # choose action that gives max Q

        
        
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
