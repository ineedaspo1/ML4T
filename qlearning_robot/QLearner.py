"""
Template for implementing QLearner  (c) 2018 T. Ruzmetov
"""

import numpy as np
import random as rand

class QLearner(object):

    """
    
    Briefly describe! Bla Bla Bla

    --------------------------
    Actions: There are 4 possible actions
    0 - move north 
    1 - move east
    2 - move south
    3 - move west
    
    States: Mapping state from 2D to 1D  state = icol * 10 + irow, why on
    earth they called it descritization, isn't is just mapping?
    """
    
    def __init__(self, num_states=100, num_actions = 4, alpha = 0.2, \
                 gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.rar = rar
        self.radr = radr
        #Uniformly Initialize Q-table with float values [-1, 1]
        self.Q = np.random.rand(num_states, num_actions) * 2 - 1

    
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        #rand_action = rand.randint(0, self.num_actions-1)
        action = np.argmax(self.Q[s,:])
        if self.verbose: print "s =", s,"a =",action
        return action

    
    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state reward
        @returns: The selected action
        """

        s = self.s # current state
        a = self.a # current action

        # update Q-table by looking into future state(s_prime)
        self.Q[s,a] = (1-self.alpha)*self.Q[s,a] + \
                           self.alpha*(r + self.gamma * np.max(self.Q[s_prime,:]))  
        
        action = np.argmax(self.Q[s_prime,:])  # take best action
        if rand.uniform(0.0, 1.0) <= self.rar: # take rand action conditionally
            action = rand.randint(0,3)         # choose the random direction
        
        self.rar = self.rar * self.radr        # update rar globally
        self.s = s_prime                       # update current state globally
        self.a = action                        # update action globally 
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

    
    def author(self):
        return 'truzmetov3'

if __name__=="__main__":
    print "I'm not fan of Star Trek, but like MCU"
