"""
Template for implementing QLearner  (c) 2018 T. Ruzmetov
"""

import numpy as np
import random as rand

class QLearner(object):

    """
    
    Briefly describe! Bla Bla Bla, balck panter lives, ya Wakanda!

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
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.rar = rar
        self.radr = radr
        #Uniformly Initialize Q-table with float values [-1, 1]
        self.Q = np.random.rand(num_states, num_actions) * 2 - 1

        # Initialize transition matrix T, transition count matrix T_c, and reward matrix R for 'dyna'.
        if self.dyna != 0:
            self.Tc = 0.00001 * np.ones((num_states, num_actions, num_states))
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            self.R = -1.0 * np.ones((num_states, num_actions))
        
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



        # Performe Dyna id specified
        if self.dyna != 0:
            # Increment count of current transition.
            self.Tc[self.s, self.a, s_prime] += 1
            
            #Normalize all counts to produce the correct probabilities in the transition matrix.
            self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / self.Tc[self.s, self.a, :].sum()
            
            #Update rewards
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            #run dyna
            self._run_dyna()

            
            
        action = np.argmax(self.Q[s_prime,:])  # take best action
        if rand.uniform(0.0, 1.0) <= self.rar: # take rand action conditionally
            action = rand.randint(0,3)         # choose the random direction
                
        self.rar = self.rar * self.radr        # update rar globally
        self.s = s_prime                       # update current state globally
        self.a = action                        # update action globally 
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        return action

    def _run_dyna(self):
        # Generate state and action samples to speed up hallucination.
        s_samples = np.random.randint(0, self.num_states, self.dyna)
        a_samples = np.random.randint(0, self.num_actions, self.dyna)

        Q1 = self.Q
        T1 = self.T
        # For each sample...
        for i in range(self.dyna):
            s = s_samples[i]
            a = a_samples[i]
            # Simulate an action with the transition model and land on an s_prime
            s_prime = np.argmax(np.random.multinomial(1, T1[s, a, :]))
            # Compute reward of simulated action.
            r = self.R[s, a]
            # Update Q
            Q1[s, a] = (1 - self.alpha) * Q1[s, a] + \
                           self.alpha * (r + self.gamma * np.max(Q1[s_prime,:]))
    
        self.Q[s,a] = Q1[s,a]
            
    def author(self):
        return 'truzmetov3'

if __name__=="__main__":
    print "I'm not fan of Star Trek, but like MCU"
