"""
particle.py

Contains implementation of the particles used to build the swarm in PSO.py.

Written By Carl Bettosi and Mamta Sofat
23/11/2020

"""

import random
import numpy as np

class particle:
    
    def __init__(self, dimensions, weight_range, learning_rate_range): #removed id swarmSize, numofinfo
        
        #--- PROPERTIES OF THE PARTICLE -------------------------------------------------------------

        #self.id = id # the unique id of the particle used for informants
        self.bestFitness = np.inf
        self.bestPosition = np.zeros((dimensions, ))
        self.dimensions = dimensions
        self.informantGroup = [] # the group of x informants that is related to this group
        
        #--- INITIALISE SOME RANDOM PARICLE PROPERTIES ----------------------------------------------
        
        # assign random velocity to paricles in every dimension
        self.velocity = np.random.uniform(learning_rate_range[0], learning_rate_range[1], (dimensions, ))
        
        # assign random velocity to paricles in every dimension
        self.currentPosition = np.random.uniform(weight_range[0], weight_range[1], (dimensions, ))
        
        #--- INFORMANTS ----------------------------------------------------------------------------
        
        # assign random particles to this particle's informant group
        #for x  in range(0, numberOfInformants):
            #self.informantGroup.append(random.randint(0, swarmSize-1))
            
