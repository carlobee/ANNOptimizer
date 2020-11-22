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
            
        # assign random particles to this particle's informant group
        #for x  in range(0, numberOfInformants):
            #self.informantGroup.append(random.randint(0, swarmSize-1))
            
   
   #-------- Functions of the particle ---------#     
         
    # evaluate the fitness of the current position
    def calculateFitness(self, func):
        self.currentFitness = func(self.currentPosition) # call the cost function with position param
        
        # evaluate if current fitness is better than recorded best, if so then update
        if self.bestFitness == -1 or self.currentFitness < self.bestFitness:
            self.bestFitness = self.currentFitness
            self.bestPosition = self.currentPosition
     
    # change the velocity of the particle
    def changeVelocity(self, global_best_position): #informant_best_position taken out
        weight = 0.5
        cog_constant = 1
        soc_constant = 1
        inf_constant = 1
        
        random1 = random.random()
        random2 = random.random()
        #random3 = random.random()
        
        for x in range(0, self.dimensions):
            # adapted from https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
            velocityCog = cog_constant*random1*(global_best_position[x]-self.currentPosition[x])
            velocitySoc = soc_constant*random2*(global_best_position[x]-self.currentPosition[x])
            #velocityInformant = inf_constant*random3*(informant_best_position[x]-self.currentPosition[x])
            self.velocity[x] = weight*self.velocity[x]+velocityCog+velocitySoc #+velocityInformant
            
    # change to the new position after the velocities have been applied to the particles
    def changePosition(self, bounds):
        for x in range(0, self.dimensions):
            self.currentPosition[x]=self.currentPosition[x]+self.velocity[x]
            
            # adjust maximum position if necessary
            #if self.currentPosition[x]>bounds[x][1]:
                #self.currentPosition[x]=bounds[x][1]

            # adjust minimum position if neseccary
            #if self.currentPosition[x] < bounds[x][0]:
               #self.currentPosition[x]=bounds[x][0]