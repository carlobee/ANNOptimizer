"""
A program that implements a the Particle Swarm Optimization Algorithm

Written By Carl Bettosi and Mamta Sofat
17/11/2020

"""
import numpy as np
import random

from particle import particle


class PSO():
    
    # construct
    def __init__(self, swarmSize, dimensions, weight_range, learning_rate_range, inertia_range, cog_constant, soc_constant, numberOfInformants):
        
        #--- PROPERTIES OF SWARM -----------------------------------------------------------------------
        
        self.GLOBAL_fitnessBest = -1
        self.GLOBAL_positionBest = []
        self.weight_range = weight_range
        self.learning_rate_range = learning_rate_range
        self.inertia_range = inertia_range
        self.cog_constant = cog_constant
        self.soc_constant = soc_constant
        self.dimensions = dimensions
        self.swarmSize = swarmSize
        self.numberOfInformants = numberOfInformants
         
        #--- GENERATE SWARM ---------------------------------------------------------------------------- 

        self.swarmArray = []     # create array to hold swarm particles
        
        for x in range(0, self.swarmSize):
            self.swarmArray.append(particle(dimensions, weight_range, learning_rate_range)) #removed x, swarmSize, numofinfo
        
        
        
        print(f"\nParticles with swarm size {swarmSize} generated.\n")

    
    #--- INFORMANT FUNCTIONS ----------------------------------------------------------------------------
    
    # get the id of the best informant
    def get_informant_id(self, id):
             
        best_informant_fitness = -1
        best_informant_id = -1
             
        for x in range(0, self.numberOfInformants):
            check_id = self.swarmArray[id].informantGroup[x]
                   
            #best_position = []
                 
            if self.swarmArray[check_id].bestFitness < best_informant_fitness or best_informant_fitness == -1:
                best_informant_fitness = self.swarmArray[check_id].bestFitness
                best_informant_id = check_id
             
        return check_id
         
    # get the fitness of the best informant
    def get_informant_best_fitness(self, id):
        return self.swarmArray[id].bestFitness
    
    # get the position of the best informant
    def get_informant_best_position(self, id):
        return self.swarmArray[id].bestPosition
        
        
    #--- OPTIMISATION FUNCTION ----------------------------------------------------------------------------
    
    def optimize(self, func, input_values_x, input_values_y, iterations):
        
        i=0
        while i < iterations:
            print("Iteration: " + str(i+1))
            
            informant_best_fitness = -1
            informant_best_position = []
            
            # for each particle in swarm, calculate fitness and update object's attributes to reflect
            for x in range(0, self.swarmSize):
                currentFitness = func(input_values_x, input_values_y, self.swarmArray[x])
                
                # evaluate if current fitness is better than recorded best, if so then update
                if currentFitness < self.swarmArray[x].bestFitness:
                    self.swarmArray[x].bestFitness = currentFitness
                    self.swarmArray[x].bestPosition = self.swarmArray[x].currentPosition
                
                # get informant bests for this particle's group and set them to particle's informant bests
                #best_id = self.get_informant_id(x)
                #informant_best_fitness = self.get_informant_best_fitness(best_id)
                #informant_best_position = self.get_informant_best_position(best_id)
                
                
                # check if this particle is the best in its informant group, if so then change
                #if currentFitness < informant_best_fitness or informant_best_fitness == -1:
                    #informant_best_fitness = currentFitness
                    
                # check if this particle us the best in global group
                if currentFitness < self.GLOBAL_fitnessBest or self.GLOBAL_fitnessBest == -1:
                    
                    self.GLOBAL_fitnessBest = currentFitness
                    self.GLOBAL_positionBest = self.swarmArray[x].currentPosition
            
            
            for x in self.swarmArray:
                
                weight = np.random.uniform(self.weight_range[0], self.weight_range[1], 1)[0]
                
                rand1 = random.random()
                rand2 = random.random()
                
                # update velocity 
                x.velocity = weight * x.velocity + (self.cog_constant * rand1) * \
                (x.bestPosition - x.currentPosition) + (self.soc_constant * rand2) \
                * (self.GLOBAL_positionBest - x.currentPosition)
                
                # update position
                x.currentPosition = x.currentPosition + x.velocity

            print("Current best: " + str(self.GLOBAL_fitnessBest))
            ##for x in range(0, self.swarmSize):
             ##   self.swarmArray[x].changeVelocity(self.GLOBAL_positionBest) #informant_best_position taken out
             ##   self.swarmArray[x].changePosition(bounds)
            
            i+=1
            
        #-------- Search through swarm to get best ---------#
        '''
        final_best_fitness = -1
        final_best_position = []
        
        for x in self.swarmArray:
            if x.bestFitness < final_best_fitness or final_best_fitness == -1:
                final_best_fitness = x.bestFitness
                final_best_position = x.bestPosition
        '''
        
        # print final results
        print ('Results:')
        print ("Best final position: " + str(self.GLOBAL_positionBest))
        print ("Best final fitness: " + str(self.GLOBAL_fitnessBest))
