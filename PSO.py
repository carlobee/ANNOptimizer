"""
A program that implements a the Particle Swarm Optimization Algorithm

Written By Carl Bettosi and Mamta Sofat
17/11/2020

"""
import numpy as np

from particle import particle


class PSO():
    
    # construct
    def __init__(self, func, initial, bounds, swarmSize, iterations, numberOfInformants):
        
        #-------- Generate the swarm ---------#
        
        swarmArray = []     # create array to hold swarm particles

        # create x amount of particles and add to array
        for x in range(0, swarmSize):
            swarmArray.append(particle(x, initial, swarmSize, numberOfInformants))
        
        print(f"\nParticles with swarm size {swarmSize} generated.\n")
        
        
        
        
        #-------- Get methods for the particle informant info ---------#
        
        def get_informant_id(id):
            
            best_informant_fitness = -1
            best_informant_id = -1
            
            for x in range(0, numberOfInformants):
                check_id = swarmArray[id].informantGroup[x]
                
                
                #best_position = []
                
                if swarmArray[check_id].bestFitness < best_informant_fitness or best_informant_fitness == -1:
                    best_informant_fitness = swarmArray[check_id].bestFitness
                    best_informant_id = check_id
            
            return check_id
        
        def get_informant_best_fitness(id):
            return swarmArray[id].bestFitness
        
        def get_informant_best_position(id):
            return swarmArray[id].bestPosition
        
        
        
        
        #-------- Begin the optimisation ---------#
        i=0
        while i < iterations:
            
            informant_best_fitness = -1
            informant_best_position = []
            
            # for each particle in swarm, calculate fitness and update object's attributes to reflect
            for x in range(0, swarmSize):
                swarmArray[x].calculateFitness(func)
                
                # get informant bests for this particle's group and set them to particle's informant bests
                best_id = get_informant_id(x)
                informant_best_fitness = get_informant_best_fitness(best_id)
                informant_best_position = get_informant_best_position(best_id)
                
                # check if this particle is the best in its informant group, if so then change
                if swarmArray[x].currentFitness < informant_best_fitness or informant_best_fitness == -1:
                    informant_best_fitness = swarmArray[x].currentFitness
                 
            for x in range(0, swarmSize):
                swarmArray[x].changeVelocity(informant_best_position)
                swarmArray[x].changePosition(bounds)
            
            i+=1
            
        #-------- Search through swarm to get best ---------#
        
        final_best_fitness = -1
        final_best_position = []
        
        for x in swarmArray:
            if x.bestFitness < final_best_fitness or final_best_fitness == -1:
                final_best_fitness = x.bestFitness
                final_best_position = x.bestPosition
        
        # print final results
        print ('Results:')
        print ("Best position: " + str(final_best_position))
        print ("Best fitness: " + str(final_best_fitness))