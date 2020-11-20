"""
A program that implements a the Particle Swarm Optimization Algorithm

Written By Carl Bettosi and Mamta Sofat
17/11/2020

"""
import numpy as np

from particle import particle

class PSO():
    
    # construct
    def __init__(self, func, initial, bounds, swarmSize, iterations):
        
        
        swarmArray = []     # create array to hold swarm particles
        bestFitness = -1    # best fintess for group
        bestPosition = []   # the informant group's best position
        
        # create x amount of particles and add to array
        for x in range(0, swarmSize):
            swarmArray.append(particle(initial))
        
        print(f"\nParticles with swarm size {swarmSize} generated.\n")
        
        # begin the optimization
        i=0
        while i < iterations:
            
            # for each particle in swarm, evaluate fitness
            for x in range(0, swarmSize):
                swarm[x].evaluate(func)