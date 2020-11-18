"""
A program that implements a the Particle Swarm Optimization Algorithm

Written By Carl Bettosi and Mamta Sofat
17/11/2020

"""
import particle
import numpy as np

numberOfParticles = int(input("How many particles? "))

particleArray = np.array()

for x in numberOfParticles:
    newParticle = particle(x)
    np.append()