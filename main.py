"""
A program that implements a basic feedforward multilayer Artificial Neural Network

Written By Carl Bettosi and Mamta Sofat
16/11/2020

"""

import numpy as np
import random

from PSO import PSO


#--- GENERATE ANN ----------------------------------------------------------------------  

numberOfInputs = 2

# get some properties from user
numberOfNeurons = int(input("How many neurons? "))

# give restraints to number of neurons that can be generated
if(numberOfNeurons >= (2*numberOfInputs) or numberOfNeurons < 1):
    print("Error: Number of neurons must be between 1 and twice the number of inputs.")
    numberOfNeurons = int(input("How many neurons? "))

numberOfLayers = int(input("How many hidden layers? "))

print(f"\nGenerating ANN with {numberOfLayers} layers and {numberOfNeurons} neurons...\n")

# dictionary to hold layers
HiddenLayers = {};

# create weights for input > first hidden layer
arr = np.random.random((numberOfInputs, numberOfNeurons))
HiddenLayers[0] = arr
print(arr)

for x in range(numberOfLayers-1):
    print("----------------------------------------")
    arr = np.random.random((numberOfNeurons, numberOfNeurons))
    HiddenLayers[x] = arr
    print(arr)
    
#--- ACTIVATION FUNCTIONS --------------------------------------------------------------    

#reference this https://medium.com/towards-artificial-intelligence/building-neural-networks-with-python-code-and-math-in-detail-ii-bbe8accbf3d1
def sigmoid(x):
    return 1/(1+np.exp(-x))

def hyperbolicTangent(x):
    return np.tanh(x)
    
def cosine(x):
    return np.cos(x)
    
#def Gaussian(x):
    #return exp    

#--- PARTICLE SWARM OPTIMIZATION -------------------------------------------------------

# function we are attempting to optimize (minimize)
def func(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

initial=[5,5]               # initial starting location [x1,x2...]
bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
swarmSize = 10
iterations = 50

# call PSO with hyperparams
PSO(func, initial, bounds, swarmSize, iterations)
