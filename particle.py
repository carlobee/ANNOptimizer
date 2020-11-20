import random

class particle:
    
    def __init__(self, initial):
        
        # properties of the particle itsself
        self.velocity = []
        self.position = initial
        self.best = []
        
        # assign random velocity to paricles
        for x in range(1,2):
            self.velocity.append(random.uniform(-1,1))
            
    def evaluate(self, func):
        