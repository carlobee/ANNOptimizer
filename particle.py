import random

class particle:
    
    def __init__(self, id, initial, swarmSize, numberOfInformants):
        
        # properties of the particle itsself
        self.id = id # the unique id of the particle used for informants
        self.velocity = []
        self.currentPosition = initial
        self.bestPosition = []
        self.currentFitness = -1
        self.bestFitness = -1
        
        self.informantGroup = [] # the group of x informants that is related to this group
        
        #-------- Generate some random initial values ---------#
        
        # assign random velocity to paricles
        for x in range(0,2):
            self.velocity.append(random.uniform(-1,1))
            
        # assign random particles to this particle's informant group
        for x  in range(0, numberOfInformants):
            self.informantGroup.append(random.randint(0, swarmSize-1))
            
   
   #-------- Functions of the particle ---------#     
         
    # evaluate the fitness of the current position
    def calculateFitness(self, func):
        self.currentFitness = func(self.currentPosition) # call the cost function with position param
        
        # evaluate if current fitness is better than recorded best, if so then update
        if self.bestFitness == -1 or self.currentFitness < self.bestFitness:
            self.bestFitness = self.currentFitness
            self.bestPosition = self.currentPosition
     
    # change the velocity of the particle
    def changeVelocity(self, informant_best_position, global_best_position):
        weight = 0.5
        cog_constant = 1
        soc_constant = 1
        inf_constant = 1
        
        random1 = random.random()
        random2 = random.random()
        random3 = random.random()
        
        for x in range(0,2):
            # adapted from https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
            velocityCog = cog_constant*random1*(global_best_position[x]-self.currentPosition[x])
            velocitySoc = soc_constant*random2*(global_best_position[x]-self.currentPosition[x])
            velocityInformant = inf_constant*random3*(informant_best_position[x]-self.currentPosition[x])
            self.velocity[x] = weight*self.velocity[x]+velocityCog+velocitySoc+velocityInformant
            
    # change to the new position after the velocities have been applied to the particles
    def changePosition(self, bounds):
        for x in range(0,2):
            self.currentPosition[x]=self.currentPosition[x]+self.velocity[x]
            
            # adjust maximum position if necessary
            if self.currentPosition[x]>bounds[x][1]:
                self.currentPosition[x]=bounds[x][1]

            # adjust minimum position if neseccary
            if self.currentPosition[x] < bounds[x][0]:
                self.currentPosition[x]=bounds[x][0]