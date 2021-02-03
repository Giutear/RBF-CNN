import numpy as np
import math

class RBF_Neuron():

    def __init__(self, center = np.full(1, 0), radius = 0.5, constBias = 0.02, weight = 1.0, trainRate = 1.0):
        self.c = center
        self.r = radius
        self.cB = constBias
        self.w = weight
        self.tR = trainRate
    
    def activate(self, x):
        x = x.astype(np.float)
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension"
        self.s = np.double(0)
        self.lastActivation = np.double(1)
        for i in range(x.shape[0]):
            b = (x[i] - self.c[i]) * (x[i] - self.c[i])
            self.s += b
            self.lastActivation *= math.exp( (-b) / (2 * (self.r * self.r)) )
        if self.lastActivation < math.exp(-self.s/(2 * self.r * self.r)):
            self.lastActivation =math.exp(-self.s/(2 * self.r * self.r))
        self.lastActivation = self.w * self.lastActivation - self.cB
        return self.lastActivation
        
    def backProp(self, x, derivative = 1.0):
        '''
        x should be the last input that was used to activate the neuron
        derivative should be the derivative of the previous layer
        '''
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension for backProp"
         #Calculate the error for the next layer
        dev = []
        for i in range(self.c.shape[0]):
            dev.append(-derivative * (self.lastActivation + self.cB) * (x[i] - self.c[i]) / (self.r * self.r)) 
        #Adapt the center positions
        for i in range(self.c.shape[0]):
            self.c[i] -= self.tR * derivative * (self.lastActivation + self.cB) * (self.c[i] - x[i]) / (self.r * self.r)
        #Adapt the neurons weight
        self.w -= self.tR * derivative * (self.lastActivation + self.cB) / self.w
        #adapt the radius
        self.r -= self.tR * derivative * (self.lastActivation + self.cB) * (self.s / (self.r * self.r * self.r))
        #adapt bias
        self.cB += self.tR * derivative       
        return dev

        
class RBF_Kernel(RBF_Neuron):

    def backProp(self, x, derivative = 1.0):
        '''
        image should be the image that was originaly filtered.
        derivative should be the derivative of the previous layer
        The reason this function is overwritten is becase for the kernel, the center positions don't need to be adapted
        '''
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension for backProp"
        ret = -derivative * (self.lastActivation + self.cB) * (x[0] - self.c[0]) / (self.r * self.r)
        #Adapt the neurons weight
        self.w -= self.tR * derivative * (self.lastActivation + self.cB) / self.w
        #adapt the radius
        self.r -= self.tR * derivative * (self.lastActivation + self.cB) * (self.s / (self.r * self.r * self.r))
        #Adapt the constant bias
        self.cB += self.tR * derivative
        #Return the error for the next layer
        return ret