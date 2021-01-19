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
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension"
        self.s = np.double(0)
        for i in range(x.shape[0]):
            self.s += np.double(x[i] - self.c[i]) * np.double(x[i] - self.c[i])
        self.lastActivation = self.w * np.double( math.exp( (-self.s) / (2 * (self.r * self.r)) ) ) - self.cB
        return self.lastActivation
        
    def backProp(self, x, derivative = 1.0):
        '''
        x should be the last input that was used to activate the neuron
        derivative should be the derivative of the previous layer
        '''
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension for backProp"
        #Adapt the center positions
        for i in range(self.c.shape[0]):
            self.c[i] += self.tR * derivative * (self.lastActivation + self.cB) * (self.c[i] - x[i]) / (self.r * self.r)
        #Adapt the constant bias
        self.cB -= self.tR * derivative
        #Adapt the neurons weight
        self.w += self.tR * derivative * (self.lastActivation + self.cB) / self.w
        
class RBF_Kernel(RBF_Neuron):

    def backProp(self, derivative):
        print("Todo backProp RBF_Kernel")