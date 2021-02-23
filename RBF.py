import numpy as np
import math

class RBF_Neuron():

    def __init__(self, center = np.full(1, 0), radius = 0.5, trainRate = 1.0):
        self.c = center
        self.r = radius
        self.tR = trainRate
        #We store the last input, activation and sum to save processing time during the back propagation
        self.lastInput = 0
        self.lastActivation = 0
        self.s = 0
    
    def activate(self, x):
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension"
        self.lastInput = x
        #Calculate ||x-c||^2
        self.s = x - self.c
        self.s = np.dot(self.s, self.s)
        #calculate exp(-s*r)
        self.lastActivation = np.exp(-self.s * self.r)
        #Return the last activation
        return self.lastActivation
        
    def backProp(self, derivative = 1.0):
        '''
        derivative should be the derivative of the previous layer
        '''
        #calculate the delta for r
        dr = -self.tR * derivative * self.lastActivation * self.s
        #Calculate the delta for c and x
        dc = np.fill(self.c.shape, self.lastActivation)
        dx = np.fill(self.c.shape, self.lastActivation)
        for i in range(dc.shape[0]):
            dc[i] *= 2 * self.r * (self.lastInput[i] - self.c[i])
            dx[i] *= -2 * self.r * (self.lastInput[i] - self.c[i])
        #adapt the parameters
        self.r += dr
        self.c += dc
        return dx

        
class RBF_Kernel():

    def __init__(self, radius, weight, bias):
        self.r = radius
        self.weight = weight
        self.bias = bias

    def activate(self, x):
        assert x.shape == (3,3), "x does not have the right shape for a RBF_Kernel. x.shape: " + str(x.shape) + "."
        self.lastActivation = np.zeros(x.shape)
        self.s = np.zeros(x.shape)
        self.lastInput = x
        #Calculate ||x-c||^2
        self.s = x - x[1,1]
        self.s = self.s * self.s
        #calculate exp(-s*r)
        self.lastActivation = np.exp(self.s * (-self.r))
        #Return the last activation
        return self.lastActivation