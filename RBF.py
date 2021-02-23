import numpy as np
import math

class RBF_Neuron():

    def __init__(self, center = np.full(1, 0), radius = 0.5, constBias = 0.02, weight = 1.0, trainRate = 1.0):
        self.c = center
        self.r = radius
        self.tR = trainRate
        #We store the last inputs and activations in order for batch learning to be possible. If backProp is called, these lists will be consumed and reset for the next batch
        self.lastInput = 0
        self.lastActivation = 0
        self.s = 0
    
    def activate(self, x):
        x = x.astype(np.float)
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension"
        self.lastInput = x
        #Calculate ||x-c||^2
        self.s = x - self.c
        self.s = np.dot(s, s)
        #calculate exp(-s*r)
        self.lastActivation = np.exp(-self.s[-1] * self.r)
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
