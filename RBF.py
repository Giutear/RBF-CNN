import numpy as np

class RBF_Neuron():
    def __init__(self, center = np.full(1, 0), radius = 0.5, constBias = 0.02, weight = 1.0, trainRate = 1.0):
        self.c = center
        self.r = radius
        self.cB = constBias
        self.w = weight
        self.tR = trainRate
    
    def activate(self, x):
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension"
        s = np.double(0)
        for i in range(x.shape[0]):
            s += np.double(x[i] - self.c[i]) * np.double(x[i] - self.c[i])
        self.lastActivation = self.w * np.double( math.exp( (-s) / (2 * (self.r * self.r)) ) ) - self.cB
        return self.lastActivation
        
    def backProp(self, derivative):
        print("Todo backProp RBF_Neuron")
        
class RBF_Kernel(RBF_Neuron):
    def backProp(self, derivative):
        print("Todo backProp RBF_Kernel")