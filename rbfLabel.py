import numpy as np
from RBF import RBF_Neuron as RBF

class RBF_Label_Layer():

    def __init__(self, numLabels, dims, trainRate):
        self.labelNeurons = []
        for i in range(numLabels):
            self.labelNeurons.append(RBF(center = np.asarray([np.random.uniform(low=0, high=1) for i in range(dims)]), weight= np.random.uniform(1, 2)
            , radius= np.random.uniform(1,2), trainRate = trainRate))
        
        
    def forwardPass(self, input):
        self.val = []
        for i in range(len(self.labelNeurons)):
            self.val.append((self.labelNeurons[i]).activate( (input) ))
        return self.val
        
        
    def backProp(self, input, derivative, winnerIndex):
        return self.labelNeurons[winnerIndex].backProp(input, derivative)