import numpy as np
import pandas as pd
import math
from PIL import Image
from matplotlib import pyplot as plt
from RBF import RBF_Neuron as RBF

class RBF_Label_Layer():

    def __init__(self, numLabels, dims, trainRate):
        self.labelNeurons = []
        for i in range(numLabels):
            self.labelNeurons.append(RBF(center = np.ones(dims), trainRate = trainRate))
        
        
    def forwardPass(self, input):
        self.val = []
        for i in range(len(self.labelNeurons)):
            self.val.append((self.labelNeurons[i]).activate( (input.flatten()) ))
        return self.val
        
        
    def backProp(self, input, derivative, winnerIndex):
        return self.labelNeurons[winnerIndex].backProp(input, derivative)