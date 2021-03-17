import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from RBF import RBF_Kernel as RBF


class RBF_Convolution():

    def __init__(self, numFilters, radius = 0.2, trainRate = 1.0, debug = False):
        '''
        Init function
        '''
        self.filters = []
        for i in range(numFilters):
            self.filters.append( (RBF(radius=radius, weight=1, bias=-0.1)) )
        self.tR = trainRate
        self.debug = debug

    def forwardPass(self, inputImage):
        '''
        During a forward pass, the filters will run over the input image and produce their outputs
        inputImage should be a numpy array
        Returns a vector which represents the images position in feature space
        '''
        self.lastInput = inputImage
        self.fImage = np.zeros( (len(self.filters), inputImage.shape - 2) )
        self.sImage = np.zeros( (len(self.filters), inputImage.shape - 2) )
        for reg, i, j in self.createRegions(inputImage, 1):
            for k in range(len(self.filters)):
                tImage = self.filters[k].activate(reg)
                self.sImage[k,i,j] = np.sum(tImage)
        self.fImage = 1 - self.filters[k].weight * self.sImage + self.filters[k].bias
        return self.fImage

    def backProp(self, derivative):
        '''
        The derivative should be the value of the derivative from the previous layer
        The function will return it's own derivative after adpating it's parameters
        For now this function is only usable at the input layer which means no further back propagation is possible, hence no return value
        '''
        db = 1
        dx = np.zeros(self.lastInput.shape)
        
    def createRegions(self, x, stride):
        '''
        This functions yields all 3x3 subimages of the argument x with given stride in between
        '''
        for i in range( (x.shape[0] - 3 + stride) / stride ):
            for j in range( (x.shape[1] - 3 + stride) / stride ):
                yield x[i:i+3,j:j+3], i, j