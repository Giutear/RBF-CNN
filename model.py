import numpy as np
import math
from rbfConvolution import RBF_Convolution as convolution
from rbfLabel import RBF_Label_layer as labelLayer

class Model():

    def __init(self, imageWidth, imageHeight, numLabels, numKernels, trainRate):
        self.con = convolution(imageWidth = imageWidth, imageHeight = imageHeight, numFilters = numKernels, trainRate = trainRate)
        self.labels = labelLayer(numLabels = numLabels, dims = (imageHeight - 2) * (imageHeight - 2), trainRate = trainRate)
        print("Init model")
        
    def trainModel(self, trainingImages):
        
        
    def forwardPass(self):
        print("ForwardPass model")
        
    def backProp(self):
        print("BackProp model")