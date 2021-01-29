import numpy as np
import math
from rbfConvolution import RBF_Convolution as convolution
from rbfLabel import RBF_Label_layer as labelLayer

class Model():

    def __init(self, imageWidth, imageHeight, numLabels, numKernels, trainRate, debug = False):
        self.con = convolution(imageWidth = imageWidth, imageHeight = imageHeight, numFilters = numKernels, trainRate = trainRate)
        self.labels = labelLayer(numLabels = numLabels, dims = (imageHeight - 2) * (imageHeight - 2), trainRate = trainRate)
        self.debug = debug
        print("Init model")
        
    def trainModel(self, trainingImages):
        print("Training...")
        for i in range(trainImages.shape[0]):
            self.forwardPass(trainImages[i, :, :, :])
            self.backProp(trainImages[i,:,:,:])
        print("Finished training.")
        
    def classify(self, inputImage):
        print("Processing...")
        self.forwardPass(inputImage)
        print("Done!")
            
        
    def forwardPass(self, input):
        print("ForwardPass model")
        vals = self.labels.forwardPass(self.con.forwardPass(input))
        self.s = sum(vals)
        self.prob = []
        for i in range( len(vals) ):
            self.prob.append(vals[i] / s)
        print("Done forwardPass model")
        
    def backProp(self):
        print("BackProp model")