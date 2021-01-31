import numpy as np
import math
from rbfConvolution import RBF_Convolution as convolution
from rbfLabel import RBF_Label_Layer as labelLayer
from IPython.display import clear_output

class Model():

    def __init__(self, imageWidth, imageHeight, numLabels, numKernels, trainRate, debug = False):
        self.con = convolution(imageWidth = imageWidth, imageHeight = imageHeight, numFilters = numKernels, trainRate = trainRate)
        self.labels = labelLayer(numLabels = numLabels, dims = ((imageHeight - 2) * (imageWidth - 2) * numKernels), trainRate = trainRate)
        self.debug = debug
        print("Init model")
        
    def trainModel(self, trainingImages, labels):
        for i in range(trainingImages.shape[0]):
            print("Training..." + str(i))
            self.forwardPass(trainingImages[i, :, :])
            self.backProp(trainingImages[i,:,:], int(labels[i]))
        print("Finished training.")
        
    def classify(self, inputImage):
        print("Processing...")
        self.forwardPass(inputImage)
        print("Done!")
        maxIndex = np.argmax(np.asarray(self.prob))
        print("Label: " + str(maxIndex))
        print("Probability: " + self.prob[maxIndex])
        return maxIndex            
        
    def forwardPass(self, input):
        print("ForwardPass model")
        self.filteredImage = self.con.forwardPass(input)
        self.vals = self.labels.forwardPass(self.filteredImage)
        self.s = sum(self.vals)
        self.prob = []
        for i in range( len(self.vals) ):
            self.prob.append(self.vals[i] / self.s)
        print("Done forwardPass model")
        
    def backProp(self, input, correctLabel):
        print("BackProp model")
        print("Error: " + str(- math.log(self.prob[correctLabel])))
        lDiv = self.labels.backProp(input = self.filteredImage.flatten(), derivative = -(1 / self.prob[correctLabel]), winnerIndex = correctLabel)
        self.con.backProp(input = input, derivative = lDiv)
        print("Done backProp model")
        
        
        
        