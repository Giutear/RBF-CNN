import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from RBF import RBF_Kernel as RBF


class RBF_Convolution():

    def __init__(self, imageWidth, imageHeight, numFilters, stride = 1, trainRate = 1.0, debug = False):
        self.debug = debug
        #Here the RBFs which filter the image are stored
        filterWidth = int(((imageWidth - 3) / stride) + 1)
        filterHeight = int(((imageHeight - 3) / stride) + 1)
        self.RBFFilters = []
        for k in range(numFilters):
            self.RBFFilters.append([])
            for i in range(filterWidth):
                self.RBFFilters[k].append([])
                for j in range(filterHeight):
                    self.RBFFilters[k][i].append(RBF(trainRate = trainRate, constBias = np.random.uniform(low= 0.01, high=0.1), weight= np.random.uniform(low= 0.01, high=0.9)))
        #The stride defines how far a filter will move between calculations
        self.stride = stride
        self.tR = trainRate
        self.currentCycle = 0

        
    def forwardPass(self, inputImage):
        '''
        During a forward pass, the filters will run over the input image and produce their outputs
        inputImage should be a numpy array
        Returns a vector which represents the images position in feature space
        '''
        self.filteredImage = np.ones(( len(self.RBFFilters),  len(self.RBFFilters[0]), len(self.RBFFilters[0][0]) ))
        #Iterate over all filters, pixels and colours in that order
        for k in range(self.filteredImage.shape[0]):
            for i in range(self.filteredImage.shape[1]):
                for j in range(self.filteredImage.shape[2]):
                    #Set the center of the RBF to the colour of the central pixel
                    self.RBFFilters[k][i][j].c = np.full(1, inputImage[self.stride * i + 1][self.stride * j + 1])
                    #Calculate the filtered colour
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i][self.stride * j]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i][self.stride * j + 1]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i][self.stride * j + 2]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j + 1]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j + 2]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j + 1]))
                    self.filteredImage[k][i][j] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j + 2]))

        self.filteredImage = np.clip(self.filteredImage, 0.0, 1.0)
        #Print some information if debugging was enabled
        if(self.debug):
            fig = plt.figure()
            plt.imshow(Image.fromarray(inputImage))
            #Normalize the image for displaying
            normImage = np.clip(self.filteredImage, 0.0, 1.0)
            for i in range(normImage.shape[0]):
                fig = plt.figure()
                plt.imshow(Image.fromarray((normImage[i,:,:] * 255).astype(np.uint8)))
            plt.show()
        return self.filteredImage
    
    def backProp(self, input, derivative):
        '''
        The derivative should be the value of the derivative from the previous layer
        The function will return it's own derivative after adpating it's parameters
        For now this function is only usable at the input layer which means no further back propagation is possible, hence no return value
        '''
        if(self.currentCycle % 50 == 0):
            for i in range(self.filteredImage.shape[0]):
                fig = plt.figure()
                plt.imshow(Image.fromarray((self.filteredImage[i,:,:] * 255).astype(np.uint8)))
            plt.show()
        for k in range(self.filteredImage.shape[0]):
            for i in range(self.filteredImage.shape[1]):
                for j in range(self.filteredImage.shape[2]):
                    self.RBFFilters[k][i][j].backProp(np.full(1, input[i,j]), -derivative[k][i][j])
        self.currentCycle += 1
