import numpy as np
import pandas as pd
import math
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
                    self.RBFFilters[k][i].append(RBF(trainRate = trainRate))
        #The stride defines how far a filter will move between calculations
        self.stride = stride
        self.tR = trainRate

        
    def forwardPass(self, inputImage):
        '''
        During a forward pass, the filters will run over the input image and produce their outputs
        inputImage should be a numpy array
        Returns a vector which represents the images position in feature space
        '''
        self.filteredImage = np.ones(( len(self.RBFFilters),  len(self.RBFFilters[0]), len(self.RBFFilters[0][0]), inputImage.shape[2] ))
        #Iterate over all filters, pixels and colours in that order
        for k in range(self.filteredImage.shape[0]):
            for i in range(self.filteredImage.shape[1]):
                for j in range(self.filteredImage.shape[2]): 
                    for c in range(self.filteredImage.shape[3]):
                        #Set the center of the RBF to the colour of the central pixel
                        self.RBFFilters[k][i][j].c = np.full(1, inputImage[self.stride * i + 1][self.stride * j + 1][c])
                        #Calculate the filtered colour
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i][self.stride * j][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i][self.stride * j + 1][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i][self.stride * j + 2][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j + 1][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j + 2][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j + 1][c]))
                        self.filteredImage[k][i][j][c] -= self.RBFFilters[k][i][j].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j + 2][c]))
        #Print some information if debugging was enabled
        if(self.debug):
            fig = plt.figure()
            plt.imshow(Image.fromarray(inputImage))
            #Normalize the image for displaying
            normImage = np.clip(self.filteredImage, 0.0, 1.0)
            for i in range(normImage.shape[0]):
                fig = plt.figure()
                plt.imshow(Image.fromarray((normImage[i,:,:,:] * 255).astype(np.uint8)))
        return self.filteredImage
    
    def backProp(self, input, derivative = 1):
        '''
        The derivative should be the value of the derivative from the previous layer
        The function will return it's own derivative after adpating it's parameters
        For now this function is only usable at the input layer which means no further back propagation is possible, hence no return value
        '''
        for k in range(self.filteredImage.shape[0]):
            for i in range(self.filteredImage.shape[1]):
                for j in range(self.filteredImage.shape[2]):
                    self.RBFFilters[k][i][j].backProp(input[i,j,0], -derivative)
        
    def __maxPooling(self, inputImage):
        '''
        An optional maxPooling function which I currently don't plan on using
        '''
        pooledImage = np.zeros((int(inputImage.shape[0] / 2), int(inputImage.shape[1] / 2), inputImage.shape[2], self.n_kernels), dtype = int)
        for i in range(pooledImage.shape[0]):
            for j in range(pooledImage.shape[1]):
                for k in range(pooledImage.shape[2]):
                    for l in range(pooledImage.shape[3]):
                        j0 = 2 * j
                        i0 = np.argmax(inputImage[2 * i : 2 * i + 1, 2 * j, k, l])
                        i1 = np.argmax(inputImage[2 * i : 2 * i + 1,2 * j + 1, k, l])
                        if inputImage[2 * i + i0][2 * j][k][l] < inputImage[2 * i + i1][2 * j + 1][k][l]:
                            i0 = i1
                            j0 += 1
                        self.poolingIndices.append((2 * i + i0, j0))
                        pooledImage[i][j][k][l] = inputImage[i0][j0][k][l]
        if self.debug:
            for i in range(0, self.n_kernels):
                fig = plt.figure()
                plt.imshow(pooledImage[0:,0:,0:,i])
                print("Pooled image shape:")
                print(pooledImage.shape)
        return pooledImage
       
                    
        