import numpy as np
import pandas as pd
import math
from PIL import Image
from matplotlib import pyplot as plt


class ConvolutionLayer():

    def __init__(self, numFilters, numLabels, stride = 1, width = 0, height = 0, channels = 0, debug = False):
        self.debug = debug
        #Here the RBFs which will label the input are stored
        self.labelRBFs = []
        for i in range(0, numLabels):
            self.labelRBFs.append( RFB_Kernel(center = np.zeros( int( (width - 3) / stride + 1 ) * int( (height - 3) / stride + 1 ) * channels * numFilters )) )
        #Here the RBFs which filter the image are stored
        self.RBFFilters = []
        for i in range(0, numFilters):
            self.RBFFilters.append(RFB_Kernel())
        #The stride defines how far a filter will move between calculations
        self.stride = stride
        self.width = width
        self.height = height
        self.channels = channels       
        
    def forwardPass(self, inputImage):
        '''
        During a forward pass, the filters will run over the input image and produce their outputs
        '''
        filteredImage = np.ones(( len(self.RBFFilters),  int(((inputImage.shape[0] - 3) / self.stride) + 1), int(((inputImage.shape[1] - 3) / self.stride) + 1), inputImage.shape[2] ))
        # Iterate over all filters, colours and pixels in tat order
        for k in range(filteredImage.shape[0]):
            for i in range(filteredImage.shape[1]):
                for j in range(filteredImage.shape[2]): 
                    for c in range(filteredImage.shape[3]):
                        #Set the center of the rbf to the colour of the central pixel
                        self.RBFFilters[k].c = np.full(1, inputImage[self.stride * i + 1][self.stride * j + 1][c])
                        #Calculate the filtered colour
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i][self.stride * j][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i][self.stride * j + 1][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i][self.stride * j + 2][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j + 1][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i + 1][self.stride * j + 2][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j + 1][c]))
                        filteredImage[k][i][j][c] -= self.RBFFilters[k].activate(np.full(1, inputImage[self.stride * i + 2][self.stride * j + 2][c]))
                        #Set the maximum and minimum in the range of [0, 1]
                        if(filteredImage[k][i][j][c] < 0.0):
                            filteredImage[k][i][j][c] = 0.0
                        elif(filteredImage[k][i][j][c] > 1.0):
                            filteredImage[k][i][j][c] = 1.0
        #Flatten the image into a single vector  
        flattenedImage = filteredImage.flatten()              
        #After the convolution we do the classification using a fully connected layer of RBFs
        s = 0
        for i in range(len(self.labelRBFs)):
            s += self.labelRBFs[i].activate(flattenedImage)
        #Lastly we calculate the probability by dividing the label by the sum of the result for all labels
        probabilities = []
        for i in range(len(self.labelRBFs)):
            probabilities.append(self.labelRBFs[i].lastActivation / s)
        #Print some information if debugging was enabled
        if(self.debug):
            print("Filtered shape:")
            print(filteredImage.shape)
            for i in range(len(self.labelRBFs)):
                print( "Probability Label " +  str(i) + ": " + str(probabilities[i]) )
            fig = plt.figure()
            plt.imshow(Image.fromarray(inputImage))
            for i in range(filteredImage.shape[0]):
                fig = plt.figure()
                plt.imshow(Image.fromarray((filteredImage[i,:,:,:] * 255).astype(np.uint8)))
    
    def backProp(self):
        #TODO
        return 0
        
        
        
              
    def backProp(self):
        print("TBI backProp")
        self.poolingIndices = []
        
    def __maxPooling(self, inputImage):
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
        
    def __iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''
        for i in range(image.shape[0] - 2):
            for j in range(image.shape[1] - 2):
                for k in range(image.shape[2]):
                    im_region = image[i:(i + 3), j:(j + 3), k]
                    yield im_region, i, j, k
                    
        
class RFB_Kernel():
    def __init__(self, center = np.full(1, 0), radius = 0.5, constBias = 0.02, weight = 1.0):
        self.c = center
        self.r = radius
        self.cB = constBias
        self.w = weight
    
    def activate(self, x):
        assert x.shape == self.c.shape, "x" + str(x.shape) + " and center " + str(self.c.shape) + " do not have the same dimension"
        s = np.double(0)
        for i in range(x.shape[0]):
            s += np.double(x[i] - self.c[i]) * np.double(x[i] - self.c[i])
        self.lastActivation = self.w * np.double( math.exp( (-s) / (2 * (self.r * self.r)) ) ) - self.cB
        return self.lastActivation
        
        