import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RBF_Filter():

    def __init__(self, iWidth, iHeight, radius = 0.01, debug = False):
        self.debug = debug
        self.radius = float(radius)
        self.width = iWidth
        self.height = iHeight

    def __rbf(self, x, c):
        return np.exp( -np.dot((x - c), (x - c)) / self.radius )
    
    def firstForwad(self, image):
        '''
        In the first forward pass the filter is simply initialized as the image
        '''
        assert image.shape == (self.width, self.height), "Image does not have the right dimensions for filtering."
        if not hasattr(self, 'weights'):
            #Find the least squares such that Hx=b
            H = np.zeros([self.width * self.width, self.height * self.height])
            self.center = np.zeros([2, self.width, self.height])
            for i in range(self.width):
                for j in range(self.height):
                    self.center[0,i,j] = i
                    self.center[1,i,j] = j
            #Reshape the centers and result vector
            self.center = self.center.reshape(2, self.width * self.height)
            #Prepare the matrix H
            for i in range(self.width * self.width):
                for j in range(i, self.height * self.height):
                    H[i,j] = self.__rbf(self.center[:,i], self.center[:,j])
                    H[j,i] = H[i,j]
            self.weights = np.linalg.solve(H, image.flatten())
            self.weights = self.weights.reshape(self.width, self.height)
            self.filteredImage = np.zeros([self.width, self.height])
            for i in range(self.width):
                for j in range(self.height):
                    for w in range(self.width):
                        for v in range(self.height):
                            self.filteredImage[i,j] += self.weights[w,v] * self.__rbf(self.center[:,(i*j)], self.center[:,(w*v)])
            if self.debug:
                X,Y = np.meshgrid(self.center[1,:self.width], self.center[1,:self.height])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe (X,Y,Z=image)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe (X,Y,Z=self.filteredImage)
                fig = plt.figure()
                plt.imshow(image)
                fig = plt.figure()
                plt.imshow(self.filteredImage)
                plt.show()
        else:
            print("Weights has already been defined.")


    def forwardPass(self, image):
        print("TODO forwardPass")

    def backProp(self, derivative = 1.0):
        print("TODO backProp")