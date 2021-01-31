import numpy as np
import matplotlib as ml
#The following lines are only meant to guarante that the files are up to date
import model
import RBF
import rbfLabel
import rbfConvolution

trainSet = np.loadtxt("mnist_train.csv", delimiter = ",")
labels = trainSet[:,0]
images = trainSet[:,1:]
trainImages = np.zeros((images.shape[0], 28, 28))                     
for i in range(images.shape[0]):
    trainImages[i] = images[i].reshape((28,28))
    

m = model.Model(imageWidth = 28, imageHeight = 28, numLabels = 10, numKernels = 4, trainRate = 1.0, debug = True)
m.trainModel(trainImages, labels)