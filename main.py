import numpy as np
from perceptronlib import Perceptron

# Training data set
trainingdata = []
trainingdata.append(np.array([1,1]))
trainingdata.append(np.array([1,0]))
trainingdata.append(np.array([0,1]))
trainingdata.append(np.array([0,0]))

# AND GATE
targets = np.array([1,0,0,0])

# Variables
numberofinputs = 2

# SINGLE LAYER PERCEPTRON
perceptron = Perceptron(numberofinputs)
perceptron.train(trainingdata, targets, True)
#perceptron.train2(trainingdata, targets, True)

# Real Data Set
print ("Real Data Set")
input_1 = np.array([1,1])
print ("Input: ", input_1)
print (perceptron.predict(input_1))