import numpy as np

class Perceptron(object):
	def __init__(self, ninput, epochs = 40, learning_rate=0.01, random_weights = True):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.ninput = ninput
		if (random_weights):
			self.weights = np.zeros(ninput + 1)
		else:
			self.weights = np.random.rand(ninput + 1)

	def predict(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		if summation > 0:
			activation = 1
		else:
			activation = 0
		return activation

	def train(self, trainingdata, targets, verbose = False):
		# Loop through epochs
		for epoch in range(self.epochs):
			for inputs, target in zip(trainingdata, targets):
				prediction = self.predict(inputs)
				# Note: Perceptron stops adjusting once the values are all correct (No gradient descent)
				self.weights[1:] += self.learning_rate * (target - prediction) * inputs
				self.weights[0] += self.learning_rate * (target - prediction)
			if (verbose):
				print ("Epoch #", epoch, "; Bias: ", self.weights[0], "; Weights:", self.weights[1:])	