import numpy as np

class Perceptron(object):
	def __init__(self, ninput, epochs = 100, learning_rate=0.01, random_weights = True):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.ninput = ninput
		if (not random_weights):
			self.weights = np.zeros(ninput + 1) # + 1 for the bias at nth value [0., 0., ... , 0ninput]
		else:
			self.weights = np.random.rand(ninput + 1)

	def predict(self, inputs):
		# Same as activationfunction
		if np.dot(inputs, self.weights[1:]) + self.weights[0] > 0:
			return 1
		else:
			return 0


	def activationfunction(self, inputs):
		dotproduct = 0
		for i in range(self.ninput):
			dotproduct += inputs[i] * self.weights[i + 1] # 0th is for bias which is added after this line
		dotproduct += self.weights[0]
		# --- Same as above
		# dotproduct = sum([i*j for (i, j) in zip(inputs, self.weights[1:])])
		# dotproduct += self.weights[0]

		if dotproduct > 0:
			return 1
		else:
			return 0

	def train(self, trainingdata, targets, verbose = False):
		# Loop through epochs
		for epoch in range(self.epochs):
			for input, target in zip(trainingdata, targets):
				prediction = self.activationfunction(input)
				# Note: Perceptron stops adjusting once the values are all correct
				# Prediction returns 0 ot 1, therefore, this is a sub-gradient stochastic gradient descent (SSGD)
				self.weights[1:] += self.learning_rate * (target - prediction) * input
				self.weights[0] += self.learning_rate * (target - prediction)
				
			if (verbose and epoch % 10 == 0):
				print ("Epoch #", epoch, "; Bias: ", self.weights[0], "; Weights:", self.weights[1:])	

	# ----- Same code as above but done using a library ---------
	# activationfunction2 - takes an array of inputs to calculate output
	def activationfunction2(self, inputs):
		return np.where(np.dot(inputs, self.weights[1:]) + self.weights[0] > 0, 1, 0)

	# train2 - implicit numpy library call for training procedure (same result as train) 
	def train2(self, trainingdata, targets, verbose = False):
		trainingdata = np.array(trainingdata)
		for epoch in range(self.epochs): 
			predictions = self.activationfunction2(trainingdata)
			errors = (targets - predictions)
			self.weights[1:] += self.learning_rate * trainingdata.T.dot(errors)
			self.weights[0] += self.learning_rate * errors.sum()

			if (verbose and epoch % 10 == 0):
				print ("Epoch #", epoch, "; Bias: ", self.weights[0], "; Weights:", self.weights[1:])	