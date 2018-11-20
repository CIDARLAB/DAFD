from sklearn.neural_network import MLPRegressor
import numpy as np

class NeuralNetModel:

	regression_model = None

	def __init__(self, features, labels):
		self.regression_model =  MLPRegressor(solver='lbfgs', tol=0.00001, hidden_layer_sizes=(100,100,100,100))
		self.regression_model.fit(features, labels)

	def predict(self, features):
		return self.regression_model.predict(np.asarray(features).reshape(1, -1))[0]

