from sklearn.svm import SVR
import numpy as np

class SVRModel:

	regression_model = None

	def __init__(self, features, labels):
		self.regression_model = SVR(C=100000, epsilon=0.00001)
		self.regression_model.fit(features, labels)

	def predict(self, features):
		return self.regression_model.predict(np.asarray(features).reshape(1, -1))[0]

