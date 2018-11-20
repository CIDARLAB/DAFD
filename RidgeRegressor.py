from sklearn import linear_model
import numpy as np

class RidgeRegressor:

	regressor_model = None

	def __init__(self, features, labels):
		self.regressor_model = linear_model.Ridge(alpha=0.5)
		self.regressor_model.fit(features,labels)

	def predict(self, features):
		return self.regressor_model.predict(np.asarray(features).reshape(1, -1))[0]
