from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RandomForestModel:

	regression_model = None

	def __init__(self, features, labels):
		self.regression_model = RandomForestRegressor(1000)
		self.regression_model.fit(features, labels)

	def predict(self, features):
		return self.regression_model.predict(np.asarray(features).reshape(1, -1))[0]
