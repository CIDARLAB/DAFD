from sklearn.svm import SVC
import numpy as np

class SVMModel:


	classification_model = None

	def __init__(self, features, labels):
		self.classification_model = SVC(C=100000)
		self.classification_model.fit(features, labels)

	def predict(self, features):
		return self.classification_model.predict(np.asarray(features).reshape(1, -1))[0]