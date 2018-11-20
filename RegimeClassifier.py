from SVMModel import SVMModel
from ModelHelper import ModelHelper

class RegimeClassifier:

	svm = None

	def __init__(self):
		self.MH = ModelHelper.get_instance()
		self.svm = SVMModel(self.MH.train_features_dat, self.MH.train_regime_dat)


	def predict(self,features):
		return self.svm.predict(features)

