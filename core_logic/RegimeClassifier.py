from models.regime_models.SVMModel import SVMModel
from helper_scripts.ModelHelper import ModelHelper

class RegimeClassifier:
	"""
	Small adapter class that handles regime prediction
	For now, we only use SVMs, but this may change in the future.
	"""

	svm = None

	def __init__(self):
		self.MH = ModelHelper.get_instance()
		self.svm = SVMModel(self.MH.train_features_dat, self.MH.train_regime_dat)


	def predict(self,features):
		return self.svm.predict(features)

