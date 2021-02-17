from DAFD.models.regime_models.NeuralNetModel_regime import NeuralNetModel_regime
from DAFD.helper_scripts.ModelHelper import ModelHelper
import sklearn.metrics
import numpy as np

load_model = True	# Load the file from disk

class RegimeClassifier:
	"""
	Small adapter class that handles regime prediction
	"""

	neuralnet = None

	def __init__(self):
		self.MH = ModelHelper.get_instance() # type: ModelHelper
		self.neuralnet = NeuralNetModel_regime()
		print("regime classifier")
		if load_model:
			print("Loading classifier")
			self.neuralnet.load_model()
		else:
			print("Training classifier")
			print("Data points: " + str(len(self.MH.train_features_dat_wholenorm)))
			self.neuralnet.train_model(self.MH.train_features_dat_wholenorm, self.MH.train_regime_dat)

		train_features = np.stack(self.MH.train_features_dat_wholenorm)
		train_labels = np.stack(self.MH.train_regime_dat)
		print("Train accuracy: " + str(sklearn.metrics.accuracy_score(train_labels-1,self.neuralnet.classifier_model.predict_classes(train_features))))
		print()

	def predict(self,features):
		return self.neuralnet.predict(features)

