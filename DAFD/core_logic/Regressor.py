from DAFD.models.forward_models.NeuralNetModel_rate1 import NeuralNetModel_rate1
from DAFD.models.forward_models.NeuralNetModel_rate2 import NeuralNetModel_rate2
from DAFD.models.forward_models.NeuralNetModel_size1 import NeuralNetModel_size1
from DAFD.models.forward_models.NeuralNetModel_size2 import NeuralNetModel_size2
from DAFD.helper_scripts.ModelHelper import ModelHelper
import numpy as np
import sklearn.metrics

load_model = True	# Load the file from disk

class Regressor:
	"""
	Small adapter class that handles training and usage of the underlying regression models
	"""

	regression_model = None


	def __init__(self, output_name, regime):
		self.MH = ModelHelper.get_instance() # type: ModelHelper
		self.regime = regime

		regime_indices = self.MH.regime_indices[regime]
		regime_feature_data = [self.MH.train_features_dat_regnorm[x] for x in regime_indices]
		regime_label_data = [self.MH.train_labels_dat[output_name][x] for x in regime_indices]

		print("Regression model " + output_name + str(regime))
		if output_name == "generation_rate":
			if regime == 1:
				self.regression_model = NeuralNetModel_rate1()
			elif regime == 2:
				self.regression_model = NeuralNetModel_rate2()
		elif output_name == "droplet_size":
			if regime == 1:
				self.regression_model = NeuralNetModel_size1()
			elif regime == 2:
				self.regression_model = NeuralNetModel_size2()

		if load_model:
			print("Loading Regressor")
			self.regression_model.load_model(output_name, regime)
		else:
			print("Training Regressor")
			print("All data points: " + str(len(self.MH.train_features_dat_regnorm)))
			print("Train points: " + str(len(regime_indices)))
			self.regression_model.train_model(output_name, regime, regime_feature_data, regime_label_data)

		train_features = np.stack(regime_feature_data)
		train_labels = np.stack(regime_label_data)
		#print(",".join(self.MH.input_headers) + ",label,prediction")
		#for i,label in enumerate(list(train_labels)):
		#	print(",".join(list(map(str,list(train_features[i])))) + "," + str(label) + "," + str(self.regression_model.regression_model.predict(train_features)[i][0]))
		#print(train_labels)
		#print()
		#print(train_features)
		print("R square (R^2) for Train:                 %f" % sklearn.metrics.r2_score(train_labels, self.regression_model.regression_model.predict(train_features)))
		print()


	def predict(self,features):
		# We expect the features to be a whole data normalized set, so denormalize and then normalize with respect to regime
		features = self.MH.denormalize_set(features)
		features = self.MH.normalize_set(features,"_regime"+str(self.regime))
		return self.regression_model.predict(features)
