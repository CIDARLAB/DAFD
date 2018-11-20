from SVRModel import SVRModel
from NearestDataPointModel import NearestDataPointModel
from RidgeRegressor import RidgeRegressor
from LassoRegressor import LassoRegressor
from RandomForestModel import RandomForestModel
from LinearModel import LinearModel
from NeuralNetModel import NeuralNetModel
from ModelHelper import ModelHelper


class Regressor:

	regression_model = None

	def __init__(self, output_name, regime):
		self.MH = ModelHelper.get_instance() # type: ModelHelper

		regime_indices = self.MH.regime_indices[regime]
		regime_feature_data = [self.MH.train_features_dat[x] for x in regime_indices]
		regime_label_data = [self.MH.train_labels_dat[output_name][x] for x in regime_indices]

		self.regression_model = RandomForestModel(regime_feature_data, regime_label_data)

	def predict(self,features):
		return self.regression_model.predict(features)
