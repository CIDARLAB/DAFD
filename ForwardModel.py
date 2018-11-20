from ModelHelper import ModelHelper
from RegimeClassifier import RegimeClassifier
from Regressor import Regressor


class ForwardModel:

	regime_classifier = None
	regressor = None

	def __init__(self):
		self.MH = ModelHelper.get_instance() # type: ModelHelper
		self.regime_classifier = RegimeClassifier()
		self.model_dict = {}
		for regime in self.MH.regime_indices:
			for header in self.MH.output_headers:
				self.model_dict[header+str(regime)] = Regressor(header, regime)


	def predict(self, features):
		ret_dict = {}
		norm_features = self.MH.normalize_set(features)
		regime = self.regime_classifier.predict(norm_features)
		ret_dict["regime"] = regime
		for header in self.MH.output_headers:
			ret_dict[header] = self.model_dict[header+str(regime)].predict(norm_features)
		return ret_dict




