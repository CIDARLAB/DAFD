""" Interface class for DAFD"""
from core_logic.ForwardModel import ForwardModel
from core_logic.InterModel import InterModel
from helper_scripts.ModelHelper import ModelHelper


class DAFD_Interface:
	"""A class that provides an interface for DAFD"""

	def __init__(self):
		self.it = InterModel()
		self.fw = self.it.fwd_model

		self.MH = ModelHelper.get_instance() # type: ModelHelper

		self.ranges_dict = self.MH.ranges_dict
		self.input_headers = self.MH.input_headers
		self.output_headers = self.MH.output_headers

	def runInterp(self, desired_vals, constraints):
		""" Run the design automation tool"""
		results = self.it.interpolate(desired_vals,constraints)
		return results

	def runForward(self, features):
		""" Run the forward model
				- features is a dictionary containing the name of each feature as the key and the feature value
					(not normalized) as the value
		"""
		raw_features = [features[x] for x in self.input_headers] # Order the features correctly
		results = self.fw.predict(raw_features)
		design_params = {}
		for feature in features:
			design_params[feature] = features[feature]
		for result in results:
			design_params[result] = results[result]
		oil_rate, water_rate, inferred_droplet_size = self.MH.calculate_formulaic_relations(design_params)
		results["oil_rate"] = oil_rate
		results["water_rate"] = water_rate
		results["inferred_droplet_size"] = inferred_droplet_size
		return results

