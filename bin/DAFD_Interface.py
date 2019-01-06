""" Interface class for DAFD"""
from core_logic.ForwardModel import ForwardModel
from core_logic.InterModel import InterModel
from helper_scripts.ModelHelper import ModelHelper

import rpyc
from rpyc.utils.server import ThreadedServer
from threading import Thread

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
		results = self.it.interpolate(desired_vals,constraints)
		return results

	def runForward(self, features):
		# features is a dictionary containing the name of each feature as the key and the feature value as the value
		raw_features = [features[x] for x in self.input_headers]
		results = self.fw.predict(raw_features)
		return results

