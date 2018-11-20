import os
import csv
import sys


def resource_path(relative_path):
	""" Get absolute path to resource, works for dev and for PyInstaller """
	try:
		# PyInstaller creates a temp folder and stores path in _MEIPASS
		base_path = sys._MEIPASS
	except Exception:
		base_path = os.path.abspath(".")

	return os.path.join(base_path, relative_path)






class ModelHelper:

	RESOURCE_PATH = "ExperimentalResultsRegime_chipNum.csv"		# Experimental data location
	NUM_OUTPUTS = 2												# Droplet Generation Rate + Droplet Size

	instance = None				# Singleton
	input_headers = []			# Feature names (orifice size, aspect ratio, etc...)
	output_headers = []			# Output names (droplet size, generation rate)
	all_dat = []				# Raw experimental data
	train_features_dat = []		# Normalized and reduced features from all_dat
	train_labels_dat = {}		# Normalized and reduced labels from all_dat
	train_regime_dat = []		# Regime labels from all_dat
	regime_indices = {}			# Indices of the TRAIN DATA (not the all_dat) that belong to each regime
	ranges_dict = {}			# Dictionary of ranges for each ranges variable

	def __init__(self):
		if ModelHelper.instance is None:
			self.get_data()
			self.make_train_data()
			ModelHelper.instance = self

	@staticmethod
	def get_instance():
		if ModelHelper.instance is None:
			ModelHelper()
		return ModelHelper.instance




	def get_data(self):
		""" Read the data from the CSV list """

		values_dict = {}  # Temporary variable used for calculating ranges. Dict with input header as key and a list of all values of that header as values

		with open(resource_path(self.RESOURCE_PATH)) as f:
			# Make a list of lists of our csv data
			lines = csv.reader(f, delimiter=',')

			# Save header info
			headers = next(lines)
			self.input_headers = [x for x in headers[:-self.NUM_OUTPUTS] if
								  x != "regime" and x != "chip_num"]  # Regime and chip number isn't a feature we want to train our regressor on
			self.output_headers = headers[-self.NUM_OUTPUTS:]

			# Init values dict
			for head in headers:
				values_dict[head] = []

			# Get save the info of each row to var_dat and values_dict
			for row in lines:
				self.all_dat.append({headers[x]: float(row[x]) for x in range(len(headers))})
				for head_i in range(len(headers)):
					values_dict[headers[head_i]].append(float(row[head_i]))

		# Find the min and max to each data type
		for head in headers:
			self.ranges_dict[head] = (min(values_dict[head]), max(values_dict[head]))

	def normalize_set(self, values):
		""" Normalizes a set of features
		Args:
			values: list of features to be normalized (same order as input_headers)

		Returns list of normalized features in the same order as input_headers
		"""
		ret_list = []
		for i, header in enumerate(self.input_headers):
			ret_list.append(self.normalize(values[i], header))

		return ret_list


	def normalize(self, value, inType):
		"""Return min max normalization of a variable
		Args:
			value: Value to be normalized
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized

		Returns 0-1 normalization of value with 0 being the min and 1 being the max
		"""
		return (value - self.ranges_dict[inType][0]) / (self.ranges_dict[inType][1] - self.ranges_dict[inType][0])


	def denormalize(self, value, inType):
		"""Return actual of a value of a normalized variable
		Args:
			value: Value to be corrected
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized

		Returns actual value of given 0-1 normalized value
		"""
		return value * (self.ranges_dict[inType][1] - self.ranges_dict[inType][0]) + self.ranges_dict[inType][0]

	def make_train_data(self, indices=-1):
		""" Make training data from test data
			Indices are a list of wanted data (-1 indicates use all data)
		"""

		if indices == -1:
			indices = [x for x in range(len(self.all_dat))]

		self.train_features_dat = []
		for header in self.output_headers:
			self.train_labels_dat[header] = []
		self.train_regime_dat = []
		self.regime_indices = {}

		for i in indices:
			normal_features = [self.normalize(self.all_dat[i][x], x) for x in self.input_headers]
			self.train_features_dat.append(normal_features)
			for header in self.output_headers:
				self.train_labels_dat[header].append(self.all_dat[i][header])
			regime_label = self.all_dat[i]["regime"]
			self.train_regime_dat.append(regime_label)
			if regime_label not in self.regime_indices:
				self.regime_indices[regime_label] = []
			self.regime_indices[regime_label].append(len(self.train_features_dat)-1)


