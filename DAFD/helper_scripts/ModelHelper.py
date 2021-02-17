import os
import csv
import sklearn
import numpy as np


# Set this true to normalize the whole dataset against itself. If this is false, then the dataset will only get normalized
# within each regime. The correct option will depend on how the underlying machine learning models were trained. If you
# separated the data into regime 1 and regime 2 sets, then normalized each set indepently, set this to False. If you
# used one big dataset, normalized the entire set, then split the data after for each model, then this should be True.
always_use_whole_dataset_normalization = True

class ModelHelper:
	"""
	This Singleton class is a utility used by various scripts for data retrieval, partitioning, and normalization
	"""

	RESOURCE_PATH = "experimental_data/ExperimentalResults_888.csv"		# Experimental data location
	NUM_OUTPUTS = 2													# Droplet Generation Rate + Droplet Size

	instance = None						# Singleton. Use the get_instance method to return this.
	input_headers = []					# Feature names (orifice size, aspect ratio, etc...)
	output_headers = []					# Output names (droplet size, generation rate)
	all_dat = []						# Raw experimental data
	train_data_size = 0					# Number of data points in the training set
	train_features_dat_regnorm = []		# Normalized and reduced features from all_dat (regime normalized)
	train_features_dat_wholenorm = []	# Normalized and reduced features from all_dat (whole dataset normalized)
	train_labels_dat = {}				# Labels from all_dat
	train_regime_dat = []				# Regime labels from all_dat
	regime_indices = {}					# Indices of the TRAIN DATA (not the all_dat) that belong to each regime
	ranges_dict = {} 					# Dictionary of ranges for each ranges variable
	ranges_dict_normalized = {} 		# Dictionary of ranges for each ranges variable
	transform_dict = {}			# Dictionary of sklearn transform objects for normalization

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



	def resource_path(self, relative_path):
		""" Get absolute path to resource, works for dev and for PyInstaller """
		#try:
		#	# PyInstaller creates a temp folder and stores path in _MEIPASS
		#	base_path = sys._MEIPASS
		#except Exception:
		#	base_path = os.path.abspath(".")
		#
		#return os.path.join(base_path, relative_path)
		return os.path.dirname(os.path.abspath(__file__)) + "/" + relative_path

	def get_data(self):
		""" Read the data from the CSV list """

		# Temporary variable used for calculating ranges. Dict with input header as key and a list of all values of that header as values
		values_dict = {}

		with open(self.resource_path(self.RESOURCE_PATH)) as f:
			# Make a list of lists of our csv data
			lines = csv.reader(f, delimiter=',')

			# Save header info
			headers = next(lines)
			self.input_headers = [x for x in headers[:-self.NUM_OUTPUTS] if
									x != "regime" and x != "chip_number"]  # Regime and chip number isn't a feature we want to train our regressor on
			self.output_headers = headers[-self.NUM_OUTPUTS:]

			# Init values dict
			for head in headers:
				values_dict[head] = []
				values_dict[head+"_regime1"] = []
				values_dict[head+"_regime2"] = []

			# Get save the info of each row to var_dat and values_dict
			for row in lines:
				self.all_dat.append({headers[x]: float(row[x]) for x in range(len(headers))})
				for head_i in range(len(headers)):
					values_dict[headers[head_i]].append(float(row[head_i]))
				if float(row[9]) == 1:
					for head_i in range(len(headers)):
						values_dict[headers[head_i]+"_regime1"].append(float(row[head_i]))
				else:
					for head_i in range(len(headers)):
						values_dict[headers[head_i]+"_regime2"].append(float(row[head_i]))


		# Find the min and max to each data type
		for head in headers:
			self.transform_dict[head] = sklearn.preprocessing.StandardScaler()
			self.transform_dict[head].fit(np.array(values_dict[head]).reshape(-1,1))

			self.transform_dict[head+"_regime1"] = sklearn.preprocessing.StandardScaler()
			self.transform_dict[head+"_regime1"].fit(np.array(values_dict[head+"_regime1"]).reshape(-1,1))

			self.transform_dict[head+"_regime2"] = sklearn.preprocessing.StandardScaler()
			self.transform_dict[head+"_regime2"].fit(np.array(values_dict[head+"_regime2"]).reshape(-1,1))

			self.ranges_dict[head] = (min(values_dict[head]), max(values_dict[head]))
			self.ranges_dict_normalized[head] = (self.transform_dict[head].transform([[min(values_dict[head])]])[0][0],
									  self.transform_dict[head].transform([[max(values_dict[head])]])[0][0])

	def normalize_set(self, values, regime=""):
		""" Normalizes a set of features
		Args:
			values: list of features to be normalized (same order as input_headers)
			regime: regime1 or regime2 (nothing for whole-dataset normalization)

		Returns list of normalized features in the same order as input_headers
		"""
		ret_list = []
		for i, header in enumerate(self.input_headers):
			ret_list.append(self.normalize(values[i], header, regime))

		return ret_list

	def denormalize_set(self, values,regime=""):
		""" Denormalizes a set of features
		Args:
			values: list of features to be denormalized (same order as input_headers)
			regime: regime1 or regime2 (nothing for whole-dataset normalization)

		Returns list of denormalized features in the same order as input_headers
		"""
		ret_list = []
		for i, header in enumerate(self.input_headers):
			ret_list.append(self.denormalize(values[i], header, regime))

		return ret_list

	def normalize(self, value, inType,regime=""):
		"""Return min max normalization of a variable
		Args:
			value: Value to be normalized
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized
			regime: regime1 or regime2 (nothing for whole-dataset normalization)

		Returns 0-1 normalization of value with 0 being the min and 1 being the max
		"""
		if always_use_whole_dataset_normalization:
			regime=""
		return self.transform_dict[inType+regime].transform([[value]])[0][0]


	def denormalize(self, value, inType,regime=""):
		"""Return actual of a value of a normalized variable
		Args:
			value: Value to be corrected
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized
			regime: regime1 or regime2 (nothing for whole-dataset normalization)

		Returns actual value of given 0-1 normalized value
		"""
		if always_use_whole_dataset_normalization:
			regime=""
		return self.transform_dict[inType+regime].inverse_transform([[value]])[0][0]

	def make_train_data(self, indices=-1):
		""" Make training data from test data
			Indices are a list of wanted data (-1 indicates use all data)
		"""

		if indices == -1:
			indices = [x for x in range(len(self.all_dat))]

		self.train_features_dat_regnorm = []
		self.train_features_dat_wholenorm = []
		for header in self.output_headers:
			self.train_labels_dat[header] = []
		self.train_regime_dat = []
		self.regime_indices = {}

		for i in indices:
			regime_label = int(self.all_dat[i]["regime"])
			normal_features_regnorm = [self.normalize(self.all_dat[i][x], x, regime="_regime"+str(regime_label)) for x in self.input_headers]
			normal_features_wholenorm = [self.normalize(self.all_dat[i][x], x, regime="") for x in self.input_headers]
			self.train_features_dat_regnorm.append(normal_features_regnorm)
			self.train_features_dat_wholenorm.append(normal_features_wholenorm)
			for header in self.output_headers:
				self.train_labels_dat[header].append(self.all_dat[i][header])
			self.train_regime_dat.append(regime_label)
			if regime_label not in self.regime_indices:
				self.regime_indices[regime_label] = []
			self.regime_indices[regime_label].append(len(self.train_features_dat_wholenorm)-1)

		self.train_data_size = len(self.train_features_dat_wholenorm)


	def make_test_data(self, indices=-1):
		""" Make training data from test data
			Indices are a list of wanted data (-1 indicates use all data)
		"""

		if indices == -1:
			indices = [x for x in range(len(self.all_dat))]

		self.test_features_dat_regnorm = []
		self.test_features_dat_wholenorm = []
		self.test_labels_dat = {}
		self.test_regime_dat = []
		for header in self.output_headers:
			self.test_labels_dat[header] = []

		for i in indices:
			regime_label = int(self.all_dat[i]["regime"])
			normal_features_regnorm = [self.normalize(self.all_dat[i][x], x, regime="_regime"+str(regime_label)) for x in self.input_headers]
			normal_features_wholenorm = [self.normalize(self.all_dat[i][x], x, regime="_regime"+str(regime_label)) for x in self.input_headers]
			self.test_regime_dat.append(regime_label)
			self.test_features_dat_regnorm.append(normal_features_regnorm)
			self.test_features_dat_wholenorm.append(normal_features_wholenorm)
			for header in self.output_headers:
				self.test_labels_dat[header].append(self.all_dat[i][header])

	def calculate_formulaic_relations(self,design_inputs):

		"""
			Calculate water flow rate, oil flow rate, and inferred droplet size off the design inputs from DAFD forward model
		"""

		orifice_size = design_inputs["orifice_size"]
		aspect_ratio = design_inputs["aspect_ratio"]
		expansion_ratio = design_inputs["expansion_ratio"]
		normalized_orifice_length = design_inputs["normalized_orifice_length"]
		normalized_water_inlet = design_inputs["normalized_water_inlet"]
		normalized_oil_inlet = design_inputs["normalized_oil_inlet"]
		flow_rate_ratio = design_inputs["flow_rate_ratio"]
		capillary_number = design_inputs["capillary_number"]
		generation_rate = design_inputs["generation_rate"]

		channel_height = orifice_size * aspect_ratio
		outlet_channel_width = orifice_size * expansion_ratio
		orifice_length = orifice_size * normalized_orifice_length
		water_inlet_width = orifice_size * normalized_water_inlet
		oil_inlet = orifice_size * normalized_oil_inlet
		oil_flow_rate = (capillary_number * 0.005 * channel_height * oil_inlet * 1e-12) / \
						(0.0572 * ((water_inlet_width*1e-6)) * ((1 / (orifice_size*1e-6)) - (1 / (2 * oil_inlet * 1e-6))))
		oil_flow_rate_ml_per_hour = oil_flow_rate * 3600 * 1e6
		water_flow_rate = oil_flow_rate_ml_per_hour / flow_rate_ratio
		water_flow_rate_ul_per_min = water_flow_rate * 1000 / 60
		water_flow_rate_m3_per_s = water_flow_rate_ul_per_min * 1e-9/60

		droplet_volume_m3 = water_flow_rate_m3_per_s / generation_rate
		droplet_volume_nl = droplet_volume_m3 * 1e12
		droplet_diameter_m = (((droplet_volume_m3*6)/(3.14159)))**(1/3)

		droplet_inferred_size = droplet_diameter_m * 1e6

		return oil_flow_rate_ml_per_hour, water_flow_rate_ul_per_min, droplet_inferred_size
