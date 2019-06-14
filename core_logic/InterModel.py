"""The interpolation model that DAFD runs on"""

from tqdm import tqdm
from scipy.interpolate import Rbf
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
import random
import numpy
import itertools
import csv
import sys
import os
from core_logic.ForwardModel import ForwardModel
from helper_scripts.ModelHelper import ModelHelper

from matplotlib import pyplot as plt

def resource_path(relative_path):
	""" Get absolute path to resource, works for dev and for PyInstaller """
	try:
		# PyInstaller creates a temp folder and stores path in _MEIPASS
		base_path = sys._MEIPASS
	except Exception:
		base_path = os.path.abspath(".")

	return os.path.join(base_path, relative_path)

class InterModel:
	"""
	This class handles interpolation over our forward models to make the reverse predictions
	"""

	def __init__(self):
		"""Make and save the interpolation models"""
		self.MH = ModelHelper.get_instance() # type: ModelHelper
		self.fwd_model = ForwardModel()


	def get_closest_point(self, desired_vals, constraints):
		"""Return closest real data point to our desired values that is within the given constraints
		Used to find a good starting point for our solution
		THIS IS FUNDAMENTALLY DIFFERENT FROM THE NEAREST DATA POINT FORWARD MODEL!
			Nearest data point forward model - Find the outputs of the closest data point to the prediction
			This method - Find the data point closest to the desired outputs

		We will try to find the point closest to the center of our constraints that is close to the target answer

		ALL INPUTS ARE NORMALIZED!

		By itself, this class really isn't all that bad at performing DAFD's main functionality: reverse model prediction
			Therefore, this class should be the baseline level of accuracy for DAFD.
		"""

		use_regime_2 = False
		if "orifice_size" in constraints and "droplet_size" in desired_vals and self.MH.denormalize(constraints["orifice_size"][1],"orifice_size") < self.MH.denormalize(desired_vals["droplet_size"],"droplet_size"):
			use_regime_2 = True

		closest_point = {}
		min_val = float("inf")
		match_index = -1
		for i in range(self.MH.train_data_size):
			if use_regime_2 and self.MH.train_regime_dat[i] != 2:
				continue

			nval = sum([abs(self.MH.normalize(self.MH.train_labels_dat[x][i], x) - desired_vals[x]) for x in desired_vals])

			feat_point = self.MH.train_features_dat[i]
			for j in range(len(self.MH.input_headers)):
				if self.MH.input_headers[j] in 	constraints:
					cname = self.MH.input_headers[j]
					nval += abs(feat_point[j] - (constraints[cname][0] + constraints[cname][1])/2.0)

			if nval < min_val:
				closest_point = feat_point
				min_val = nval
				match_index = i

		return closest_point, match_index

	def model_error(self, x):
		"""Returns how far each solution mapped on the model deviates from the desired value
		Used in our minimization function
		"""
		prediction = self.fwd_model.predict(x, normalized=True)
		#merrors = [abs(self.MH.normalize(prediction[head], head) - self.norm_desired_vals_global_adjusted[head]) for head in self.norm_desired_vals_global_adjusted]
		with open("../model_data/InterResults.csv","a") as f:
			f.write(",".join(map(str,x)) + "," + str(prediction['regime']) + "," + str(prediction['generation_rate']) + "," + str(prediction['droplet_size']) + "\n")
		merrors = [abs(self.MH.normalize(prediction[head], head) - self.norm_desired_vals_global[head]) for head in self.norm_desired_vals_global]
		return sum(merrors)


	def interpolate(self,desired_val_dict,constraints):
		"""Return an input set within the given constraints that produces the output set
		The core part of DAFD
		Args:
			desired_val_dict: Dict with output type as the key and desired value as the value
				Just don't include other output type if you just want to interpolate on one

			constraints: Dict with input type as key and acceptable range as the value
				The acceptable range should be a tuple with the min as the first val and the max as the second val
				Again, just leave input types you don't care about blank
		"""

		norm_constraints = {}
		for cname in constraints:
			cons_low = self.MH.normalize(constraints[cname][0],cname)
			cons_high = self.MH.normalize(constraints[cname][1],cname)
			norm_constraints[cname] = (cons_low, cons_high)

		norm_desired_vals = {}
		for lname in desired_val_dict:
			norm_desired_vals[lname] = self.MH.normalize(desired_val_dict[lname], lname)

		self.norm_desired_vals_global = norm_desired_vals
		start_pos, closest_index = self.get_closest_point(norm_desired_vals, norm_constraints)
		closest_labels = {}
		for head in self.MH.output_headers:
			closest_labels[head] = self.MH.train_labels_dat[head][closest_index]

		#Get acceptable starting point
		self.first_point = start_pos

		prediction = self.fwd_model.predict(start_pos, normalized=True)
		all_dat_labels = ["chip_number"] + self.MH.input_headers + ["regime"] + self.MH.output_headers
		print(",".join(all_dat_labels))
		print("Starting point")
		print(self.MH.all_dat[closest_index])
		print([self.MH.all_dat[closest_index][x] for x in all_dat_labels])
		print("Start pred")
		print(prediction)

		with open("../model_data/InterResults.csv","w") as f:
			f.write("Experimental outputs:"+str(self.MH.all_dat[closest_index]["generation_rate"])+","+str(self.MH.all_dat[closest_index]["droplet_size"])+"\n")

			if "generation_rate" not in desired_val_dict:
				des_rate = "-1"
			else:
				des_rate = str(desired_val_dict["generation_rate"])

			if "droplet_size" not in desired_val_dict:
				des_size = "-1"
			else:
				des_size = str(desired_val_dict["droplet_size"])

			f.write("Desired outputs:"+des_rate+","+des_size+"\n")
			f.write(",".join(self.MH.input_headers) + ",regime,generation_rate,droplet_size\n")

		#Adjust target to match forward model error (gradient method)
		#prediction = self.fwd_model.predict(start_pos, normalized=True)
		#self.norm_desired_vals_global_adjusted = {}
		#for head in norm_desired_vals:
		#	diff = self.MH.normalize(prediction[head],head) - self.MH.normalize(closest_labels[head],head)
		#	self.norm_desired_vals_global_adjusted[head] = norm_desired_vals[head] + diff




		options = {'eps':1e-6}

		#Minimization function
		res = minimize(self.model_error,
				start_pos, 
				method='SLSQP',
				options=options,
				bounds = tuple([(norm_constraints[x][0],norm_constraints[x][1])
								if x in norm_constraints
								else (self.MH.ranges_dict_normalized[x][0],self.MH.ranges_dict_normalized[x][1])
								for x in self.MH.input_headers]))

		self.last_point = [res["x"][i] for i in range(len(res["x"]))]

		#Denormalize results
		results = {x: self.MH.denormalize(res["x"][i], x) for i, x in enumerate(self.MH.input_headers)}
		prediction = self.fwd_model.predict([results[x] for x in self.MH.input_headers])
		print("Final Prediction")
		print(prediction)
		return results

