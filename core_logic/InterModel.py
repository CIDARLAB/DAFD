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


	def get_closest_point(self, desired_vals, constraints={}, max_drop_exp_error=-1, skip_list = []):
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
			if i in skip_list:
				continue
			if use_regime_2 and self.MH.train_regime_dat[i] != 2:
				continue

			if max_drop_exp_error != -1 and "droplet_size" in desired_vals:
				exp_error = abs(self.MH.denormalize(desired_vals["droplet_size"],"droplet_size") - self.MH.train_labels_dat["droplet_size"][i])
				if exp_error > max_drop_exp_error:
					continue

			feat_point = self.MH.train_features_dat[i]
			prediction = self.fwd_model.predict(feat_point, normalized=True)

			if prediction["regime"] != self.MH.train_regime_dat[i]:
				continue

			if self.constrained_regime != -1 and prediction["regime"] != self.constrained_regime:
				continue

			nval = sum([abs(self.MH.normalize(self.MH.train_labels_dat[x][i], x) - desired_vals[x]) for x in desired_vals])
			if "droplet_size" in desired_vals:
				nval += abs(self.MH.normalize(prediction["droplet_size"],"droplet_size") - desired_vals["droplet_size"])

				denorm_feat_list = self.MH.denormalize_set(feat_point)
				denorm_feat = {x:denorm_feat_list[i] for i,x in enumerate(self.MH.input_headers)}
				denorm_feat["generation_rate"] = prediction["generation_rate"]

				_,_,inferred_size = self.MH.calculate_formulaic_relations(denorm_feat)
				inferred_size_error = abs(desired_vals["droplet_size"] - self.MH.normalize(inferred_size,"droplet_size"))
				nval+=inferred_size_error

			if "generation_rate" in desired_vals:
				nval += abs(self.MH.normalize(prediction["generation_rate"],"generation_rate") - desired_vals["generation_rate"])

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
		val_dict = {self.MH.input_headers[i]:self.MH.denormalize(val,self.MH.input_headers[i]) for i,val in enumerate(x)}
		val_dict["generation_rate"] = prediction["generation_rate"]
		_, _, droplet_inferred_size = self.MH.calculate_formulaic_relations(val_dict)
		if "droplet_size" in self.desired_vals_global:
			denorm_drop_error = abs(droplet_inferred_size - self.desired_vals_global["droplet_size"])
			drop_error = abs(self.MH.normalize(droplet_inferred_size,"droplet_size") - self.norm_desired_vals_global["droplet_size"])
		else:
			denorm_drop_error = abs(droplet_inferred_size - prediction["droplet_size"])
			drop_error = abs(self.MH.normalize(droplet_inferred_size,"droplet_size") - self.MH.normalize(prediction["droplet_size"],"droplet_size"))
		print(prediction["droplet_size"])
		print(droplet_inferred_size)
		print(denorm_drop_error)
		print(drop_error)
		print()

		with open("InterResults.csv","a") as f:
			f.write(",".join(map(str,x)) + "," + str(prediction['regime']) + "," + str(prediction['generation_rate']) +
					"," + str(prediction['droplet_size']) + "," + str(denorm_drop_error) + "\n")

		merrors = [abs(self.MH.normalize(prediction[head], head) - self.norm_desired_vals_global[head]) for head in self.norm_desired_vals_global]
		return sum(merrors) + drop_error*1


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

		self.constrained_regime = constraints.pop("regime", -1)

		norm_constraints = {}
		for cname in constraints:
			cons_low = self.MH.normalize(constraints[cname][0],cname)
			cons_high = self.MH.normalize(constraints[cname][1],cname)
			norm_constraints[cname] = (cons_low, cons_high)

		norm_desired_vals = {}
		for lname in desired_val_dict:
			norm_desired_vals[lname] = self.MH.normalize(desired_val_dict[lname], lname)

		self.norm_desired_vals_global = norm_desired_vals
		self.desired_vals_global = desired_val_dict



		skip_list = []
		while(True):
			start_pos, closest_index = self.get_closest_point(norm_desired_vals, constraints=norm_constraints, max_drop_exp_error=5, skip_list=skip_list)
			if closest_index == -1:
				start_pos, closest_index = self.get_closest_point(norm_desired_vals, constraints=norm_constraints)
				break
			skip_list.append(closest_index)

			prediction = self.fwd_model.predict(start_pos, normalized=True)
			all_dat_labels = ["chip_number"] + self.MH.input_headers + ["regime"] + self.MH.output_headers
			print(",".join(all_dat_labels))
			print("Starting point")
			print(self.MH.all_dat[closest_index])
			print([self.MH.all_dat[closest_index][x] for x in all_dat_labels])
			print("Start pred")
			print(prediction)

			should_skip_optim_rate = True
			should_skip_optim_size = True
			should_skip_optim_constraints = True

			for constraint in constraints:
				cons_range = constraints[constraint]
				this_val = self.MH.all_dat[closest_index][constraint]
				if this_val < cons_range[0] or this_val > cons_range[1]:
					should_skip_optim_constraints = False


			if self.constrained_regime != -1 and self.MH.all_dat[closest_index]["regime"] != self.constrained_regime:
				should_skip_optim_constraints = False

			if "generation_rate" in desired_val_dict:
				if desired_val_dict["generation_rate"] > 100:
					pred_rate_error = abs(desired_val_dict["generation_rate"] - prediction["generation_rate"]) / desired_val_dict["generation_rate"]
					exp_rate_error = abs(desired_val_dict["generation_rate"] - self.MH.all_dat[closest_index]["generation_rate"]) / self.MH.all_dat[closest_index]["generation_rate"]
					if pred_rate_error > 0.15 or exp_rate_error > 0.15:
						should_skip_optim_rate = False
				else:
					pred_rate_error = abs(desired_val_dict["generation_rate"] - prediction["generation_rate"])
					exp_rate_error = abs(desired_val_dict["generation_rate"] - self.MH.all_dat[closest_index]["generation_rate"])
					if pred_rate_error > 15 or exp_rate_error > 15:
						should_skip_optim_rate = False

			if "droplet_size" in desired_val_dict:
				pred_size_error = abs(desired_val_dict["droplet_size"] - prediction["droplet_size"])
				exp_size_error = abs(desired_val_dict["droplet_size"] - self.MH.all_dat[closest_index]["droplet_size"])
				print(self.MH.all_dat[closest_index])
				pred_point = {x:self.MH.all_dat[closest_index][x] for x in self.MH.all_dat[closest_index]}
				pred_point["generation_rate"] = prediction["generation_rate"]
				print(self.MH.all_dat[closest_index])
				_,_,inferred_size = self.MH.calculate_formulaic_relations(pred_point)
				inferred_size_error = abs(desired_val_dict["droplet_size"] - inferred_size)
				print(inferred_size)
				print(inferred_size_error)
				if pred_size_error > 10 or inferred_size_error > 10 or exp_size_error > 5:
					should_skip_optim_size = False

			if should_skip_optim_rate and should_skip_optim_size and should_skip_optim_constraints:
				results = {x: self.MH.all_dat[closest_index][x] for x in self.MH.input_headers}
				results["point_source"] = "Experimental"
				print(results)
				return results

		with open("InterResults.csv","w") as f:
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
			f.write(",".join(self.MH.input_headers) + ",regime,generation_rate,droplet_size,inferred_drop_error\n")

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
		print("Final Suggestions")
		print(",".join(self.MH.input_headers) + "," + "desired_size" + "," + "predicted_generation_rate" + "," + "predicted_droplet_size")
		output_string = ",".join([str(results[x]) for x in self.MH.input_headers])
		output_string += "," + str(desired_val_dict["droplet_size"])
		output_string += "," + str(prediction["generation_rate"])
		output_string += "," + str(prediction["droplet_size"])
		print(output_string)
		print("Final Prediction")
		print(prediction)
		results["point_source"] = "Predicted"
		return results

