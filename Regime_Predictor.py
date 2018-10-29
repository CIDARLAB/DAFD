""" Predict the regime given a set of chip geometries """

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
import random
import numpy
import itertools
import csv
import sys
import os

class RegimeClassifier:
	"""The interpolation models plus wrapper functions to use them"""

	def __init__(self):
		"""Make and save the interpolation models"""
		self.noutputs = 2 #Number of outputs (droplet size and generation rate)
		self.var_dat = [] #List of dicts of the real csv data
		self.input_headers = []	#List of the names of our csv input columns
		self.output_headers = [] #List of the names of our csv output columns
		self.ranges_dict = {} #The min and max of each input type
		self.read_data()
		#self.make_models()

	
	def read_data(self):
		"""Read the data from the CSV list"""

		values_dict = {} #Temporary variable used for calculating ranges. Dict with input header as key and a list of all values of that header as values

		with open(resource_path('ExperimentalResultsRegime_chipNum.csv')) as f:
			#Make a list of lists of our csv data
			lines = csv.reader(f, delimiter=',')

			#Save header info
			headers = next(lines)
			self.input_headers = [x for x in headers[:-self.noutputs] if x!= "regime" and x!= "chip_num"] # Regime and chip number isn't a feature we want to train our regressor on
			self.output_headers = headers[-self.noutputs:]
			
			#Init values dict
			for head in headers:
				values_dict[head] = [] 

			#Get save the info of each row to var_dat and values_dict
			for row in lines:
				self.var_dat.append({headers[x]:float(row[x]) for x in range(len(headers))})
				for head_i in range(len(headers)):
					values_dict[headers[head_i]].append(float(row[head_i]))

		#Find the min and max to each data type
		for head in headers:
			self.ranges_dict[head] = (min(values_dict[head]),max(values_dict[head]))

		
	
	def make_models(self):
		""" Make the interpolation model from the data"""
		self.make_models_reduced(self.var_dat)


	def make_models_reduced(self,dat_subset):
		""" Make the interpolation model from a subset of the data"""
		#Min max normalize all of our data
		var_dat_normal = []
		for var_point in dat_subset:
			normal_inputs = {x:self.normalize(var_point[x],x) for x in self.input_headers}
			var_dat_normal.append([normal_inputs[x] for x in self.input_headers])

		var_dat_normal = np.asarray(var_dat_normal)
		output_vars = {x:numpy.asarray([dat_line[x] for dat_line in dat_subset]) for x in self.output_headers}

		regime_vars = numpy.asarray([dat_line["regime"] for dat_line in dat_subset]) 


		self.svr_models = {}

		#Build an SVR model for each output variable at each jetting regime
		for reg in [1,2]:
			var_dat_reg = np.asarray([var_dat_normal[i] for i in range(len(var_dat_normal)) if regime_vars[i] == reg])
			for header in self.output_headers:
				header_reg = np.asarray([output_vars[header][i] for i in range(len(output_vars[header])) if regime_vars[i] == reg])

				SVR_model = SVR(C=100000,epsilon=0.00001)
				SVR_model.fit(var_dat_reg,header_reg)
				self.svr_models[str(reg) + header] = SVR_model 

		self.SVM_model = SVC(C=100000)
		self.SVM_model.fit(var_dat_normal,regime_vars)


	def validate_model(self, test_dat):
		""" Get test accuracies for the model """
		validations = {}
		for header in self.output_headers:
			val_header = []
			for test_point in test_dat:
				normal_inputs = {x:self.normalize(test_point[x],x) for x in self.input_headers}

				actual_regime = int(test_point["regime"])
				pred_reg = int(self.SVM_model.predict(np.asarray([normal_inputs[x] for x in self.input_headers]).reshape(1,-1))[0])

				actual_val = test_point[header]
				pred_val = self.svr_models[str(pred_reg)+header].predict(np.asarray([normal_inputs[x] for x in self.input_headers]).reshape(1,-1))[0]

				chip_num = int(test_point["chip_number"])

				val_header.append([actual_val,pred_val,abs(pred_val-actual_val),actual_regime,pred_reg,chip_num])

			validations[header] = val_header
		return validations
		

	def cross_validate(self):
		""" Perform cross-validation check of the data accuracy """
		folds = len(self.var_dat)
		
		rand_var_dat = random.sample(self.var_dat,len(self.var_dat))
		groups = [rand_var_dat[x:x+int(len(rand_var_dat)/folds)] for x in range(0,len(rand_var_dat),int(len(rand_var_dat)/folds))]

		validations = {}
		for i in range(folds):
			train_dat = [x for x in itertools.chain.from_iterable(groups[:i] + groups[i+1:])]
			test_dat = groups[i]
			self.make_models_reduced(train_dat)
			fold_validations = self.validate_model(test_dat)
			for header in fold_validations:
				if header not in validations:
					validations[header] = []
				validations[header].extend(fold_validations[header])
			
		
		self.wrong_regimes = {}
		self.perc10_wrong_preds = {}
		for i in range(2):
			out_var = self.output_headers[i]
			print(out_var)
			self.wrong_regimes[out_var] = []
			self.perc10_wrong_preds[out_var] = []

			grs = sorted(validations[self.output_headers[i]],key = lambda x: x[5])

			total_num = len(grs)
			set_mean = sum([x[0] for x in grs])/total_num

			total_right = 0
			for x in grs:
				if int(x[3]) == int(x[4]):
					total_right+=1
				else:
					self.wrong_regimes[out_var].append(x)
					

			print("SVM Accuracy: " + str(total_right/total_num))

			total_squared_pred_error = 0
			total_squared_mean_error = 0
			for x in grs:
				deviation = abs(x[0]-x[1])
				total_squared_pred_error += (deviation)**2
				total_squared_mean_error += (x[1]-set_mean)**2
				
				if deviation > x[0]*0.1:
					self.perc10_wrong_preds[out_var].append(x)

			r2 = 1 - (total_squared_pred_error/total_squared_mean_error)
			print("SVR R2 (no bias): " + str(r2))

			
			pred_vals = np.asarray([x[0] for x in grs])
			actual_vals = np.asarray([x[1] for x in grs])
			
			plt.figure()
			plt.title(self.output_headers[i])
			plt.plot(pred_vals,actual_vals,'o')
			
			reg = LinearRegression().fit(pred_vals.reshape(-1,1),actual_vals.reshape(-1,1))
			print("SVR R2 (bias): " + str(reg.score(pred_vals.reshape(-1,1),actual_vals.reshape(-1,1))))
			
			# Write to disc
			out_headers = ["actual_val","pred_val","deviation","actual_regime","pred_regime","chip_number"]
			with open("all_preds_" + out_var + ".csv","w") as f:
				f.write(",".join(out_headers) + "\n") 
				for x in grs:
					f.write(",".join([str(xi) for xi in x]) + "\n")

			with open("bad_regimes_" + out_var + ".csv","w") as f:
				f.write(",".join(out_headers) + "\n") 
				for x in self.wrong_regimes[out_var]:
					f.write(",".join([str(xi) for xi in x]) + "\n")


			with open("bad_preds_" + out_var + ".csv","w") as f:
				f.write(",".join(out_headers) + "\n") 
				for x in self.perc10_wrong_preds[out_var]:
					f.write(",".join([str(xi) for xi in x]) + "\n")


				
				


		plt.show()






	def normalize(self,value,inType):
		"""Return min max normalization of a variable
		Args:
			value: Value to be normalized
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized

		Returns 0-1 normalization of value with 0 being the min and 1 being the max
		"""
		return (value-self.ranges_dict[inType][0])/(self.ranges_dict[inType][1]-self.ranges_dict[inType][0])
		
	def denormalize(self,value,inType):
		"""Return actual of a value of a normalized variable
		Args:
			value: Value to be corrected
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized

		Returns actual value of given 0-1 normalized value
		"""
		return value*(self.ranges_dict[inType][1]-self.ranges_dict[inType][0])+self.ranges_dict[inType][0]

	def rbf_error(self,x):
		"""Returns how far each solution mapped on the model deviates from the desired value
		Used in our minimization function
		"""
		merrors = [abs(self.interp_models[head](*x) - self.desired_val_dict[head]) for head in self.desired_val_dict]
		return sum(merrors)

	def getClosestPoint(self,constraints):
		"""Return closest real data point to our desired values that is within the given constraints
		Used to find a good starting point for our solution
		Also used for the nearest data point testing method
		"""
		self.closest_point = {}
		min_val = float("inf")
		for point in self.var_dat:
			nval = (sum([self.normalize(abs(point[x]-self.desired_val_dict[x]),x) for x in self.desired_val_dict]) +
				sum([self.normalize(abs(point[x]-(sum(constraints[x])/2)),x) for x in constraints]))
			
			if nval < min_val:
				self.closest_point = point
				min_val = nval

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

		self.desired_val_dict = desired_val_dict

		self.getClosestPoint(constraints)
		


		# Normalize constraints
		nconstraints = {x:( max([self.normalize(constraints[x][0],x),0]), min([self.normalize(constraints[x][1],x),1]) ) for x in constraints}

		#Get acceptable starting point
		#Middle of our constraints if there is a constraint for that input type, otherwise the value of the type at the closest data point
		start_pos = numpy.asarray([(nconstraints[x][0]+nconstraints[x][1])/2 if x in constraints else self.normalize(self.closest_point[x],x)  for x in self.input_headers])

		#Minimization function
		res = minimize(self.rbf_error, 
				start_pos, 
				method='SLSQP',
				bounds = tuple([(nconstraints[x][0],nconstraints[x][1]) if x in nconstraints else (0,1) for x in self.input_headers]))

		#Denormalize results
		results = {self.input_headers[i]:
				self.denormalize(res["x"][i],self.input_headers[i])
				for i in range(len(self.input_headers))}

		return results
	
	
        
svrm = SVRModel()
svrm.cross_validate()

