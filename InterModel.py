"""The interpolation model that DAFD runs on"""

from tqdm import tqdm
from scipy.interpolate import Rbf
import numpy as np
import math
from random import random
from scipy.optimize import minimize
import numpy
import csv

class InterModel:
	"""The interpolation models plus wrapper functions to use them"""

	def __init__(self):
		"""Make and save the interpolation models"""
		noutputs = 2 #Number of outputs (droplet size and generation rate)
		self.rand_dat = [] #List of dicts of the real csv data
		self.input_headers = []	#List of the names of our csv input columns
		self.output_headers = [] #List of the names of our csv output columns
		self.ranges_dict = {} #The min and max of each input type
		values_dict = {} #Temporary variable used for calculating ranges. Dict with input header as key and a list of all values of that header as values

		with open('MicroFluidics_Random.csv') as f:
			#Make a list of lists of our csv data
			lines = csv.reader(f, delimiter=',')

			#Save header info
			headers = next(lines)
			self.input_headers = headers[:-noutputs]
			self.output_headers = headers[-noutputs:]
			
			#Init values dict
			for head in headers:
				values_dict[head] = [] 

			#Get save the info of each row to rand_dat and values_dict
			for row in lines:
				self.rand_dat.append({headers[x]:float(row[x]) for x in range(len(headers))})
				for head_i in range(len(headers)):
					values_dict[headers[head_i]].append(float(row[head_i]))

		#Find the min and max to each data type
		for head in headers:
			self.ranges_dict[head] = (min(values_dict[head]),max(values_dict[head]))


		#Min max normalize all of our data
		rand_dat_np = numpy.asarray([[dat_line[x] for x in self.input_headers] for dat_line in self.rand_dat])
		rand_dat_normal = np.asarray([(rand_dat_np[:,x]-rand_dat_np[:,x].min())/(rand_dat_np[:,x].max()-rand_dat_np[:,x].min()) for x in range(rand_dat_np.shape[1])])

		#Build an interpolation model for each output variable
		self.interp_models = {x:Rbf(*rand_dat_normal, numpy.asarray([dat_line[x] for dat_line in self.rand_dat])) for x in self.output_headers}




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
		for point in self.rand_dat:
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
	
	

