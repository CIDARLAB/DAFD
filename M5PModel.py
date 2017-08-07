"""M5P tree regression model

Only used now as a comparison tool
"""

from pprint import pprint
import math
from random import random
from scipy.optimize import minimize
import numpy
import csv



class M5PModel:
	"""Class to use the M5P trees"""
	def __init__(self):
		"""Read train data and initialize the model"""
		noutputs = 2 #Number of outputs (droplet size and generation rate)
		self.rand_dat = [] #List of dicts of the real csv data
		self.input_headers = []	#List of the names of our csv input columns
		self.output_headers = [] #List of the names of our csv output columns
		self.ranges_dict = {} #The min and max of each input type
		self.stddev_list = [] #The standard deviation of each input type
		values_dict = {}#Temporary variable used for calculating ranges. Dict with input header as key and a list of all values of that header as values

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

		#Find the range and standard deviation of each data type
		for head in headers:
			self.ranges_dict[head] = (min(values_dict[head]),max(values_dict[head]))
			self.stddev_list.append(numpy.asarray(values_dict[head]).std())


		#BUild the interpolation model
		self.M5P_models = {x:M5PTree(x,self) for x in self.output_headers}

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



	def getClosestPoint(self,constraints):
		"""Return closest real data point to our desired values that is within the given constraints
		Used to find a good starting point for our solution
		Also used for the nearest data point testing method
		"""
		self.closest_point = {}
		min_val = float("inf")
		for point in self.rand_dat:
			nval = (sum([abs(point[x]-self.desired_val_dict[x]) for x in self.desired_val_dict]) +
				sum([abs(point[x]-(sum(constraints[x])/2)) for x in constraints]))

			if nval < min_val:
				self.closest_point = point
				min_val = nval


	def linearizedError(self,x):
		"""Returns how far each solution deviates from the closest point
		Used in our minimization function
		"""
		run_sum = 0
		for i in range(len(x)):
			run_sum+=abs(x[i]/(self.stddev_list[i]**2)) #Standard dev is included to punish deviating from the smaller ranges more than the larger ranges
		return run_sum


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

		#The actual equation that all our solutions must conform to (from tree)
		cons_dict = {x:self.M5P_models[x].getEq(self.closest_point,desired_val_dict[x]) for x in desired_val_dict} 

		#Start at 0 (remember that our results are the changes to the closest data point, not final answers)
		start_pos = numpy.zeros(len(self.input_headers))

		#Find the solution to the equations that has a minimal linearized error
		res = minimize(self.linearizedError, start_pos, method='SLSQP', constraints=tuple(cons_dict.values()))
		dv = {self.input_headers[i]:res["x"][i] for i in range(len(self.input_headers))}

		#Return our real results
		results = {x:self.closest_point[x] + dv[x] for x in self.input_headers}

		return results




class M5PTree:
	"""The M5P tree that we must traverse for our linear equations"""
	def __init__(self,tree_output_var_name,master_class):
		self.pm = master_class
		self.output_var = tree_output_var_name
		self.rule_lines = []
		self.rules_level = []
		self.all_leaf_lines = {}
		read_to_leaf = False
		leaf_lines = []
		leaf_key = ""
		with open("M5P_models/"+tree_output_var_name+".tree","r") as f:
			for line in f:
				line = line.strip()
				if "<" in line or ">" in line:
					self.rule_lines.append(line)
				elif "LM num:" in line:
					read_to_leaf = True
					leaf_lines = []
					leaf_key = line.replace("LM num: ","")
				elif line == "":
					read_to_leaf = False
					if(leaf_key != ""):
						self.all_leaf_lines[leaf_key] = leaf_lines
				elif read_to_leaf:
					leaf_lines.append(line)

		self.rules_level = [x.count("|") for x in self.rule_lines]

	def getEq(self,real_data_line,wanted_solution):
		"""Return the linear regression equation of our closest data point"""
		index = 0
		keepLooking = True
		linearEquationKey = ""
		#Go through the tree to get our linear equation
		while(keepLooking):
			line = self.rule_lines[index]
			line = line.replace(" ","").replace("|","")
			rule = line.split(":")[0]
			variable = rule.split("<=")[0]
			condition = rule.split("<=")[1]
			if real_data_line[variable] <= float(condition):
				if "LM" in line:
					keepLooking = False
					linearEquationKey = line.split(":")[1].split("(")[0].replace("LM","")
					break
				index+=1
			else:
				start_level = self.rules_level[index]
				while(True):
					index+=1
					if self.rules_level[index] == start_level:
						if "LM" in self.rule_lines[index]:
							keepLooking = False
							linearEquationKey = self.rule_lines[index].split(":")[1].split("(")[0].replace("LM","").replace(" ","")
							break
						index+=1
						break


		linearEquation = self.all_leaf_lines[linearEquationKey]

		coefficients = {}
		for i in range(1,len(linearEquation)-1):
			line = linearEquation[i].replace(" ","")
			value = float(line.split("*")[0])
			variable = line.split("*")[1]
			coefficients[variable] = value

		constant = real_data_line[self.output_var] - sum([coefficients[x]*real_data_line[x] if x in coefficients else 0 for x in self.pm.input_headers])

		coefficients_list = [coefficients[x] if x in coefficients else 0 for x in self.pm.input_headers]
		line_list = [real_data_line[x] for x in self.pm.input_headers]
		
		cons = ({'type': 'eq', 'fun': lambda x:  sum([coefficients_list[i]*(x[i]+line_list[i]) for i in range(len(x))]) + constant - wanted_solution  })

		return cons
