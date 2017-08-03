from pprint import pprint
import math
from random import random
from scipy.optimize import minimize
import numpy
import csv



class M5PModel:
	def __init__(self):
		noutputs = 2 #Droplet size and generation rate
		self.rand_dat = []
		self.input_headers = []
		self.output_headers = []
		self.ranges_dict = {}
		self.stddev_list = []
		values_dict = {}
		with open('MicroFluidics_Random.csv') as f:
			lines = csv.reader(f, delimiter=',')
			headers = next(lines)
			self.input_headers = headers[:-noutputs]
			self.output_headers = headers[-noutputs:]
			for head in headers:
				values_dict[head] = [] 
			for row in lines:
				self.rand_dat.append({headers[x]:float(row[x]) for x in range(len(headers))})
				for head_i in range(len(headers)):
					values_dict[headers[head_i]].append(float(row[head_i]))

		for head in headers:
			self.ranges_dict[head] = (min(values_dict[head]),max(values_dict[head]))
			self.stddev_list.append(numpy.asarray(values_dict[head]).std())



		self.M5P_models = {x:M5PTree(x,self) for x in self.output_headers}

	def normalize(self,value,inType):
		return (value-self.ranges_dict[inType][0])/(self.ranges_dict[inType][1]-self.ranges_dict[inType][0])

	def denormalize(self,value,inType):
		return value*(self.ranges_dict[inType][1]-self.ranges_dict[inType][0])+self.ranges_dict[inType][0]



	def getClosestPoint(self,constraints):
		self.closest_point = {}
		min_val = float("inf")
		for point in self.rand_dat:
			nval = (sum([abs(point[x]-self.desired_val_dict[x]) for x in self.desired_val_dict]) +
				sum([abs(point[x]-(sum(constraints[x])/2)) for x in constraints]))

			if nval < min_val:
				self.closest_point = point
				min_val = nval


	def linearizedError(self,x):
		run_sum = 0
		for i in range(len(x)):
			run_sum+=abs(x[i]/(self.stddev_list[i]**2))
		return run_sum


	def interpolate(self,desired_val_dict,constraints):
		self.desired_val_dict = desired_val_dict

		self.getClosestPoint(constraints)

		#nconstraints = {x:( max(constraints[x][0],self.ranges_dict[x][0]]), min(constraints[x][1],self.ranges_dict[x][1]]) ) for x in constraints}

		#start_pos = numpy.asarray([(nconstraints[x][0]+nconstraints[x][1])/2 if x in constraints else self.closest_point[x]  for x in self.input_headers])

		#res = minimize(self.rbf_error,
		#		start_pos,
		#		method='SLSQP',
		#		bounds = tuple([(nconstraints[x][0],nconstraints[x][1]) if x in nconstraints else (0,1) for x in self.input_headers]))

		#results = {self.input_headers[i]:
		#		self.denormalize(res["x"][i],self.input_headers[i])
		#		for i in range(len(self.input_headers))}

		cons_dict = {x:self.M5P_models[x].getDx(self.closest_point,desired_val_dict[x]) for x in desired_val_dict} 
		start_pos = numpy.zeros(len(self.input_headers))

		res = minimize(self.linearizedError, start_pos, method='SLSQP', constraints=tuple(cons_dict.values()))
		dv = {self.input_headers[i]:res["x"][i] for i in range(len(self.input_headers))}

		results = {x:self.closest_point[x] + dv[x] for x in self.input_headers}

		return results




class M5PTree:
	def __init__(self,tree_output_var_name,master_class):
		self.pm = master_class
		self.output_var = tree_output_var_name
		self.rule_lines = []
		self.rules_level = []
		self.all_leaf_lines = {}
		read_to_leaf = False
		leaf_lines = []
		leaf_key = ""
		with open(tree_output_var_name+".txt","r") as f:
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

	def getDx(self,real_data_line,wanted_solution):
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
