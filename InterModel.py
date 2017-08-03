#from scipy.interpolate import griddata
#from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
from scipy.interpolate import Rbf
import numpy as np
import math
from random import random
from scipy.optimize import minimize
import numpy
import csv

class InterModel:

	def __init__(self):
		self.rand_dat = []
		self.input_headers = []
		self.ranges_dict = {}
		values_dict = {}
		with open('MicroFluidics_Random.csv') as f:
			lines = csv.reader(f, delimiter=',')
			headers = next(lines)
			self.input_headers = headers[:-2]
			for head in headers:
				values_dict[head] = [] 
			for row in lines:
				self.rand_dat.append({headers[x]:float(row[x]) for x in range(len(headers))})
				for head_i in range(len(headers)):
					values_dict[headers[head_i]].append(float(row[head_i]))

		for head in headers:
			self.ranges_dict[head] = (min(values_dict[head]),max(values_dict[head]))


		rand_dat_np = numpy.asarray([[dat_line[x] for x in self.input_headers] for dat_line in self.rand_dat])
		rand_dat_normal = np.asarray([(rand_dat_np[:,x]-rand_dat_np[:,x].min())/(rand_dat_np[:,x].max()-rand_dat_np[:,x].min()) for x in range(rand_dat_np.shape[1])])

		self.drop_size_fit = Rbf(*rand_dat_normal, numpy.asarray([dat_line["droplet_size"] for dat_line in self.rand_dat]))
		self.gen_rate_fit = Rbf(*rand_dat_normal, numpy.asarray([dat_line["generation_rate"] for dat_line in self.rand_dat]))




	def normalize(self,value,inType):
		return (value-self.ranges_dict[inType][0])/(self.ranges_dict[inType][1]-self.ranges_dict[inType][0])
		
	def denormalize(self,value,inType):
		return value*(self.ranges_dict[inType][1]-self.ranges_dict[inType][0])+self.ranges_dict[inType][0]

	def rbf_error(self,x):
		merror1 = abs(self.drop_size_fit(*x) - self.drop_size)
		merror2 = abs(self.gen_rate_fit(*x) - self.generation_rate)
		return merror1+merror2

	def getClosestPoint(self,constraints):
		self.closest_point = {}
		min_val = float("inf")
		for point in self.rand_dat:
			nval = (abs(point["droplet_size"]-self.drop_size) + 
				abs(point["generation_rate"]-self.generation_rate) + 
				sum([abs(point[x]-(sum(constraints[x])/2)) for x in constraints]))
			
			if nval < min_val:
				self.closest_point = point
				min_val = nval

	def interpolate(self,drop_size,generation_rate,constraints):
		self.drop_size = drop_size
		self.generation_rate = generation_rate

		self.getClosestPoint(constraints)
		


		nconstraints = {x:( max([self.normalize(constraints[x][0],x),0]), min([self.normalize(constraints[x][1],x),1]) ) for x in constraints}

		start_pos = numpy.asarray([(nconstraints[x][0]+nconstraints[x][1])/2 if x in constraints else self.normalize(self.closest_point[x],x)  for x in self.input_headers])

		res = minimize(self.rbf_error, 
				start_pos, 
				method='SLSQP',
				bounds = tuple([(nconstraints[x][0],nconstraints[x][1]) if x in nconstraints else (0,1) for x in self.input_headers]))

		results = {self.input_headers[i]:
				self.denormalize(res["x"][i],self.input_headers[i])
				for i in range(len(self.input_headers))}

		return results
	
	

