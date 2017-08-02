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
			for head in self.input_headers:
				values_dict[head] = [] 
			for row in lines:
				self.rand_dat.append({headers[x]:float(row[x]) for x in range(len(headers))})
				for head_i in range(len(self.input_headers)):
					values_dict[self.input_headers[head_i]].append(float(row[head_i]))

		for head in self.input_headers:
			self.ranges_dict[head] = (min(values_dict[head]),max(values_dict[head]))


		rand_dat_np = numpy.asarray([[dat_line[x] for x in self.input_headers] for dat_line in self.rand_dat])
		rand_dat_normal = np.asarray([(rand_dat_np[:,x]-rand_dat_np[:,x].min())/(rand_dat_np[:,x].max()-rand_dat_np[:,x].min()) for x in range(rand_dat_np.shape[1])])

		self.drop_size_fit = Rbf(*rand_dat_normal, numpy.asarray([dat_line["droplet_size"] for dat_line in self.rand_dat]))
		self.gen_rate_fit = Rbf(*rand_dat_normal, numpy.asarray([dat_line["generation_rate"] for dat_line in self.rand_dat]))






	def rbf_error(self,x):
		merror1 = abs(self.drop_size_fit(*x) - self.drop_size)
		merror2 = abs(self.gen_rate_fit(*x) - self.generation_rate)
		return merror1+merror2

	def getClosestPoint(self):
		self.closest_point = {}
		min_val = float("inf")
		for point in self.rand_dat:
			if abs(point["droplet_size"]-self.drop_size) + abs(point["generation_rate"]-self.generation_rate) < min_val:
				self.closest_point = point
				min_val = abs(point["droplet_size"]-self.drop_size) + abs(point["generation_rate"]-self.generation_rate)

	def interpolate(self,drop_size,generation_rate):
		self.drop_size = drop_size
		self.generation_rate = generation_rate

		self.getClosestPoint()
		start_pos = numpy.asarray([(self.closest_point[x]-self.ranges_dict[x][0])/(self.ranges_dict[x][1]-self.ranges_dict[x][0])  for x in self.input_headers])

		res = minimize(self.rbf_error, start_pos, method='SLSQP',bounds = tuple([(0,1) for x in self.input_headers]))

		results = {self.input_headers[i]:
				res["x"][i]*(self.ranges_dict[self.input_headers[i]][1]-self.ranges_dict[self.input_headers[i]][0])+self.ranges_dict[self.input_headers[i]][0] 
				for i in range(len(self.input_headers))}

		return results
	
	

