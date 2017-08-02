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
	
	


#Used in testing. The dummy equation the data was built from
def equationOutputs(args):
	drop_size = prod([args["orifice_size"],args["aspect_ratio"],args["width_ratio"],args["normalized_orifice_length"],args["normalized_water_input_width"]])/prod([args["normalized_oil_input_width"],args["capillary_number"],args["flow_rate_ratio"],10]);
	generation_rate = prod([sum([args["aspect_ratio"],args["width_ratio"],args["normalized_orifice_length"],args["normalized_oil_input_width"],args["normalized_water_input_width"],args["flow_rate_ratio"]]),args["capillary_number"],5000]) / args["orifice_size"]

	return drop_size,generation_rate


def prod(args):
	product = 1
	for x in args:
		product*=x
	return product

drop_size_errors = []
generation_rate_errors = []

drop_size_nonmod_errors = []
generation_rate_nonmod_errors = []
it = InterModel()
for i in tqdm(range(100)):
		dummy_inputs = {"orifice_size":random()*250+50,
			"aspect_ratio":random()*2+1,
			"width_ratio":random()*2+2,
			"normalized_orifice_length":random()*8+1,
			"normalized_oil_input_width":random()*2+2,
			"normalized_water_input_width":random()*2+2,
			"capillary_number":random()*(0.2222-0.02)+0.02,
			"flow_rate_ratio":random()*18+2}

		drop_size, generation_rate = equationOutputs(dummy_inputs)
		results = it.interpolate(drop_size,generation_rate)
			

		drop_size_errors.append( abs(equationOutputs(results)[0] - drop_size)/drop_size)
		generation_rate_errors.append( abs(equationOutputs(results)[1] - generation_rate)/generation_rate)
		drop_size_nonmod_errors.append( abs(equationOutputs(it.closest_point)[0] - drop_size)/drop_size)
		generation_rate_nonmod_errors.append( abs(equationOutputs(it.closest_point)[1] - generation_rate)/generation_rate)

		

		
print()
print("drop size errors:			"+str(round(sum(drop_size_errors)/len(drop_size_errors) * 100,4)).zfill(7) + "%")
print("generation rate errors:			"+str(round(sum(generation_rate_errors)/len(generation_rate_errors) * 100,4)).zfill(7) + "%")
print()
print("drop size closest point errors:		"+str(round(sum(drop_size_nonmod_errors)/len(drop_size_nonmod_errors) * 100,4)).zfill(7) + "%")
print("generation rate closest point errors:	"+str(round(sum(generation_rate_nonmod_errors)/len(generation_rate_nonmod_errors) * 100,4)).zfill(7) + "%")


