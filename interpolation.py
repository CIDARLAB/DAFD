#from scipy.interpolate import griddata
#from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf
import numpy as np
import math
from random import random
from scipy.optimize import minimize
import numpy
import csv

rand_dat = []
input_headers = []
min_dict = {}
max_dict = {}
stddev_dict = {} #Dict of standard deviations
ranges_dict = {}
with open('MicroFluidics_Random.csv') as f:
	lines = csv.reader(f, delimiter=',')
	headers = next(lines)
	input_headers = headers[:-2]
	for head in input_headers:
		stddev_dict[head] = [] #Starts off as a python list before converting to a numpy list
	for row in lines:
		rand_dat.append({headers[x]:float(row[x]) for x in range(len(headers))})
		[stddev_dict[input_headers[x]].append(float(row[x])) for x in range(len(input_headers))]

for head in input_headers:
    min_dict[head] = min(stddev_dict[head])
    max_dict[head] = max(stddev_dict[head])
    ranges_dict[head] = (numpy.asarray(stddev_dict[head]).min(),numpy.asarray(stddev_dict[head]).max())
    stddev_dict[head] = numpy.asarray(stddev_dict[head]).std()

stddev_list = [stddev_dict[x] for x in input_headers]


#grids = np.mgrid[[slice(min_dict[x],max_dict[x],10j) for x in input_headers]]

#Convert from indicies to real numbers
conversion_eqs = {x: (lambda y : (y*((max_dict[x]-min_dict[x])/100.0) + min_dict[x])) for x in input_headers}

rand_dat_np = numpy.asarray([[dat_line[x] for x in input_headers] for dat_line in rand_dat])

#mtest = LinearNDInterpolator(rand_dat_np, numpy.asarray([dat_line["droplet_size"] for dat_line in rand_dat]))

rand_dat_normal = np.asarray([(rand_dat_np[:,x]-rand_dat_np[:,x].min())/(rand_dat_np[:,x].max()-rand_dat_np[:,x].min()) for x in range(rand_dat_np.shape[1])])
print(rand_dat_normal.shape)
drop_size_fit = Rbf(*rand_dat_normal, numpy.asarray([dat_line["droplet_size"] for dat_line in rand_dat]),epsilon = 0.001)
gen_rate_fit = Rbf(*rand_dat_normal, numpy.asarray([dat_line["generation_rate"] for dat_line in rand_dat]),epsilon = 0.001)



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


target_vals = (0,0)
def rbf_error(x):
	merror1 = abs(drop_size_fit(*x) - target_vals[0])
	merror2 = abs(gen_rate_fit(*x) - target_vals[1])
	return merror1+merror2

def linearizedError(x):
	distances = []
	for dat_line in rand_dat:
		run_sum = 0
		for i in range(len(x)):
			run_sum+=abs(abs(x[i]-dat_line[input_headers[i]])/(stddev_list[i]))
		distances.append(run_sum)
	return min(distances)

def getClosestPoint(droplet_size_wanted,generation_rate_wanted):
	min_point = {}
	min_val = float("inf")
	for point in rand_dat:
		if abs(point["droplet_size"]-droplet_size_wanted) + abs(point["generation_rate"]-generation_rate_wanted) < min_val:
			min_point = point
			min_val = abs(point["droplet_size"]-droplet_size_wanted) + abs(point["generation_rate"]-generation_rate_wanted)
	return min_point


O1_errors = []
O2_errors = []

O1_nonmod_errors = []
O2_nonmod_errors = []
for i in range(100):
		dummy_inputs = {"orifice_size":random()*250+50,
			"aspect_ratio":random()*2+1,
			"width_ratio":random()*2+2,
			"normalized_orifice_length":random()*8+1,
			"normalized_oil_input_width":random()*2+2,
			"normalized_water_input_width":random()*2+2,
			"capillary_number":random()*(0.2222-0.02)+0.02,
			"flow_rate_ratio":random()*18+2}

		#dummy_inputs = rand_dat[round(random()*100)]
		wanted_vals = equationOutputs(dummy_inputs)
		
		target_vals = wanted_vals

		closest_point = getClosestPoint(wanted_vals[0],wanted_vals[1])
		start_pos = numpy.asarray([(closest_point[x]-ranges_dict[x][0])/(ranges_dict[x][1]-ranges_dict[x][0])  for x in input_headers])

		res = minimize(rbf_error, start_pos, method='SLSQP',bounds = tuple([(0,1) for x in input_headers]))

		results = {input_headers[i]:res["x"][i]*(ranges_dict[input_headers[i]][1]-ranges_dict[input_headers[i]][0])+ranges_dict[input_headers[i]][0] for i in range(len(input_headers))}
		print(results)
		print(equationOutputs(results)[1] - wanted_vals[1])
		# print(dummy_inputs)
		# print(mtest(*[dummy_inputs[x] for x in input_headers]) - wanted_vals[0])
		#
		# min_val = float("inf")
		# min_index = 0
		# for index,val in np.ndenumerate(droplet_size_grid):
		#	if(abs(val - wanted_vals[0])<min_val):
		#		min_index = index
		#		min_val = abs(val-wanted_vals[0])
		# results = {x:conversion_eqs[x](min_index[index]) for index,x in enumerate(input_headers)}

		O1_errors.append( abs(equationOutputs(results)[0] - wanted_vals[0])/wanted_vals[0])
		O2_errors.append( abs(equationOutputs(results)[1] - wanted_vals[1])/wanted_vals[1])
		O1_nonmod_errors.append( abs(equationOutputs(closest_point)[0] - wanted_vals[0])/wanted_vals[0])
		O2_nonmod_errors.append( abs(equationOutputs(closest_point)[1] - wanted_vals[1])/wanted_vals[1])
		

		
print(sum(O1_errors)/len(O1_errors))
print(sum(O1_nonmod_errors)/len(O1_nonmod_errors))
print(sum(O2_errors)/len(O2_errors))
print(sum(O2_nonmod_errors)/len(O2_nonmod_errors))


