"""Generate accuracy data for various DAFD methods

All new reverse prediction methods should be tested with this first.
"""

from tqdm import tqdm
from InterModel import InterModel
from M5PModel import M5PModel
from random import random


def equationOutputs(args):
	"""Returns the *real* outputs (size and rate tuple) of the input args

	This is the equation that our csv file is built from.
	"""

	#Drop Size = (Orifice Size * Aspect Ratio * Width Ratio * Orifice Length * Water Input Width)/(Oil Input Width * Capillary Number * Flow Rate Ratio *10) 
	drop_size = prod([args["orifice_size"],args["aspect_ratio"],args["width_ratio"],args["normalized_orifice_length"],args["normalized_water_input_width"]])/prod([args["normalized_oil_input_width"],args["capillary_number"],args["flow_rate_ratio"],10]);
	

	#Generation Rate = ([Oil Input Width + Aspect Ratio + Width Ratio + Orifice Length + Water Input Width + Flow Rate Ratio] * Capillary Number * 5000) / (Orifice Size) 
	generation_rate = prod([sum([args["aspect_ratio"],args["width_ratio"],args["normalized_orifice_length"],args["normalized_oil_input_width"],args["normalized_water_input_width"],args["flow_rate_ratio"]]),args["capillary_number"],5000]) / args["orifice_size"]

	return drop_size,generation_rate


def prod(args):
	"""Returns the product of an aribitrary list of numbers"""
	product = 1
	for x in args:
		product*=x
	return product


#RBF Interpolation model
drop_size_it_errors = [] 
generation_rate_it_errors = []

#M5P Tree error
drop_size_m5p_errors = []
generation_rate_m5p_errors = []

#Nearest data point errors
drop_size_nonmod_errors = []
generation_rate_nonmod_errors = []

m5p_model = M5PModel()
it_model = InterModel()
for i in tqdm(range(1000)):

		#Make random inputs in the same range as our test data.
		dummy_inputs = {"orifice_size":random()*250+50,
			"aspect_ratio":random()*2+1,
			"width_ratio":random()*2+2,
			"normalized_orifice_length":random()*8+1,
			"normalized_oil_input_width":random()*2+2,
			"normalized_water_input_width":random()*2+2,
			"capillary_number":random()*(0.2222-0.02)+0.02,
			"flow_rate_ratio":random()*18+2}

		#Find out what size and generation rate they produce (this ensures that there is at least one solution)
		drop_size, generation_rate = equationOutputs(dummy_inputs)

		#Make our models guess the input set
		it_results = it_model.interpolate({"droplet_size":drop_size,"generation_rate":generation_rate},{})
		m5p_results = m5p_model.interpolate({"droplet_size":drop_size,"generation_rate":generation_rate},{})
			

		# Find the errors in the returned results
		# |real(i) - f(i)| / f(i) where f is the model we are testing
		drop_size_m5p_errors.append( abs(equationOutputs(m5p_results)[0] - drop_size)/drop_size)
		generation_rate_m5p_errors.append( abs(equationOutputs(m5p_results)[1] - generation_rate)/generation_rate)

		drop_size_it_errors.append( abs(equationOutputs(it_results)[0] - drop_size)/drop_size)
		generation_rate_it_errors.append( abs(equationOutputs(it_results)[1] - generation_rate)/generation_rate)
		
		drop_size_nonmod_errors.append( abs(equationOutputs(it_model.closest_point)[0] - drop_size)/drop_size)
		generation_rate_nonmod_errors.append( abs(equationOutputs(it_model.closest_point)[1] - generation_rate)/generation_rate)

		


#Print out the average error for each model's list of errors
print()
print("drop size interp errors:		"+str(round(sum(drop_size_it_errors)/len(drop_size_it_errors) * 100,4)).zfill(7) + "%")
print("generation rate interp errors:		"+str(round(sum(generation_rate_it_errors)/len(generation_rate_it_errors) * 100,4)).zfill(7) + "%")
print()
print("drop size m5p errors:			"+str(round(sum(drop_size_m5p_errors)/len(drop_size_m5p_errors) * 100,4)).zfill(7) + "%")
print("generation rate m5p errors:		"+str(round(sum(generation_rate_m5p_errors)/len(generation_rate_m5p_errors) * 100,4)).zfill(7) + "%")
print()
print("drop size closest point errors:		"+str(round(sum(drop_size_nonmod_errors)/len(drop_size_nonmod_errors) * 100,4)).zfill(7) + "%")
print("generation rate closest point errors:	"+str(round(sum(generation_rate_nonmod_errors)/len(generation_rate_nonmod_errors) * 100,4)).zfill(7) + "%")


