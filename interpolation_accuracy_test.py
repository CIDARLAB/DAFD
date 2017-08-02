from tqdm import tqdm
from InterModel import InterModel
from random import random


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


