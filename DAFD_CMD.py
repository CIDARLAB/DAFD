#!/usr/bin/python3

import sys
import os
from bin.DAFD_Interface import DAFD_Interface

di = DAFD_Interface()

constraints = {}
desired_vals = {}
features = {}

stage = 0
with open(os.path.dirname(os.path.abspath(__file__)) + "/" + "cmd_inputs.txt","r") as f:
	for line in f:
		line = line.strip()
		if line == "CONSTRAINTS":
			stage=0
			continue
		elif line == "DESIRED_VALS":
			stage=1
			continue
		elif line == "FORWARD":
			stage=2
			continue

		if stage == 0:
			param_name = line.split("=")[0]
			param_pair = line.split("=")[1].split(":")
			wanted_constraint = (float(param_pair[0]), float(param_pair[1]))
			constraints[param_name] = wanted_constraint

		if stage == 1:
			param_name = line.split("=")[0]
			param_val = float(line.split("=")[1])
			desired_vals[param_name] = param_val

		if stage == 2:
			param_name = line.split("=")[0]
			param_val = float(line.split("=")[1])
			features[param_name] = param_val


if stage == 2:
	results = di.runForward(features)
else:
	results = di.runInterp(desired_vals, constraints)

result_str = ""
for x in di.MH.get_instance().input_headers:
	result_str += str(results[x]) + "|"
print(result_str)
new_results = di.runForward(results)
print("forward_model="+str(new_results))
