#!/usr/bin/python3

import os
from bin.DAFD_Interface import DAFD_Interface
from tolerance_study.TolHelper import TolHelper

di = DAFD_Interface()

constraints = {}
desired_vals = {}
features = {}

stage = 0
tolerance_test = False
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
		elif line == "TOLERANCE":
			tolerance_test=True
			continue

		if tolerance_test:
			tolerance = float(line.split("=")[1])
			continue

		if stage == 0:
			param_name = line.split("=")[0]
			param_pair = line.split("=")[1].split(":")
			if param_name=="regime":
				wanted_constraint=float(param_pair[0])
				constraints[param_name] = wanted_constraint
			else:
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
	fwd_results = di.runForward(features)

	result_str = "BEGIN:"

	for x in di.MH.get_instance().output_headers:
		result_str += str(fwd_results[x]) + "|"
	result_str += str(fwd_results["regime"]) + "|"
	result_str += str(fwd_results["oil_rate"]) + "|"
	result_str += str(fwd_results["water_rate"]) + "|"
	result_str += str(fwd_results["inferred_droplet_size"]) + "|"
	print(result_str)

else:
	rev_results = di.runInterp(desired_vals, constraints)
	fwd_results = di.runForward(rev_results)

	print(rev_results)
	print(fwd_results)


	result_str = "BEGIN:"
	for x in di.MH.get_instance().input_headers:
		result_str += str(rev_results[x]) + "|"

	result_str += str(rev_results["point_source"]) + "|"

	for x in di.MH.get_instance().output_headers:
		result_str += str(fwd_results[x]) + "|"
	result_str += str(fwd_results["regime"]) + "|"
	result_str += str(fwd_results["oil_rate"]) + "|"
	result_str += str(fwd_results["water_rate"]) + "|"
	result_str += str(fwd_results["inferred_droplet_size"]) + "|"

	print(result_str)

if tolerance_test:
	if stage == 1:
		tol_features = rev_results.copy()
		del tol_features["point_source"]
	else:
		tol_features = features
	TH = TolHelper(tol_features, di=di, tolerance=tolerance)
	TH.run_all()
	TH.plot_all()
	TH.generate_report()