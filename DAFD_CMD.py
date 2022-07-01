#!/usr/bin/python3
import os
from DAFD.bin.DAFD_Interface import DAFD_Interface
from DAFD.tolerance_study.TolHelper import TolHelper
from DAFD.metrics_study.MetricHelper import MetricHelper
import pandas as pd

di = DAFD_Interface()

constraints = {}
desired_vals = {}
features = {}

stage = -1
tolerance_test = False
flow_stability=False
versatility=False
ranking = ""
with open(os.path.dirname(os.path.abspath(__file__)) + "/" + "DAFD/cmd_inputs.txt","r") as f:
	for line in f:
		line = line.strip()
		if line == "":
			continue
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
		elif line == "VERSATILITY":
			versatility=True
			ranking="all_overall_score"
		elif line == "FLOW_STABILITY":
			flow_stability = True
			ranking="flow_stability"
		elif "RANK_BY" in line:
			ranking=line.split("=")[1]

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
			param_name = line.split("=")[0]

		if stage == 2:
			param_name = line.split("=")[0]
			param_val = float(line.split("=")[1])
			features[param_name] = param_val

	if "regime" in constraints.keys() and "score" in ranking:
		if constraints["regime"] == 1:
			replace = "dripping"
		elif constraints["regime"] == 2:
			replace = "jetting"
		split = ranking.split("_")
		ranking = replace + "_" + split[1] + "_" + split[2]


if stage == 2:
	fwd_results = di.runForward(features)

	result_str = "BEGIN:"

	for x in di.MH.get_instance().output_headers:
		result_str += str(fwd_results[x]) + "|"
	result_str += str(fwd_results["regime"]) + "|"
	result_str += str(fwd_results["oil_rate"]) + "|"
	result_str += str(fwd_results["water_rate"]) + "|"
	result_str += str(fwd_results["inferred_droplet_size"]) + "|"
	if flow_stability:
		MetHelper = MetricHelper(fwd_results)
		MetHelper.run_all_flow_stability()
		fwd_results["flow_stability"] = MetHelper.point_flow_stability
	if versatility:
		try:
			MetHelper.run_all_versatility()
		except:
			MetHelper = MetricHelper(fwd_results)
			MetHelper.run_all_versatility()
		fwd_results.update(MetHelper.versatility_results)


	print(result_str)

else:
	if flow_stability or versatility:
		results = di.runInterpQM(desired_vals, constraints)
		for i, result in enumerate(results):
			MetHelper = MetricHelper(result)
			if flow_stability:
				MetHelper.run_all_flow_stability()
				results[i]["flow_stability"] = MetHelper.point_flow_stability
			if versatility:
				MetHelper.run_all_versatility()
				results[i].update(MetHelper.versatility_results)
			results.update(di.runForward(result))
		results_df = pd.DataFrame(results)
		MetHelper.generate_report(flow_stability=flow_stability, versatility=versatility)
		rev_results = results_df.sort_values(by=ranking)[0]
		# TODO: PICK THE HIGHEST FLOW STABILITY AND INTEGRATE IT IN WITH THE REST OF THE TIMELINE
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

flow_stability_test = True
if flow_stability or versatility:
	from DAFD.metrics_study import metric_utils
	if stage == 1: # Performance prediction first
		# Get the base device design features
		MetricH = MetricHelper(rev_results)
		MetricH.run_all_flow_stability() # run flow stability study on the chip
		MetricH.plot_all() #TODO: MAKE ALL PLOTS NEEDED FOR THE REPORT
		print("DONE")
		MetricH.generate_report(filepath="PLACEHOLDER_FSstudy.csv") #TODO: FIGURE OUT WHAT AN OUTPUT REPORT WOULD LOOK LIKE