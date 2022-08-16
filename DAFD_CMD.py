#!/usr/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
sort_by = None
flow_stability = False
versatility = False

with open(os.path.dirname(os.path.abspath(__file__)) + "/" + "DAFD/cmd_inputs.txt","r") as f:
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
			try:
				tolerance = float(line.split("=")[1])
			except:
				tolerance = 10
			continue
		elif line == "FLOW_STABILITY":
			flow_stability=True
			continue
		elif line == "VERSATILITY":
			versatility=True
			continue
		elif line == "SORT_BY":
			sort_by = line.split("=")[1]
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

if flow_stability or versatility:
	if sort_by is None:
		if flow_stability:
			sort_by = "flow_stability"
		else:
			sort_by = "overall_versatility"
	reg_str = ""
	if "versatility" in sort_by:
		try:
			if constraints["regime"] == 1:
				reg_str = "dripping"
			else:
				reg_str = "jetting"
		except:
			reg_str = "all"
		sort_by = reg_str + "_" + sort_by.split("_")[0] + "_" + "score"



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
	if flow_stability or versatility:
		results = features.copy()
		results.update(fwd_results)
		if fwd_results["regime"] == 1:
			reg_str = "Dripping"
		else:
			reg_str = "Jetting"
		MetHelper = MetricHelper(results, di=di)
		MetHelper.run_all_flow_stability()
		MetHelper.run_all_versatility()
		results.update(MetHelper.versatility_results)
		results.update({"flow_stability":MetHelper.point_flow_stability})
		report_info = {
			"regime": reg_str,
			"results_df": pd.DataFrame([results]),
			"sort_by": sort_by
		}
		report_info["feature_denormalized"] = MetHelper.features_denormalized
		MetHelper.generate_report(report_info)

else:
	if flow_stability or versatility:
		results = di.runInterpQM(desired_vals, constraints.copy())
		for i, result in enumerate(results):
			MetHelper = MetricHelper(result, di=di)
			MetHelper.run_all_flow_stability()
			results[i]["flow_stability"] = MetHelper.point_flow_stability
			MetHelper.run_all_versatility()
			results[i].update(MetHelper.versatility_results)
			results[i].update(di.runForward(result))
		results_df = pd.DataFrame(results)

		results_df.sort_values(by=sort_by, ascending=False, inplace=True)
		report_info = {
			"regime": reg_str,
			"results_df": results_df,
			"sort_by": sort_by
		}
		MetHelper = MetricHelper(results_df.to_dict(orient="records")[0], di=di)
		MetHelper.run_all_flow_stability()
		MetHelper.run_all_versatility()
		report_info["feature_denormalized"] = MetHelper.features_denormalized

		rev_results = results_df.to_dict(orient="records")[0]
		fwd_results = di.runForward(rev_results)

		import datetime
		date = datetime.datetime.today().isoformat()[:16]
		size = int(fwd_results["droplet_size"])
		rate = int(rev_results["generation_rate"])
		filepath = f"{date}_{size}um_{rate}Hz.csv"
		filepath = filepath.replace(":", "_")
		results_df.to_csv(filepath)
		results_df.to_csv(filepath)

		MetHelper.generate_report(report_info)

	else:
		rev_results = di.runInterp(desired_vals, constraints)
		fwd_results = di.runForward(rev_results)

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