'''
This script will be a placeholder for integrating a tolerance tester into the DAFD workflow.

Inputs: User parameters, and a set tolerance

Algorithm:
- Take in input specifications and tolerance, calculate upper and lower bounds
- With this, mash them together and calculate all combinations (there's a set version for this somewhere)
- With these combinations, Brute force DAFD evaluating every single one
- DATA VIZ NEEDED NEXT:
-- with this, work on putting everything together and effectively representing all of the data points
'''

#from helper_scripts.ModelHelper import ModelHelper
from bin.DAFD_Interface import DAFD_Interface
import random as r
import itertools
import time as t
import pandas as pd
import matplotlib.pyplot as plt

def run_analysis(features, tolerance, di):
    tol = tolerance/100 # Assuming that tolerance is given in percent, format
    feat_denorm = denormalize_features(features)
    max_feat, min_feat = make_tol_dicts(feat_denorm, tol)
    combos = all_combos(feat_denorm, min_feat, max_feat)
    combos_normed = [renormalize_features(option) for option in combos]
    start = t.time()
    outputs = [di.runForward(option) for option in combos_normed]
    e1 = t.time()
    print(e1-start)
    return outputs


def sobol_prep(features, tolerance):
    tol = tolerance/100 # Assuming that tolerance is given in percent, format
    feat_denorm = denormalize_features(features)
    max_feat, min_feat = make_tol_dicts(feat_denorm, tol)
    tol_df = pd.DataFrame([min_feat, feat_denorm, max_feat])
    return tol_df

def plot_results(outputs, original, tolerance):
    plt.scatter([i["droplet_size"] for i in outputs], [i["generation_rate"] for i in outputs])
    plt.scatter(original["droplet_size"], original["generation_rate"])
    plt.xlabel("Droplet Size")
    plt.ylabel("Generation Rate")
    plt.title("All possible outputs with tolerance of %d percent" % tolerance)
    plt.legend(["Results with Tolerance", "User Input"])
    plt.show()

def all_combos(features, min_feat, max_feat):
    feat_op = []
    for key in features.keys():
        feat_op.append(
            [min_feat[key], features[key], max_feat[key]]
        )
    combo_Iter = itertools.product(feat_op[0], feat_op[1], feat_op[2], feat_op[3],
                               feat_op[4], feat_op[5], feat_op[6], feat_op[7])
    combos = []
    for option in combo_Iter:
        combos.append({key:option[i] for i,key in enumerate(features.keys())})
    return combos


def random_features(di):
    headers = di.MH.get_instance().input_headers
    ranges = di.ranges_dict
    feature_set = {head: (round(r.random()*(ranges[head][1] - ranges[head][0])+ranges[head][0], 2)) for head in headers}
    return feature_set


def make_tol_dicts(features, tol):
    max_feat = {key: (features[key] + tol*features[key]) for key in features.keys()}
    min_feat = {key: (features[key] - tol*features[key]) for key in features.keys()}
    return max_feat, min_feat

def renormalize_features(features):
    channel_height = features["depth"]
    outlet_channel_width = features["outlet_width"]
    orifice_length = features["orifice_length"]
    water_inlet_width = features["water_inlet"]
    oil_inlet = features["oil_inlet"]
    oil_flow_rate_ml_per_hour = features["oil_flow"]
    water_flow_rate_ul_per_min = features["water_flow"]

    Or = features["orifice_size"]
    As = channel_height/Or
    Exp = outlet_channel_width/Or
    norm_Ol = orifice_length/Or
    norm_Wi = water_inlet_width/Or
    norm_Oi = oil_inlet/Or

    Q_ratio = oil_flow_rate_ml_per_hour / (water_flow_rate_ul_per_min*(60/1000))

    Ca_num = ((0.0572*water_inlet_width * 1e-6*(oil_flow_rate_ml_per_hour/(3600*1e6))) / \
             (0.005 * channel_height * 1e-6 * oil_inlet * 1e-6)) * (1/(Or * 1e-6) - 1/(2*oil_inlet*1e-6))

    ret_dict = {}
    ret_dict["orifice_size"] = Or
    ret_dict["aspect_ratio"] = As
    ret_dict["expansion_ratio"] = Exp
    ret_dict["normalized_orifice_length"] = norm_Ol
    ret_dict["normalized_water_inlet"] = norm_Wi
    ret_dict["normalized_oil_inlet"] = norm_Oi
    ret_dict["flow_rate_ratio"] = Q_ratio
    ret_dict["capillary_number"] = round(Ca_num, 5)
    return ret_dict

def denormalize_features(features):
    Or = features["orifice_size"]
    As = features["aspect_ratio"]
    Exp = features["expansion_ratio"]
    norm_Ol = features["normalized_orifice_length"]
    norm_Wi = features["normalized_water_inlet"]
    norm_Oi = features["normalized_oil_inlet"]
    Q_ratio = features["flow_rate_ratio"]
    Ca_num = features["capillary_number"]

    channel_height = Or * As
    outlet_channel_width = Or * Exp
    orifice_length = Or * norm_Ol
    water_inlet_width = Or * norm_Wi
    oil_inlet = Or * norm_Oi
    oil_flow_rate = (Ca_num * 0.005 * channel_height * oil_inlet * 1e-12) / \
                    (0.0572 * ((water_inlet_width * 1e-6)) * (
                                (1 / (Or * 1e-6)) - (1 / (2 * oil_inlet * 1e-6))))
    oil_flow_rate_ml_per_hour = oil_flow_rate * 3600 * 1e6
    water_flow_rate = oil_flow_rate_ml_per_hour / Q_ratio
    water_flow_rate_ul_per_min = water_flow_rate * 1000 / 60

    ret_dict = {}
    ret_dict["orifice_size"] = Or
    ret_dict["depth"] = channel_height
    ret_dict["outlet_width"] = outlet_channel_width
    ret_dict["orifice_length"] = orifice_length
    ret_dict["water_inlet"] = water_inlet_width
    ret_dict["oil_inlet"] = oil_inlet
    ret_dict["oil_flow"] = oil_flow_rate_ml_per_hour
    ret_dict["water_flow"] = water_flow_rate_ul_per_min
    return ret_dict

if __name__ == "__main__":
    test_features = {
        "orifice_size": 100,
        "aspect_ratio": 2,
        "expansion_ratio": 4,
        "normalized_orifice_length": 2,
        "normalized_water_inlet": 3,
        "normalized_oil_inlet": 4,
        "flow_rate_ratio": 10,
        "capillary_number": 0.05
    }
    tolerance = 10
    di = DAFD_Interface()
    outputs = run_analysis(test_features, tolerance, di)
    plots = plot_results(outputs, di.runForward(test_features), tolerance)