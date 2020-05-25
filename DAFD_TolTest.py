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

from helper_scripts.ModelHelper import ModelHelper
from bin.DAFD_Interface import DAFD_Interface
import random as r

def run_analysis(features, tolerance):
    tol = tolerance/100 # Assuming that tolerance is given in percent, format
    feat_denorm = denormalize_features(features)
    max_feat, min_feat = make_tol_dicts(feat_denorm, tol)
    print(max_feat)
    print(feat_denorm)
    print(min_feat)



def random_features(di):
    headers = di.MH.get_instance().input_headers
    ranges = di.ranges_dict
    feature_set = {head: (round(r.random()*(ranges[head][1] - ranges[head][0])+ranges[head][0], 2)) for head in headers}
    return feature_set


def make_tol_dicts(features, tol):
    max_feat = {key: (features[key] + tol*features[key]) for key in features.keys()}
    min_feat = {key: (features[key] - tol*features[key]) for key in features.keys()}
    return max_feat, min_feat


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
    water_flow_rate_m3_per_s = water_flow_rate_ul_per_min * 1e-9 / 60

    ret_dict = {}
    ret_dict["orifice_size"] = orifice_length
    ret_dict["depth"] = channel_height
    ret_dict["outlet_width"] = outlet_channel_width
    ret_dict["Orifice_length"] = outlet_channel_width
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
    run_analysis(test_features, 10)