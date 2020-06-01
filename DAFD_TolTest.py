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
from tol_utils import *
import random as r
import itertools
import time as t
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import *

class ToleranceHelper:
    """This class contains the main functions needed for the tolerance study."""
    features_normalized = {}
    features_denormalized = {}
    tolerance = None
    tol_df = None

    flow_heatmap_size = None
    flow_heatmap_gen = None
    flow_grid_size = None

    feature_heatmaps = None
    feature_grid_size = None

    di = None

    def __init__(self, feature_inputs, tolerance=10, feature_grid_size = 11, flow_grid_size = 51):
        self.inputs_normalized = feature_inputs
        self.inputs_denormalized = self.denormalize_features(self.inputs_normalized)
        self.tolerance = tolerance/100
        self.tol_df = self.make_tol_df(self.inputs_normalized, self.tolerance)
        self.feature_grid_size = feature_grid_size
        self.flow_grid_size = flow_grid_size
        self.di = DAFD_Interface


    def make_tol_df(self, features, tol):
        max_feat = {key: (features[key] + tol * features[key]) for key in features.keys()}
        min_feat = {key: (features[key] - tol * features[key]) for key in features.keys()}
        return pd.DataFrame(min_feat, features, max_feat)


    def make_flow_heatmaps(self, oil_range, water_range):
        oil = np.around(make_grid_range(pd.Series(oil_range), self.flow_grid_size), 2)
        water = np.around(make_grid_range(pd.Series(water_range), self.flow_grid_size), 2)

        grid_dict = {"oil_flow": oil, "water_flow": water}
        self.flow_heatmap_size = self.generate_heatmap_data(grid_dict, di, "droplet_size", percent=False)
        self.flow_heatmap_gen = self.generate_heatmap_data(grid_dict, di, "generation_rate", percent=False)

    def make_feature_heatmaps(self, input_data, pc_s, pc_g, tol_df, di, grid_size=11):
        tol_df_shuff = self.tol_df[[col for col in tol_df.columns if col != pc_s] + [pc_s]]
        tol_df_shuff = tol_df_shuff[[col for col in tol_df.columns if col != pc_g] + [pc_g]]

        heatmap_data_s = self.heatmap_loop(, pc_s, tol_df_shuff, di, "droplet_size", grid_size)
        heatmap_data_g = self.heatmap_loop(self.features_normalized, pc_g, tol_df, di, "generation_rate", grid_size)
        return heatmap_data_s, heatmap_data_g


    def heatmap_loop(self, pc, tol_df, output, grid_size):
        pc_range = make_grid_range(tol_df.loc[:, pc], grid_size)
        features = [feat for feat in tol_df.columns if feat != pc]
        heatmap_data = []
        for feat in features:
            feat_range = make_grid_range(tol_df.loc[:, feat], grid_size)
            grid_dict = {pc: pc_range, feat: feat_range}
            heatmap_data.append(self.generate_heatmap_data(grid_dict, output))
        return heatmap_data


    def generate_heatmap_data(self, grid_dict, output, percent=True):
        key_names = list(grid_dict.keys())
        pts, grid = make_sample_grid(self.features_denormalized, grid_dict)
        grid_measure = [self.di.runForward(self.renormalize_features(pt)) for pt in grid]
        outputs = [out[output] for out in grid_measure]
        for i, pt in enumerate(pts):
            pt.append(outputs[i])
        heat_df = pd.DataFrame(pts, columns=[key_names[0], key_names[1], output])
        if percent:
            heat_df.loc[:, key_names[0]] = pct_change(heat_df.loc[:, key_names[0]],
                                                      self.features_denormalized[key_names[0]]).astype(int)
            heat_df.loc[:, key_names[1]] = pct_change(heat_df.loc[:, key_names[1]],
                                                      self.features_denormalized[key_names[1]]).astype(int)
            base_out = di.runForward(self.features_denormalized)[output]
            heat_df.loc[:, output] = pct_change(heat_df.loc[:, output], base_out)
        heat_pivot = heat_df.pivot(index=key_names[1], columns=key_names[0], values=output)
        return heat_pivot[::-1]


    def denormalize_features(self, features):
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

        ret_dict = {
            "orifice_size": Or
            "depth": channel_height
            "outlet_width": outlet_channel_width
            "orifice_length": orifice_length
            "water_inlet": water_inlet_width
            "oil_inlet": oil_inlet
            "oil_flow": oil_flow_rate_ml_per_hour
            "water_flow": water_flow_rate_ul_per_min
        }
        return ret_dict


    def renormalize_features(self, features):
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
        ret_dict = {
            "orifice_size": Or
            "aspect_ratio": As
            "expansion_ratio": Exp
            "normalized_orifice_length": norm_Ol
            "normalized_water_inlet": norm_Wi
            "normalized_oil_inlet": norm_Oi
            "flow_rate_ratio": Q_ratio
            "capillary_number":  round(Ca_num, 5)
            }
        return ret_dict















if __name__ == "__main__":
    test_features = {
        "orifice_size": 150,
        "aspect_ratio": 1,
        "expansion_ratio": 2,
        "normalized_orifice_length": 2,
        "normalized_water_inlet": 2,
        "normalized_oil_inlet": 2,
        "flow_rate_ratio": 6,
        "capillary_number": 0.05
    }
    tolerance = 10
    sobol_samples = 100
    grid_size = 11

    di = DAFD_Interface()
    tol_df = sobol_prep(test_features, tolerance)
    results, si_size, si_gen = sobol_analyis(tol_df, sobol_samples, di, calc_second_order=True)
    fig = plot_sobol_results(si_size, si_gen, tol_df.columns)
    plt.savefig("test2.png")

    pc_s = get_principal_feature(si_size, tol_df.columns)
    pc_g = get_principal_feature(si_gen, tol_df.columns)
    hm_s, hm_g = heatmap_workflow(test_features, pc_s, pc_g, tol_df,di, grid_size=grid_size)
    fig = plot_heatmaps(hm_s, hm_g)
    plt.savefig("test.png")

    feat_denormed = denormalize_features(test_features)
    oil_range = [0.1, 2*feat_denormed["oil_flow"]]
    water_range = [0.5, 2*feat_denormed["water_flow"]]
    flow_grid = 51
    size_df, rate_df = flow_heatmaps(oil_range, water_range, flow_grid)
    fig = plot_flow_heatmaps(size_df, rate_df, test_features)
    plt.savefig("test_3.png")




    #TODO: Integrate into DAFD Workflow (cmd first, then think about GUI)
    #TODO: generate PDF
    #TODO: Get a standard water/oil flow heatmap (across all ranges). DONE GET INTO WORKFLOW
    #TODO: Add in any other images that are needed
    #TODO: Just clean up the entire system, make it pythonic
    #TODO: Eliminate unnecessary functions from the system