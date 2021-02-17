"""
Created David McIntyre, 12/15/20
Create utils for robustness & versatility study
"""
import numpy as np
import os
from DAFD.bin.DAFD_Interface import DAFD_Interface
from DAFD.tolerance_study.TolHelper import TolHelper
import pandas as pd
from DAFD.tolerance_study.tol_utils import make_sample_grid
di = DAFD_Interface()

def make_sweep_range(input_range, sweep_size):
    return np.linspace(np.min(input_range), np.max(input_range), sweep_size)

"""
Method used for sweeping the entire design space and (then) running DAFD on it.
This is going to be nested, in the way that we
    (1) Make a library of different device chips (this setup 
"""
def generate_design_space_grid(min_all, max_all, increment=.5):
    grid_dict = {}
    for key in min_all.keys():
        if key == "orifice_size":
            grid_dict[key] = np.arange(min_all[key], max_all[key] + 25, 25)
        else:
            grid_dict[key] = np.arange(min_all[key], max_all[key] + increment, increment)
    pts, grid = make_sample_grid({}, grid_dict, entire_chip=True)
    return grid

# Method used for versatility score
def sweep_results(chip_design, ca_range=[.05, .25], q_range=[2, 22], sweep_size=25, jet_drop=False):
    grid_dict = {
        "flow_rate_ratio": make_sweep_range(q_range,sweep_size),
        "capillary_number": make_sweep_range(ca_range, sweep_size)
    }
    pts, grid = make_sample_grid(chip_design, grid_dict)
    grid_measure = [di.runForward(pt) for pt in grid]
    if jet_drop:
        grid, grid_measure = drop_jetting_points(grid_measure, grid)
    sizes = [out["droplet_size"] for out in grid_measure]
    rates = [out["generation_rate"] for out in grid_measure]
    out = pd.DataFrame(grid)
    out["size"] = sizes
    out["rate"] = rates
    return sizes, rates, out


def drop_jetting_points(grid_measure, grid):
    track_r1 = []
    track_ir1 = []
    track_ir2 = []
    track_r2 = []
    drop_counter = 0
    for i, point in enumerate(grid_measure):
        if point["regime"] == 2:
            drop_counter += 1
            track_ir2.append(i)
            track_r2.append(grid[i]["capillary_number"])
        else:
            track_ir1.append(i)
            track_r1.append(grid[i]["capillary_number"])

    grid_measure = [pt for i, pt in enumerate(grid_measure) if i not in track_ir2]
    grid = [pt for i, pt in enumerate(grid) if i not in track_ir2]
    return grid, grid_measure


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`
    Taken from Stack Overflow

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def update_constant_flows(init_features, to_update):
    init_features_denormed = denormalize_features(init_features)
    new_features = init_features.copy()
    new_features.update(to_update)
    new_features_denormed = denormalize_features(new_features)
    new_features_denormed.update({"oil_flow": init_features_denormed["oil_flow"],
                                  "water_flow": init_features_denormed["water_flow"]})
    return renormalize_features(new_features_denormed)

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

    ret_dict = {
        "orifice_size": Or,
        "depth": channel_height,
        "outlet_width": outlet_channel_width,
        "orifice_length": orifice_length,
        "water_inlet": water_inlet_width,
        "oil_inlet": oil_inlet,
        "oil_flow": oil_flow_rate_ml_per_hour,
        "water_flow": water_flow_rate_ul_per_min
    }
    return ret_dict


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
    ret_dict = {
        "orifice_size": Or,
        "aspect_ratio": As,
        "expansion_ratio": Exp,
        "normalized_orifice_length": norm_Ol,
        "normalized_water_inlet": norm_Wi,
        "normalized_oil_inlet": norm_Oi,
        "flow_rate_ratio": Q_ratio,
        "capillary_number":  round(Ca_num, 5)
        }
    return ret_dict


def calculate_robust_score(features, sweep_size=5, tol=10):
    initial_outputs = di.runForward(features)
    scores = []
    size_score = []
    rate_score = []
    features_to_change = features.copy()
    del features_to_change["flow_rate_ratio"]
    del features_to_change["capillary_number"]
    for feature in features_to_change.keys():
        # Make grid_dict with tol of 10% in  a SINGLE dimension
        sweep_range = make_sweep_range([features[feature]*(1-tol/100), features[feature]*(1+tol/100)], sweep_size)
        grid = []
        for i in range(len(sweep_range)):
            grid.append(update_constant_flows(features, {feature: sweep_range[i]}))
        grid_measure = [di.runForward(pt) for pt in grid]
        sizes = [out["droplet_size"] for out in grid_measure]
        size_range = (np.max(sizes) - np.min(sizes))/initial_outputs["droplet_size"]

        rates = [out["generation_rate"] for out in grid_measure]
        rate_range = (np.max(rates) - np.min(rates)) / initial_outputs["generation_rate"]
        sz_score = np.log10(1/size_range)
        r_score = np.log10(1/rate_range)
        size_score.append(sz_score)
        rate_score.append(r_score)
        scores.append(np.mean([sz_score, r_score])) # TODO: update this with a beter score (doesn't bias one or the other)

    return np.mean(scores), np.mean(size_score), np.mean(rate_score)


def calculate_versatility_score(sizes, rates):
    # Find max and min values
    # Normalize by maximum values of entire dataset (so like model helper)
    # Just calculate root sum of each
    range_size = np.max(sizes) - np.min(sizes)
    range_rate = np.max(rates) - np.min(rates)
    score = np.sqrt(range_rate**2 + range_size**2)/2
    return score

