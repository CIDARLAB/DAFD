"""
Created David McIntyre
Utils for robustness & versatility study
"""
import numpy as np
from DAFD.bin.DAFD_Interface import DAFD_Interface
import pandas as pd
from DAFD.tolerance_study.tol_utils import make_sample_grid
di = DAFD_Interface()

# Method returns linspace of specific range given number of divisions
def make_sweep_range(input_range, sweep_size):
    return np.linspace(np.min(input_range), np.max(input_range), sweep_size)


# Method used for sweeping the entire design space and (then) running DAFD on it.
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
def sweep_results(chip_design, ca_range=[], q_range=[], sweep_size=25, jet_drop=False):
    if bool(ca_range):
        ca_range = make_sweep_range(ca_range,sweep_size)
    else:
        ca_range = np.concatenate([np.arange(.05,.11,step=.01), np.linspace(.161111,1.05,9)])
    if bool(q_range):
        q_range = make_sweep_range(q_range, sweep_size)
    else:
        q_range = np.linspace(2,22,10)
    grid_dict = {
        "flow_rate_ratio": q_range,
        "capillary_number": ca_range
    }
    pts, grid = make_sample_grid(chip_design, grid_dict)
    grid_measure = [di.runForward(pt) for pt in grid]
    if jet_drop: # Drop jetting regime points if needed
        grid, grid_measure = drop_jetting_points(grid_measure, grid)
    sizes = [out["droplet_size"] for out in grid_measure]
    rates = [out["generation_rate"] for out in grid_measure]
    out = pd.DataFrame(grid)
    out["size"] = sizes
    out["rate"] = rates
    return sizes, rates, out


# Method to remove points in jetting regime
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


# Denormalizes features into raw parameters
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


# Denormalizes features into parameters compatible with DAFD forward model
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