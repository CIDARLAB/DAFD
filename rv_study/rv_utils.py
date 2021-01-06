"""
Created David McIntyre, 12/15/20
Create utils for robustness & versatility study
"""
import numpy as np
import os
from bin.DAFD_Interface import DAFD_Interface
from tolerance_study.TolHelper import TolHelper
from tolerance_study.tol_utils import make_sample_grid
di = DAFD_Interface()

def make_sweep_range(input_range, sweep_size):
    return np.linspace(np.min(input_range), np.max(input_range), sweep_size)


# Method used for versatility score
def sweep_results(chip_design, ca_range=[.05, .25], q_range=[2, 22], sweep_size=25, jet_drop=False):

    grid_dict = {
        "flow_rate_ratio": make_sweep_range(q_range, sweep_size),
        "capillary_number": make_sweep_range(ca_range, sweep_size)
    }
    pts, grid = make_sample_grid(chip_design, grid_dict)
    grid_measure = [di.runForward(pt) for pt in grid]
    if jet_drop:
        drop_counter = 0
        for i, point in enumerate(grid_measure):
            if point["regime"] == 2:
                del grid_measure[i]
                drop_counter += 1
        print("Dropped %d points" % drop_counter)
    sizes = [out["droplet_size"] for out in grid_measure]
    rates = [out["generation_rate"] for out in grid_measure]
    return sizes, rates


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

def robust_score(features, sweep_size=25, tol=10):
    score_dict = {}
    for feature in features.keys():
        # Make grid_dict with tol of 10%
        sweep_range = make_sweep_range([features[feature]*.9, features[feature]*1.1], sweep_size)
        grid = []
        for i in range(len(sweep_range)):
            copy = features.copy()
            copy.update({feature: sweep_range[i]})
            grid.append(copy)
        grid_measure = [di.runForward(pt) for pt in grid]
        sizes = [out["droplet_size"] for out in grid_measure]
        size_diff = (np.max(sizes) - np.min(sizes)) /di.runForward(features)["droplet_size"]

        rates = [out["generation_rate"] for out in grid_measure]
        rate_diff = (np.max(rates) - np.min(rates)) /di.runForward(features)["droplet_size"]
        score = size_diff + rate_diff #TODO: Look into best way to change this
        score_dict[feature] = score
    return score_dict


def calculate_versatility_score(sizes, rates):
    # Find max and min values
    # Normalize by maximum values of entire dataset (so like model helper)
    # Just calculate root sum of each
    range_size = np.max(sizes) - np.min(sizes)
    range_rate = np.max(rates) - np.min(rates)
    score = np.sqrt(range_rate**2 + range_size**2)/2 #I dont like this
    #TODO: Think about accounting for spread of the distribution. min/max is sensitive to outliers
    #Maybe take the covariance matrix of the setup? Determinant of the cov matrix?
    return score

