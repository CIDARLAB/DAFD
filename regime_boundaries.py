import pandas as pd
import numpy as np
from tqdm import tqdm
from DAFD.metrics_study.metric_utils import denormalize_features

def get_adjacent_points(params, base_idxs, flow=True):
    adjacent_pts = []
    base_vals = {key:params[key][base_idxs[key]] for key in params}
    if flow:
        params_iter = ["capillary_number", "flow_rate_ratio"]
    else:
        params_iter = params.keys()

    for key in params_iter:
        idx = base_idxs[key]
        if idx != 0:
            new_vals = base_vals.copy()
            new_vals[key] = params[key][base_idxs[key] -1]
            adjacent_pts.append(new_vals)
        if idx != len(params[key])-1:
            new_vals = base_vals.copy()
            new_vals[key] = params[key][base_idxs[key] + 1]
            adjacent_pts.append(new_vals)
    return adjacent_pts

def compare_regimes(base_regime, adj_pts, comp, regimes):
    boundary = 0
    for pt in adj_pts:
        adj_index = comp[(pt["capillary_number"], pt["flow_rate_ratio"])].index(pt)
        adj_regime = regimes[(pt["capillary_number"], pt["flow_rate_ratio"])][adj_index]["regime"]
        if adj_regime != base_regime:
            boundary = 1
            return boundary
    return boundary

if __name__ == "__main__":
    data = pd.read_csv("data/20220609_FSLFS_FINERESULTS.csv")

    params = {
        "orifice_size": np.sort(data.orifice_size.unique()),
        "aspect_ratio": np.sort(data.aspect_ratio.unique()),
        "normalized_oil_inlet": np.sort(data.normalized_oil_inlet.unique()),
        "normalized_water_inlet": np.sort(data.normalized_water_inlet.unique()),
        "normalized_orifice_length": np.sort(data.normalized_orifice_length.unique()),
        "expansion_ratio": np.sort(data.expansion_ratio.unique()),
        "capillary_number": np.sort(data.capillary_number.unique()),
        "flow_rate_ratio": np.sort(data.flow_rate_ratio.unique())
    }
    gr = data[list(params.keys())].groupby(["capillary_number", "flow_rate_ratio"])
    gr2 = data[["regime", "capillary_number", "flow_rate_ratio"]].groupby(["capillary_number", "flow_rate_ratio"])
    keys = gr.groups.keys()
    grouped_data = {}
    grouped_regimes = {}
    for key in keys:
        grouped_data[key] = gr.get_group(key).to_dict(orient="record")
        grouped_regimes[key] = gr2.get_group(key).to_dict(orient="record")

    boundary = 0
    for i,pt in tqdm(data.iterrows()):
        base_regime = int(pt.regime)
        # get key-idx pairs from the find fnction of params
        base_param_idxs = {key: np.argwhere(params[key]==pt[key]).ravel()[0] for key in params.keys()}
        # using this, generate library of param combinations that need to be tested (+/-1 step in param directions unless on edge)
        adj_pts = get_adjacent_points(params, base_param_idxs, flow=True)
        boundary = compare_regimes(base_regime, adj_pts, grouped_data, grouped_regimes)
        data.loc[i,"boundary"] = boundary

        #denormalize features for future analysis
        denormed = denormalize_features(pt.to_dict())
        data.loc[i,"water_flow"] = denormed["water_flow"]
        data.loc[i, "oil_flow"] = denormed["oil_flow"]

        if i % 500000 == 0 and i > 1:
            print("Saving Checkpoint: at chip %d" % i)
            data.to_csv("data/20220429/20220517_data_%d.csv" % i)

    data.to_csv("20220609_FSLFS_FINERESULTS.csv")