import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import time
import itertools

def make_grid_range(vals, size):
    return np.linspace(vals.min(), vals.max(), size)


def make_sample_grid(base_features, perturbations, entire_chip=False):
    base_copy = base_features.copy()
    pert_vals = list(perturbations.values())
    if entire_chip:
        options = itertools.product(pert_vals[0], pert_vals[1], pert_vals[2], pert_vals[3], pert_vals[4], pert_vals[5])
    else:
        options = itertools.product(pert_vals[0], pert_vals[1])
    pts = []
    grid = []
    for option in options:
        pts.append(list(option))
        base_copy.update({key: option[i] for i, key in enumerate(perturbations.keys())})
        grid.append(base_copy.copy())
    return pts, grid



def get_principal_feature(si, feature_names):
    ST = list(si["ST"])
    return feature_names[ST.index(max(ST))]


def min_dist_idx(pt, array):
    distances = [np.linalg.norm(pt - arraypt) for arraypt in array]
    return distances.index(min(distances))


def main_effect_analysis(data, inputs_df):
    size_vars = []
    gen_vars = []
    for col in inputs_df.columns:
        size_means = data.groupby(col)["droplet_size"].mean()
        gen_means = data.groupby(col)["generation_rate"].mean()
        size_vars.append(np.var(size_means))
        gen_vars.append(np.var(gen_means))

    size_var = np.var(data.loc[:, "droplet_size"])
    gen_var = np.var(data.loc[:, "generation_rate"])
    summary = pd.DataFrame([size_vars / size_var, gen_vars / gen_var], index=["size var", "gen var"],
                           columns=inputs_df.columns)
    summary = summary.T
    return summary

def to_list_of_dicts(samples, keys):
    sample_dict_list = []
    for sample in samples:
        sample_dict_list.append({key: sample[i] for i, key in enumerate(keys)})
    return sample_dict_list


def pct_change(array, base):
    return np.around((array - base)/base * 100, 3)
