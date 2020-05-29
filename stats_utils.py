import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from DAFD_TolTest import renormalize_features
import time


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


def sobol_analyis(df, sample_size, di, calc_second_order=False):
    num_vars = len(df.columns)
    mins = df.min()
    maxs = df.max()
    problem = {
        'num_vars': num_vars,
        'names': list(df.columns),
        'bounds': [[mins[i], maxs[i]] for i in range(num_vars)]
    }
    results = sobol_sampling(problem, sample_size,di, calc_second_order=calc_second_order)
    sizes = list(results.loc[:, "droplet_size"])
    gens = list(results.loc[:, "generation_rate"])
    si_size = sobol.analyze(problem, np.array(sizes), calc_second_order=calc_second_order, print_to_console=False)
    si_gen = sobol.analyze(problem, np.array(gens), calc_second_order=calc_second_order, print_to_console=False)
    return results, si_size, si_gen


def sobol_sampling(problem, sample_size,di, calc_second_order=False):
    samples = saltelli.sample(problem, sample_size, calc_second_order=calc_second_order)
    sample_dicts = to_list_of_dicts(samples, problem["names"])
    samples_normed = [renormalize_features(sample_dict) for sample_dict in sample_dicts]
    samples_df = pd.DataFrame(sample_dicts)
    outputs = [di.runForward(sample_normed) for sample_normed in samples_normed]
    outputs_df = pd.DataFrame(outputs).loc[:, ["droplet_size", "generation_rate"]]
    return pd.concat([samples_df, outputs_df], axis=1)


def to_list_of_dicts(samples, keys):
    sample_dict_list = []
    for sample in samples:
        sample_dict_list.append({key: sample[i] for i, key in enumerate(keys)})
    return sample_dict_list


def pct_change(array, base):
    return round((array - base)/base * 100, 4)
