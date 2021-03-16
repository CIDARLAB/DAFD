import pandas as pd
import matplotlib.pyplot as plt
from DAFD.rv_study.rv_utils import *
import time
# Generate minimum and maximum values
# Taken from DAFD paper

min_all = {
    'orifice_size': 75,
    'aspect_ratio': 1,
    'expansion_ratio': 2,
    'normalized_water_inlet': 2,
    'normalized_oil_inlet': 2,
    'normalized_orifice_length': 1,
}

max_all = {
    'orifice_size': 175,
    'aspect_ratio': 3,
    'expansion_ratio': 6,
    'normalized_water_inlet': 4,
    'normalized_oil_inlet': 4,
    'normalized_orifice_length': 3,
}



tolerance = 0.1
tol_spread = 5
cap_nums = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.16111111, 0.27222222, 0.38333333, 0.49444444,
                     0.60555556, 0.71666667, 0.82777778, 0.93888889, 1.05]

q_vals = [2,  4.22222222,  6.44444444,  8.66666667, 10.88888889, 13.11111111, 15.33333333, 17.55555556,
          19.77777778, 22.]

results_df = pd.read_csv("data/robustness_study/20210310_robustness_chkpt_28000.csv")
results_df = results_df.drop(labels="Unnamed: 0", axis=1)
results = results_df.to_dict(orient="records")

chip_grid = generate_design_space_grid(min_all, max_all, increment=.5)
start = time.time()
for i, chip in enumerate(chip_grid):
    features = chip.copy()
    grid_dict = {
        "flow_rate_ratio": q_vals,
        "capillary_number": cap_nums
    }
    pts, grid = make_sample_grid(features, grid_dict)
    # on each point in the grid, run the tolerance study
    scores = []
    size_scores = []
    rate_scores = []
    for pt in grid:
        score, size_score, rate_score = calculate_robust_score(pt)
        init = di.runForward(pt)
        pt["droplet_size"] = init["droplet_size"]
        pt["generation_rate"] = init["generation_rate"]
        pt["regime"] = init["regime"]
        pt["score"] = score
        pt["size_score"] = size_score
        pt["rate_score"] = rate_score
        pt["chip_number"] = i
        results.append(pt)


    if i % 100 == 0:
        print("at chip %d" % i)
        print("ELAPSED: %f" % ((time.time() - start)/60))
    if i % 1000 == 0 and i > 1:
        print("Saving Checkpoint: at chip %d" % i)
        results_df = pd.DataFrame(results)
        results_df.to_csv("data/robustness_study/20210310_robustness_chkpt_%d.csv" % i)

results_df = pd.DataFrame(results)
results_df.to_csv("20210310_robustness_designspace.csv")

