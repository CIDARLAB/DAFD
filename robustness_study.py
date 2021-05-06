import pandas as pd
from DAFD.rv_study.rv_utils import *
from tqdm import tqdm

# Bound design space parameters (both geometric and flow)
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

tolerance = 0.1 # Set tolerance to 10%
cap_nums = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.16111111, 0.27222222, 0.38333333, 0.49444444,
            0.60555556, 0.71666667, 0.82777778, 0.93888889, 1.05]

q_vals = [2,  4.22222222,  6.44444444,  8.66666667, 10.88888889, 13.11111111, 15.33333333, 17.55555556,
          19.77777778, 22.]

results = []

# Make grid to iterate over
chip_grid = generate_design_space_grid(min_all, max_all, increment=.5)

for i, chip in tqdm(enumerate(chip_grid)):
    features = chip.copy()
    grid_dict = {
        "flow_rate_ratio": q_vals,
        "capillary_number": cap_nums
    }
    # Make secondary grid for all flow conditions
    pts, grid = make_sample_grid(features, grid_dict)
    scores = []
    size_scores = []
    rate_scores = []
    for pt in grid:
        # Sweep through all conditions and calculate robustness scores
        init = di.runForward(pt) # Initial result before perturbation
        robustness_results = calculate_robust_score(pt)
        pt.update(robustness_results)
        pt["droplet_size"] = init["droplet_size"]
        pt["generation_rate"] = init["generation_rate"]
        pt["regime"] = init["regime"]
        pt["chip_number"] = i
        results.append(pt)
    # Save checkpoints
    if i % 5000 == 0 and i > 1:
        print("Saving Checkpoint: at chip %d" % i)
        results_df = pd.DataFrame(results)
        results_df.to_csv("robustness_chkpt_%d.csv" % i)
# At the end, save results
results_df = pd.DataFrame(results)
results_df.to_csv("robustness_results.csv")
