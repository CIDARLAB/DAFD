import pandas as pd
import matplotlib.pyplot as plt
from rv_study.rv_utils import *
from scipy.spatial import ConvexHull, convex_hull_plot_2d

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
score = []
size_score = []
rate_score = []
all_points = {}
hulls = []


chip_grid = generate_design_space_grid(min_all, max_all, increment=.5)
for i, chip in enumerate(chip_grid):
    if i % 100 == 0:
        print("Processed %d chips out of %d total"%(i, len(chip_grid)))
    sizes, rates, full_sweep = sweep_results(chip, sweep_size=10, jet_drop=False, ca_range=[.05, 1.05])
    full_sweep["chip_num"] = i
    if i == 0:
        complete_sweep = full_sweep
    else:
        complete_sweep = pd.concat([complete_sweep, full_sweep])

    points = np.array([[sizes[i], rates[i]] for i in range(len(sizes))])
    try:
        hull = ConvexHull(points)
        hulls.append(hull)
        size_score.append(np.max(sizes) - np.min(sizes))
        rate_score.append(np.max(rates) - np.min(rates))
        score.append(hull.volume)
    except:
        hulls.append(-1)
        score.append(-1)
        size_score.append(-1)
        rate_score.append(-1)
    if i % 5000 == 0 and i > 1:
        print("Saving Checkpoint: at chip %d" % i)
        results = pd.DataFrame(chip_grid[:i+1])
        results["score"] = score
        results["size_score"] = size_score
        results["rate_score"] = rate_score
        results.to_csv("checkpoints3/results_chkpt_%d.csv" % i)
        complete_sweep.to_csv("checkpoints3/complete_sweep_chkpt_%d.csv" % i)


results = pd.DataFrame(chip_grid)
results["score"] = score
results["size_score"] = size_score
results["rate_score"] = rate_score
results.to_csv("20210125_designspace_sweep3.csv")
complete_sweep.to_csv("20210125_complete_sweep3.csv")

