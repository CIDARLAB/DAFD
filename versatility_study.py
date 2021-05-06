import pandas as pd
from DAFD.rv_study.rv_utils import *
from scipy.spatial import ConvexHull
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
score = []
size_score = []
rate_score = []
all_points = {}
hulls = []

# Make grid to iterate over
chip_grid = generate_design_space_grid(min_all, max_all, increment=.5)

for i, chip in tqdm(enumerate(chip_grid)):
    # Sweep through all flow combinations and save to master dataset
    sizes, rates, fwd_results = sweep_results(chip, sweep_size=5, ca_range=[.05, 0.1])
    fwd_results["chip_num"] = i
    if i == 0:
        complete_sweep = fwd_results
    else:
        complete_sweep = pd.concat([complete_sweep, fwd_results], sort=True)
    # Combine points and calculate convex hulls and scores
    points = np.array([[sizes[i], rates[i]] for i in range(len(sizes))])
    try:
        hull = ConvexHull(points)
        hulls.append(hull)
        size_score.append(np.max(sizes) - np.min(sizes))
        rate_score.append(np.max(rates) - np.min(rates))
        score.append(hull.volume)  # hull.volume calculates the area of the polygon
    # Catch errors if a convex hull cannot be calculated (i.e less than 3 points)
    except:
        hulls.append(-1)
        score.append(-1)
        size_score.append(-1)
        rate_score.append(-1)
    # Save checkpoints
    if i % 5000 == 0 and i > 1:
        print("Saving Checkpoint: at chip %d" % i)
        results = pd.DataFrame(chip_grid[:i + 1])
        results["score"] = score
        results["size_score"] = size_score
        results["rate_score"] = rate_score
        results.to_csv("results_chkpt_%d.csv" % i)
        complete_sweep.to_csv("complete_sweep_chkpt_%d.csv" % i)

# At the end, save results
results = pd.DataFrame(chip_grid)
results["score"] = score
results["size_score"] = size_score
results["rate_score"] = rate_score
results.to_csv("versatility_results.csv")
complete_sweep.to_csv("versatility_sweep_results.csv")
