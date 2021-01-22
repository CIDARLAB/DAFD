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
        print("Processed %f chips out of %f total"%(i, len(chip_grid)))
    sizes, rates = sweep_results(chip, sweep_size=10, jet_drop=True, ca_range=[.05, .4])

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
results = pd.DataFrame(chip_grid)
results["score"] = score
results["size_score"] = size_score
results["rate_score"] = rate_score
results["hulls"] = hulls
results.to_csv("20200121_designspace_sweep.csv")