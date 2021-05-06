import pandas as pd
import matplotlib.pyplot as plt
from DAFD.rv_study.rv_utils import *
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# Generate minimum and maximum values
# Taken from DAFD paper

score = []
size_score = []
rate_score = []
all_points = {}
hulls = []

# Load Dataset
chip_grid = generate_design_space_grid(min_all, max_all, increment=.5)
chips = pd.read_csv("data/old/20210212_designspace_DRIPPING.csv")
all_results = pd.read_csv("data/old/20210212_fullsweep.csv")
all_results = all_results.loc[all_results["regime"] == 1, :]
all_results = all_results.rename(columns={"size":"droplet_size", "rate":"generation_rate"}).reset_index()



for i, chip in enumerate(chip_grid):
    if i % 100 == 0:
        print("Processed %d chips out of %d total"%(i, len(chip_grid)))
        print("Took %f seconds to process 100 chips. Avg of %f times per calculation" %(el, el/(100*100)))
    #chip = chip_grid[13999]
    sizes, rates, full_sweep = sweep_results(chip, sweep_size=5, jet_drop=False, ca_range=[.05, 0.1])
    full_sweep["chip_num"] = i
    if i == 0:
        complete_sweep = full_sweep
    else:
        complete_sweep = pd.concat([complete_sweep, full_sweep], sort=True)

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
    if i % 500 == 0:
        print("at chip %d" % i)
    if i % 5000 == 0 and i > 1:
        print("Saving Checkpoint: at chip %d" % i)
        results = pd.DataFrame(chip_grid[:i+1])
        results["score"] = score
        results["size_score"] = size_score
        results["rate_score"] = rate_score
        # results.to_csv("DAFD/other_ignore_git/FINE_results_chkpt_%d.csv" % i)
        # complete_sweep.to_csv("DAFD/other_ignore_git/FINE_complete_sweep_chkpt_%d.csv" % i)


results = pd.DataFrame(chip_grid)
results["score"] = score
results["size_score"] = size_score
results["rate_score"] = rate_score
# results.to_csv("20210211_designspace_fine.csv")
# complete_sweep.to_csv("20210211_complete_fine.csv")

