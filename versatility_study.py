import pandas as pd
from DAFD.metrics_study.rv_utils import *
from scipy.spatial import ConvexHull
from tqdm import tqdm

# Bound design space parameters (both geometric and flow)
# min_all = {
#     'orifice_size': 75,
#     'aspect_ratio': 1,
#     'expansion_ratio': 2,
#     'normalized_water_inlet': 2,
#     'normalized_oil_inlet': 2,
#     'normalized_orifice_length': 1,
# }
#
# max_all = {
#     'orifice_size': 175,
#     'aspect_ratio': 3,
#     'expansion_ratio': 6,
#     'normalized_water_inlet': 4,
#     'normalized_oil_inlet': 4,
#     'normalized_orifice_length': 3,
# }
#score = []
#size_score = []
#rate_score = []
#all_points = {}
#hulls = []
chip_cols = ['aspect_ratio','chip_number', 'expansion_ratio', 'flow_rate_ratio','normalized_oil_inlet','normalized_orifice_length', 'normalized_water_inlet', 'orifice_size']

# # Make grid to iterate over
# chip_grid = generate_design_space_grid(min_all, max_all, increment=.5)

data = pd.read_csv("20220520_allData.csv")
chips = data.groupby("chip_number")

for i in tqdm(range(len(chips))):
    chip = chips.get_group(i)
    to_add = chip.iloc[[0],:].loc[:,chip_cols]
    chip = chip.loc[chip.regime==2,:]
    # Sweep through all flow combinations and save to master dataset
    #sizes, rates, fwd_results = sweep_results(chip, sweep_size=5, ca_range=[.05, 0.4])
    #fwd_results["chip_num"] = i
    sizes = list(chip.droplet_size)
    rates = list(chip.generation_rate)
    # if i == 0:
    #     complete_sweep = fwd_results
    # else:
    #     complete_sweep = pd.concat([complete_sweep, fwd_results], sort=True)
    # Combine points and calculate convex hulls and scores
    points = np.array([[sizes[i], rates[i]] for i in range(len(sizes))])
    try:
        hull = ConvexHull(points)
        #hulls.append(hull)
        to_add["size_score"] = np.max(sizes) - np.min(sizes)
        to_add["rate_score"] = np.max(rates) - np.min(rates)
        to_add["score"] = hull.volume  # hull.volume calculates the area of the polygon
    # Catch errors if a convex hull cannot be calculated (i.e less than 3 points)
    except:
        to_add["size_score"] = -1
        to_add["rate_score"] = -1
        to_add["score"] = -1
    if i == 0:
        results_df = to_add
    else:
        results_df = pd.concat([results_df, to_add], axis=0, ignore_index=True)
    # Save checkpoints
    if i % 10000 == 0 and i > 1:
        print("Saving Checkpoint: at chip %d" % i)
        # results = pd.DataFrame(chip_grid[:i + 1])
        # results["score"] = score
        # results["size_score"] = size_score
        # results["rate_score"] = rate_score
        results_df.to_csv("data/20220520_versatility_noCutoff/20220520_versatility_results_jetting_chkpt_%d.csv" % i)
        #complete_sweep.to_csv("complete_sweep_chkpt_%d.csv" % i)

# At the end, save results
# results = pd.DataFrame(chip_grid)
# results["score"] = score
# results["size_score"] = size_score
# results["rate_score"] = rate_score
results_df.to_csv("data/20220520_versatility_noCutoff/20220520_versatility_results_jetting.csv")
#complete_sweep.to_csv("versatility_sweep_results.csv")
