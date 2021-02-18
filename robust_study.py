import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DAFD.rv_study.rv_utils import *

orthogonal_devices = pd.read_csv("DAFD/rv_study/orthogonal_devices.csv")
results = orthogonal_devices.copy()
q_vals = [4, 10, 16, 22]
cap_numbers= [0.05, 0.1, 0.2, 0.4] #Jet drop if regime change occurs
tolerance = 0.1
tol_spread = 5
# all_results = np.zeros(4,4,)
devices = orthogonal_devices.to_dict(orient='records')
device_scores = []
device_size_scores = []
device_rate_scores = []

for i, device in enumerate(devices):
    print("AT DEVICE %d" % i)
    if i == 1:
        a = 2

    features = device.copy()
    chip_num = features["Chip #"]
    del features["Chip #"]
    grid_dict = {
        "flow_rate_ratio": q_vals,
        "capillary_number":cap_numbers
    }
    pts, grid = make_sample_grid(features, grid_dict)

    # on each point in the grid, run the tolerance study
    scores = []
    size_scores = []
    rate_scores = []
    for pt in grid:
        score, size_score, rate_score = calculate_robust_score(pt)
        scores.append(score)
        size_scores.append(size_score)
        rate_scores.append(rate_score)
    device_scores.append(np.mean(scores))
    device_size_scores.append(np.mean(size_scores))
    device_rate_scores.append(np.mean(rate_scores))

results["score"] = device_scores
results["size_scores"] = device_size_scores
results["rate_scores"] = device_rate_scores
results.to_csv("20210218_robust_study_flow.csv")
