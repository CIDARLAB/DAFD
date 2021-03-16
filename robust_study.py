import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DAFD.rv_study.rv_utils import *

orthogonal_devices = pd.read_csv("DAFD/rv_study/orthogonal_devices.csv")
results = []
q_vals = [4, 10, 16, 22]
cap_numbers= [0.05, 0.1, 0.2, 0.4] #Jet drop if regime change occurs
tolerance = 0.1
tol_spread = 5
devices = orthogonal_devices.to_dict(orient='records')
device_scores = []
device_size_scores = []
device_rate_scores = []
reg_list = []
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
        init = di.runForward(pt)
        reg_list.append(init["regime"])
        pt["regime"] = init["regime"]
        pt["score"] = score
        pt["size_score"] = size_score
        pt["rate_score"] = rate_score
        results.append(pt)

results = pd.DataFrame(results)
results.to_csv("20210219_robust_study_exp_fab.csv")
