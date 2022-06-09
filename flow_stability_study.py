import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np

# Denormalize the features
# Using the flow rates, find the distance to the closest boundary point
data = pd.read_csv("20220519_boundary_points.csv")
data.water_flow = data.water_flow * 0.06
oil_flows = data.oil_flow
water_flows = data.water_flow

chips = data.chip_number.unique()
# Calculate distances (for EACH chip):
for chip in tqdm(chips):
    points = data.loc[data.chip_number == chip, :]
    ## if chip has all single regime, set to -1
    if len(points.regime.unique()) == 1:
        data.loc[points.index, "flow_stability"] = -1 #only 3 devices had this and were all in jetting
    else:
        boundary_points = points.loc[points.boundary == 1,:]
        ## for points on boundary, set to 0
        data.loc[boundary_points.index, "flow_stability"] = 0
        non_boundary_points = points.loc[points.boundary == 0,:]
        boundary_flows = list(zip(boundary_points.normed_water, boundary_points.normed_oil))
        non_boundary_flows = list(zip(non_boundary_points.normed_water, non_boundary_points.normed_oil))
        ## for each point, calculate distance of non-boundaries to boundaries
        boundary_distances = cdist(non_boundary_flows, boundary_flows)
        ## Set minimum distance as "robustness" for now
        min_distances = np.min(boundary_distances, axis=1)
        data.loc[non_boundary_points.index, "flow_stability"] = min_distances

data.to_csv("20220519_boundary_points_normed.csv")

