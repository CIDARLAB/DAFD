from DAFD.bin.DAFD_Interface import DAFD_Interface
from DAFD.metrics_study import metric_utils
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import pickle
import os
import matplotlib.pyplot as plt

class MetricHelper:
    """This class contains the main functions needed for the metrics study."""
    di = None
    point_flow_stability = None
    features_normalized = {}
    features_denormalized = {}
    chip_results = None
    ca_range = None
    q_range = None
    versatility_results = {}

    def __init__(self, feature_inputs, di=None):
        self.features_normalized = feature_inputs
        self.features_denormalized = metric_utils.denormalize_features(self.features_normalized)
        if di == None:
            self.di = DAFD_Interface()
        else:
            self.di = di
    #
    # def __init__(self, di=None):
    #     if di == None:
    #         self.di = DAFD_Interface()
    #     else:
    #         self.di = di

    def run_all_versatility(self):
        # make sweep
        if self.chip_results is None:
            self.chip_results = self.sweep_results(self.features_normalized)
        self.versatility_results = self.calc_versatility_score()

    def run_all_flow_stability(self):
        # make sweep
        if self.chip_results is None:
            self.chip_results = self.sweep_results(self.features_normalized)
        self.find_boundary_points()
        self.point_flow_stability = self.calc_flow_stability_score()

    # Method used for versatility score
    def sweep_results(self, chip_design, ca_range=[], q_range=[], sweep_size=25, jet_drop=False):
        if bool(ca_range):
            self.ca_range = metric_utils.make_sweep_range(ca_range, sweep_size)
        else:
            #self.ca_range = np.concatenate([np.arange(.05, .11, step=.01), np.linspace(.161111, 1.05, 9)]).round(4)
            self.ca_range = np.concatenate([np.arange(.05, .5, step=.025), np.arange(.5, 1, step=.1)]).round(4) #TODO: consider changing back
        if bool(q_range):
            self.q_range = metric_utils.make_sweep_range(q_range, sweep_size)
        else:
            self.q_range = np.linspace(2, 22, 10).round(4)
        grid_dict = {
            "flow_rate_ratio": self.q_range,
            "capillary_number": self.ca_range
        }
        pts, grid = metric_utils.make_sample_grid(chip_design, grid_dict)

        grid_measure = [self.di.runForward(pt) for pt in grid]
        if jet_drop:  # Drop jetting regime points if needed
            grid, grid_measure = metric_utils.drop_jetting_points(grid_measure, grid)
        sizes = [out["droplet_size"] for out in grid_measure]
        rates = [out["generation_rate"] for out in grid_measure]
        regime = [out["regime"] for out in grid_measure]

        out = pd.DataFrame(grid)
        out["droplet_size"] = sizes
        out["generation_rate"] = rates
        out["regime"] = regime
        return out

    def find_boundary_points(self):
        # THis is not the best way to do this with a single chip. Seems like a lot of unnecessary work
        params = {
            "capillary_number": self.ca_range,
            "flow_rate_ratio": self.q_range
        }
        for i, pt in self.chip_results.iterrows():
            base_regime = int(pt.regime)
            # get key-idx pairs from the find function of params
            base_param_idxs = {key: np.argwhere(params[key] == pt[key]).ravel()[0] for key in params.keys()}
            # using this, generate library of param combinations that need to be tested (+/-1 step in param directions unless on edge)
            adj_pts = self._get_adjacent_points(params, base_param_idxs, flow=True)
            boundary = self._compare_regimes(base_regime, adj_pts)
            self.chip_results.loc[i, "boundary"] = boundary
            # denormalize features for future analysis
            denormed = metric_utils.denormalize_features(pt.to_dict())
            self.chip_results.loc[i, "water_flow"] = denormed["water_flow"]
            self.chip_results.loc[i, "oil_flow"] = denormed["oil_flow"]

    def calc_versatility_score(self):
        results = {}
        for i in range(3):
            if i == 0:
                type = "all"
                data = self.chip_results
            elif i == 1:
                type = "dripping"
                data = self.chip_results.loc[self.chip_results.regime==1,:]
            else:
                type = "jetting"
                data = self.chip_results.loc[self.chip_results.regime==2,:]

            sizes = list(data.droplet_size)
            rates = list(data.generation_rate)
            points = np.array([[sizes[i], rates[i]] for i in range(len(sizes))])
            try:
                hull = ConvexHull(points)
                # hulls.append(hull)
                results[f"{type}_size_score"] = np.max(sizes) - np.min(sizes)
                results[f"{type}_rate_score"] = np.max(rates) - np.min(rates)
                results[f"{type}_overall_score"] = hull.volume  # hull.volume calculates the area of the polygon
            # Catch errors if a convex hull cannot be calculated (i.e less than 3 points)
            except:
                results[f"{type}_size_score"] = -1
                results[f"{type}_rate_score"] = -1
                results[f"{type}_overall_score"] = -1
        return results


    def calc_flow_stability_score(self):
        boundary_points = self.chip_results.loc[self.chip_results.boundary == 1, :]
        ## for p oints on boundary, set to 0
        self.chip_results.loc[boundary_points.index, "flow_stability"] = 0
        non_boundary_points = self.chip_results.loc[self.chip_results.boundary == 0, :]
        boundary_flows = list(zip(boundary_points.water_flow * 0.06, boundary_points.oil_flow))
        non_boundary_flows = list(zip(non_boundary_points.water_flow * 0.06, non_boundary_points.oil_flow))
        ## for each point, calculate distance of non-boundaries to boundaries
        boundary_distances = cdist(non_boundary_flows, boundary_flows)
        ## Set minimum distance as "robustness" for now
        min_distances = np.min(boundary_distances, axis=1)
        self.chip_results.loc[non_boundary_points.index, "flow_stability"] = min_distances
        base_flow_stability = self._find_flow_in_df(self.features_normalized)
        if base_flow_stability.empty:
            base_flows = (self.features_denormalized["water_flow"]*.06, self.features_denormalized["oil_flow"])
            base_distance = cdist([base_flows], boundary_flows)
            return np.min(base_distance)
        else:
            return float(base_flow_stability.flow_stability)

    def _get_adjacent_points(self, params, base_idxs, flow=True):
        adjacent_pts = []
        base_vals = {key: params[key][base_idxs[key]] for key in params}
        params_iter = params.keys()
        for key in params_iter:
            idx = base_idxs[key]
            if idx != 0:
                new_vals = base_vals.copy()
                new_vals[key] = params[key][base_idxs[key] - 1]
                adjacent_pts.append(new_vals)
            if idx != len(params[key]) - 1:
                new_vals = base_vals.copy()
                new_vals[key] = params[key][base_idxs[key] + 1]
                adjacent_pts.append(new_vals)
        return adjacent_pts

    def _find_flow_in_df(self, pt):
        return self.chip_results.loc[self.chip_results.capillary_number == pt["capillary_number"]] \
            .loc[self.chip_results.flow_rate_ratio == pt["flow_rate_ratio"], :]

    def _compare_regimes(self, base_regime, adj_pts):
        boundary = 0
        for pt in adj_pts:
            adj_regime = int(self._find_flow_in_df(pt).regime)
            if adj_regime != base_regime:
                boundary = 1
                return boundary
        return boundary

    def _normed_to_val(self, normed, ranges):
        denormed = []
        for n in normed:
            if n % 1 == 0:
                denormed.append(ranges[int(n)])
            else:
                denormed.append(np.mean([ranges[int(n - .5)], ranges[int(n + .5)]]))
        return denormed

    def _define_boundary(self, b, all_ca, all_q):
        b1 = b.loc[b.regime == 1, :]
        b2 = b.loc[b.regime == 2, :]

        ca1n = [np.where(all_ca == c)[0][0] for c in np.array(b1.capillary_number)]
        q1n = [np.where(all_q == q)[0][0] for q in np.array(b1.flow_rate_ratio)]

        ca2n = [np.where(all_ca == c)[0][0] for c in np.array(b2.capillary_number)]
        q2n = [np.where(all_q == q)[0][0] for q in np.array(b2.flow_rate_ratio)]

        points1 = np.array([[ca1n[i], q1n[i]] for i in range(len(ca1n))])
        points2 = np.array([[ca2n[i], q2n[i]] for i in range(len(ca2n))])

        distances = cdist(points1, points2)
        min_distances = np.min(cdist(points1, points2), axis=1)
        boundary_points = []
        for i in range(len(distances)):
            adj_index = np.where(distances[i, :] == min_distances[i])[0]
            for idx in adj_index:
                boundary_points.append(np.mean([points1[i], points2[idx]], axis=0))
        boundary_points = np.array(boundary_points)
        boundary_points = np.flipud(boundary_points[boundary_points[:, 0].argsort()])
        boundary_points = np.flipud(boundary_points[boundary_points[:, 1].argsort(kind="mergesort")])

        return np.array([self._normed_to_val(boundary_points[:, 0], all_ca), self._normed_to_val(boundary_points[:, 1], all_q)]).T


    def _plot_metrics(self):
        fig, axs = plt.subplots(1, 2, figsize=[12.5, 5])
        colors = ["#5B84C4", "#FB9B50"]

        out = self.chip_results
        b = out.loc[out.boundary == 1, :]
        all1 = out.loc[out.regime == 1, :]
        all2 = out.loc[out.regime == 2, :]

        axs[0].plot(all1.droplet_size, all1.generation_rate, ".", color=colors[0])
        axs[0].plot(all2.droplet_size, all2.generation_rate, ".", color=colors[1])
        axs[0].plot(self.features_normalized["droplet_size"], self.features_normalized["generation_rate"], "k*", ms=10)
        for i, data in enumerate([all1, all2]):
            sizes = np.array(data.droplet_size)
            rates = np.array(data.generation_rate)
            points = np.array([[sizes[i], rates[i]] for i in range(len(sizes))])
            hull = ConvexHull(points)

            for simplex in hull.simplices:
                axs[0].plot(points[simplex, 0], points[simplex, 1], '-', color=colors[i])

            cent = np.mean(points, 0)
            pts = []
            for pt in points[hull.simplices]:
                pts.append(pt[0].tolist())
                pts.append(pt[1].tolist())

            pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                              p[0] - cent[0]))
            pts = pts[0::2]  # Deleting duplicates
            pts.insert(len(pts), pts[0])
            k = 1
            poly = Polygon(k * (np.array(pts) - cent) + cent,
                           facecolor=colors[i], alpha=0.2)
            poly.set_capstyle('round')
            axs[0].add_patch(poly)

        axs[0].legend(["Dripping", "Jetting", "DAFD Solution"], loc="upper right")
        axs[0].set_xlabel("Droplet Size")
        axs[0].set_ylabel("Generation Rate")
        axs[0].set_xlim([0, 350])
        axs[0].set_ylim([0, 350])

        axs[1].plot(all1.capillary_number, all1.flow_rate_ratio, ".", color=colors[0])
        axs[1].plot(all2.capillary_number, all2.flow_rate_ratio, ".", color=colors[1])
        axs[1].plot(self.features_normalized["capillary_number"], self.features_normalized["flow_rate_ratio"], "k*", ms=10)

        denorm_boundary = self._define_boundary(b, np.unique(out.capillary_number), np.unique(out.flow_rate_ratio))
        points_r1 = np.append(denorm_boundary, np.array([[0.05, 2], [0.05, 22]]), axis=0)
        points_r2 = np.append(denorm_boundary, np.array([[0.9, 2], [0.9, 22]]), axis=0)

        poly_r1 = Polygon(points_r1, facecolor=colors[0], alpha=0.2)
        poly_r1.set_capstyle('round')
        axs[1].add_patch(poly_r1)

        poly_r2 = Polygon(points_r2, facecolor=colors[1], alpha=0.2)
        poly_r2.set_capstyle('round')
        axs[1].add_patch(poly_r2)
        axs[1].plot(denorm_boundary[:,0], denorm_boundary[:,1], "r-")
        axs[1].set_xlim([0.05, 0.9])
        axs[1].set_ylim([2, 22])
        axs[1].set_xlabel("Capillary Number")
        axs[1].set_ylabel("Flow Rate Ratio")
        axs[1].legend(["Dripping", "Jetting", "DAFD Solution"], loc="upper right")
        plt.savefig("DAFD/metrics_study/metrics_results.png")

    def generate_report(self, to_report):
        # TODO: make this a similar thing; figure out later
        to_report["Fluids"] = {"Dispersed phase": "DI Water",
                               "Continuous phase": "350 nf Mineral oil (viscosity: 57.2 mPa.s)",
                               "Surfactant": "5% V/V Span 80"}
        pickle.dump(to_report, open("DAFD/metrics_study/metrics.p", "wb"))
        self.features_normalized = to_report["results_df"].to_dict(orient="records")[0]
        self._plot_metrics()

        os.system('cmd /k "pweave -f md2html DAFD/metrics_study/Metrics_Report.pmd"')
