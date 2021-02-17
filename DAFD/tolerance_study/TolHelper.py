from DAFD.bin.DAFD_Interface import DAFD_Interface
from DAFD.tolerance_study.plot_utils import *
import os
import pickle
#TODO: Include package handling in correct files

class TolHelper:
    """This class contains the main functions needed for the tolerance study."""
    features_normalized = {}
    features_denormalized = {}
    warnings = []
    tolerance = None
    di = None
    tol_df = None

    flow_heatmap_size = None
    flow_heatmap_gen = None
    flow_grid_size = None

    feature_heatmap_size = None
    feature_heatmap_gen = None
    feature_grid_size = None

    pf_samples = None
    si_size = None
    si_gen = None
    file_base = None

    def __init__(self, feature_inputs, di=None, tolerance=10, feature_grid_size = 11, flow_grid_size = 11, pf_samples=100):
        self.features_normalized = feature_inputs
        self.features_denormalized = self.denormalize_features(self.features_normalized)
        self.tolerance = tolerance/100
        if di == None:
            self.di = DAFD_Interface()
        else:
            self.di = di
        self.tol_df = self.make_tol_df(self.features_denormalized, self.tolerance)
        self.feature_names = list(self.tol_df.columns)
        self.feature_grid_size = feature_grid_size
        self.flow_grid_size = flow_grid_size
        self.pf_samples = pf_samples

    def run_all(self):
        self.sobol_analysis()
        self.feature_heatmaps()
        self.flow_heatmaps()

    def plot_all(self, base="toltest"):
        self.file_base = base
        if self.flow_heatmap_size is None or self.flow_heatmap_gen is None:
            self.run_all()
        # f1 = plot_sobol_results(self.si_size, self.si_gen, self.feature_names)
        # plt.savefig(base + "_principal_features.png")
        #
        # f2 = plot_heatmaps(self.feature_heatmap_size, self.feature_heatmap_gen)
        # plt.savefig(base + "_feature_heatmaps.png")


        f2 = plot_half_heatmaps_grid(self.feature_heatmap_size, "Droplet Size", include_pcs=True, si=self.si_size,
                                     names=self.feature_names)
        plt.savefig("tolerance_study/" + base + "_SizeGRID.png")
        f3 = plot_half_heatmaps_grid(self.feature_heatmap_gen, "Generation Rate", include_pcs=True, si=self.si_gen,
                                     names=self.feature_names)
        plt.savefig("tolerance_study/" + base + "_RateGRID.png")

        f1 = plot_flow_heatmaps(self.flow_heatmap_size, self.flow_heatmap_gen, self.features_denormalized)
        plt.savefig("tolerance_study/" + base + "_flow_heatmaps.png")

        return f1, f2, f3

    def generate_report(self):
        to_report = {"features": self.features_denormalized,
                     "tolerance": self.tolerance,
                     "base_performance": self.di.runForward(self.features_normalized),
                     "Fluids": {"Dispersed phase": "DI Water",
                                "Continuous phase": "350 nf Mineral oil (viscosity: 57.2 mPa.s )",
                                "Surfactant": "5% V/V Span 80"},
                     "Warnings": self.warnings
                     }
        pickle.dump(to_report, open("tolerance_study/tol.p", "wb"))
        os.system('cmd /k "pweave -f md2html tolerance_study/Tolerance_Report.pmd"')

    def sobol_analysis(self, calc_second_order=True):
        si_size, si_gen = self.principal_feature_analysis(calc_second_order=calc_second_order)
        self.si_size = si_size
        self.si_gen = si_gen
        return si_size, si_gen

    def feature_heatmaps(self):
        if self.si_gen is None or self.si_size is None:
            _, _ = self.sobol_analysis()
        pc_s = get_principal_feature(self.si_size, self.feature_names)
        pc_g = get_principal_feature(self.si_gen, self.feature_names)
        heatmaps_size, heatmaps_rate = self.make_feature_heatmaps(pc_s, pc_g)
        self.feature_heatmap_size = heatmaps_size
        self.feature_heatmap_gen = heatmaps_rate
        return heatmaps_size, heatmaps_rate

    def flow_heatmaps(self, range_mult=None):
        if range_mult is None:
            range_mult = self.tolerance*2
        oil_range = [self.features_denormalized["oil_flow"]*(1-range_mult),
                     self.features_denormalized["oil_flow"]*(1+range_mult)]
        water_range = [self.features_denormalized["water_flow"]*(1-range_mult),
                       self.features_denormalized["water_flow"]*(1+range_mult)]
        if oil_range[0] < 0.05:
            oil_range[0] = 0.05
        if water_range[0] < 0.05:
            water_range[0] = 0.05
        flow_heatmap_size, flow_heatmap_gen = self.make_flow_heatmaps(oil_range, water_range)
        self.flow_heatmap_size = flow_heatmap_size
        self.flow_heatmap_gen  = flow_heatmap_gen
        return flow_heatmap_size, flow_heatmap_gen

    def make_tol_df(self, features, tol):
        max_feat = {key: (features[key] + tol * features[key]) for key in features.keys()}
        min_feat = {key: (features[key] - tol * features[key]) for key in features.keys()}
        return pd.DataFrame([min_feat, features, max_feat])


    def make_flow_heatmaps(self, oil_range, water_range):
        oil_rounding = int(np.abs(np.floor(np.log10((oil_range[1] - oil_range[0])/self.flow_grid_size))))
        water_rounding = int(np.abs(np.floor(np.log10((water_range[1] - water_range[0])/self.flow_grid_size))))
        oil = np.around(make_grid_range(pd.Series(oil_range), self.flow_grid_size), oil_rounding)
        water = np.around(make_grid_range(pd.Series(water_range), self.flow_grid_size), water_rounding)

        grid_dict = {"oil_flow": oil, "water_flow": water}
        flow_heatmap_size = self.generate_heatmap_data(grid_dict, "droplet_size", percent=False)
        flow_heatmap_gen = self.generate_heatmap_data(grid_dict, "generation_rate", percent=False)
        return flow_heatmap_size, flow_heatmap_gen


    def make_feature_heatmaps(self, pc_s, pc_g):
        tol_df_shuff = self.tol_df[[col for col in self.tol_df.columns if col != pc_s] + [pc_s]]
        tol_df_shuff = tol_df_shuff[[col for col in self.tol_df.columns if col != pc_g] + [pc_g]]

        heatmap_data_s = self._heatmap_loop(pc_s, tol_df_shuff, "droplet_size")
        heatmap_data_g = self._heatmap_loop(pc_g, tol_df_shuff, "generation_rate")
        return heatmap_data_s, heatmap_data_g


    def _heatmap_loop(self, pc, tol_df_shuff, output):
        pc_range = make_grid_range(tol_df_shuff.loc[:, pc], self.feature_grid_size)
        features = [feat for feat in tol_df_shuff.columns if feat != pc]
        heatmap_data = []
        for feat in features:
            feat_range = make_grid_range(tol_df_shuff.loc[:, feat], self.feature_grid_size)
            grid_dict = {pc: pc_range, feat: feat_range}
            heatmap_data.append(self.generate_heatmap_data(grid_dict, output))
        return heatmap_data


    def generate_heatmap_data(self, grid_dict, output, percent=True):
        key_names = list(grid_dict.keys())
        pts, grid = make_sample_grid(self.features_denormalized, grid_dict)
        grid_measure = [self.di.runForward(self.renormalize_features(pt)) for pt in grid]
        outputs = [out[output] for out in grid_measure]
        for i, pt in enumerate(pts):
            pt.append(outputs[i])
        heat_df = pd.DataFrame(pts, columns=[key_names[0], key_names[1], output])
        if percent:
            heat_df.loc[:, key_names[0]] = pct_change(heat_df.loc[:, key_names[0]],
                                                      self.features_denormalized[key_names[0]]).astype(float)
            heat_df.loc[:, key_names[1]] = pct_change(heat_df.loc[:, key_names[1]],
                                                      self.features_denormalized[key_names[1]]).astype(float)
            base_out = self.di.runForward(self.features_normalized)[output]
            heat_df.loc[:, output] = pct_change(heat_df.loc[:, output], base_out)
        heat_pivot = heat_df.pivot(index=key_names[1], columns=key_names[0], values=output)
        return heat_pivot[::-1]


    def principal_feature_analysis(self, calc_second_order=False):
        mins = self.tol_df.min()
        maxs = self.tol_df.max()
        problem = {
            'num_vars': len(self.feature_names),
            'names': self.feature_names,
            'bounds': [[mins[i], maxs[i]] for i in range(len(mins))]
        }
        results = self.sobol_sampling(problem, calc_second_order=calc_second_order)
        sizes = list(results.loc[:, "droplet_size"])
        gens = list(results.loc[:, "generation_rate"])
        si_size = sobol.analyze(problem, np.array(sizes), calc_second_order=calc_second_order, print_to_console=False)
        si_gen = sobol.analyze(problem, np.array(gens), calc_second_order=calc_second_order, print_to_console=False)
        return si_size, si_gen


    def sobol_sampling(self, problem, calc_second_order=False):
        samples = saltelli.sample(problem, self.pf_samples, calc_second_order=calc_second_order)
        sample_dicts = to_list_of_dicts(samples, problem["names"])
        samples_normed = [self.renormalize_features(sample_dict) for sample_dict in sample_dicts]
        samples_df = pd.DataFrame(sample_dicts)
        outputs = [self.di.runForward(sample_normed) for sample_normed in samples_normed]
        outputs_df = pd.DataFrame(outputs).loc[:, ["droplet_size", "generation_rate"]]
        return pd.concat([samples_df, outputs_df], axis=1)


    def denormalize_features(self, features):
        Or = features["orifice_size"]
        As = features["aspect_ratio"]
        Exp = features["expansion_ratio"]
        norm_Ol = features["normalized_orifice_length"]
        norm_Wi = features["normalized_water_inlet"]
        norm_Oi = features["normalized_oil_inlet"]
        Q_ratio = features["flow_rate_ratio"]
        Ca_num = features["capillary_number"]

        channel_height = Or * As
        outlet_channel_width = Or * Exp
        orifice_length = Or * norm_Ol
        water_inlet_width = Or * norm_Wi
        oil_inlet = Or * norm_Oi
        oil_flow_rate = (Ca_num * 0.005 * channel_height * oil_inlet * 1e-12) / \
                        (0.0572 * ((water_inlet_width * 1e-6)) * (
                                (1 / (Or * 1e-6)) - (1 / (2 * oil_inlet * 1e-6))))
        oil_flow_rate_ml_per_hour = oil_flow_rate * 3600 * 1e6
        water_flow_rate = oil_flow_rate_ml_per_hour / Q_ratio
        water_flow_rate_ul_per_min = water_flow_rate * 1000 / 60

        ret_dict = {
            "orifice_size": Or,
            "depth": channel_height,
            "outlet_width": outlet_channel_width,
            "orifice_length": orifice_length,
            "water_inlet": water_inlet_width,
            "oil_inlet": oil_inlet,
            "oil_flow": oil_flow_rate_ml_per_hour,
            "water_flow": water_flow_rate_ul_per_min
        }
        return ret_dict


    def renormalize_features(self, features):
        channel_height = features["depth"]
        outlet_channel_width = features["outlet_width"]
        orifice_length = features["orifice_length"]
        water_inlet_width = features["water_inlet"]
        oil_inlet = features["oil_inlet"]
        oil_flow_rate_ml_per_hour = features["oil_flow"]
        water_flow_rate_ul_per_min = features["water_flow"]

        Or = features["orifice_size"]
        As = channel_height/Or
        Exp = outlet_channel_width/Or
        norm_Ol = orifice_length/Or
        norm_Wi = water_inlet_width/Or
        norm_Oi = oil_inlet/Or

        Q_ratio = oil_flow_rate_ml_per_hour / (water_flow_rate_ul_per_min*(60/1000))

        Ca_num = ((0.0572*water_inlet_width * 1e-6*(oil_flow_rate_ml_per_hour/(3600*1e6))) / \
                 (0.005 * channel_height * 1e-6 * oil_inlet * 1e-6)) * (1/(Or * 1e-6) - 1/(2*oil_inlet*1e-6))
        ret_dict = {
            "orifice_size": Or,
            "aspect_ratio": As,
            "expansion_ratio": Exp,
            "normalized_orifice_length": norm_Ol,
            "normalized_water_inlet": norm_Wi,
            "normalized_oil_inlet": norm_Oi,
            "flow_rate_ratio": Q_ratio,
            "capillary_number":  round(Ca_num, 5)
            }
        return ret_dict