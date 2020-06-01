import matplotlib.pyplot as plt
import seaborn as sns
from tol_utils import *


def plot_results(outputs, original, tolerance):
    plt.scatter([i["droplet_size"] for i in outputs], [i["generation_rate"] for i in outputs])
    plt.scatter(original["droplet_size"], original["generation_rate"])
    plt.xlabel("Droplet Size")
    plt.ylabel("Generation Rate")
    plt.title("All possible outputs with tolerance of %d percent" % tolerance)
    plt.legend(["Results with Tolerance", "User Input"])
    plt.show()


def plot_heatmap(data, axs,cbar_ax, row=0, col=0, vmax=None, map="magma"):
    if col==6:
        plot = sns.heatmap(data, ax=axs[row][col],
                           cbar=True, vmin=-vmax, vmax=vmax,
                           cmap = map, cbar_ax=cbar_ax)
    else:
        plot = sns.heatmap(data, ax=axs[row][col],
                            cbar=False, vmin=-vmax, vmax=vmax,
                           cmap = map,)
    return plot


def plot_heatmaps(hm_s, hm_g):
    size_max = max([np.abs(df).max().max() for df in hm_s])
    rate_max = max([np.abs(df).max().max() for df in hm_g])

    dx =0.7
    dy = 0.8
    figsize = plt.figaspect(float(dx * 2) / float(dy * 8))

    fig, axs = plt.subplots(2,len(hm_s), figsize=figsize, facecolor="w")
    #cbar_ax = fig.add_axes([.91, .6, .01, .3])
    ca_pos_size = axs[0][6].get_position()
    ca_pos_rate = axs[1][6].get_position()
    cbar_ax_size = fig.add_axes([ca_pos_size.x0+0.1, ca_pos_size.y0+0.04, 0.01, 0.3])
    cbar_ax_rate = fig.add_axes([ca_pos_rate.x0+0.1, ca_pos_rate.y0+0.04, 0.01, 0.3])
    pad = 0.05  # Padding around the edge of the figure
    xpad, ypad = dx * pad/2, dy * 3*pad
    fig.subplots_adjust(left=xpad+0.02, right=(1 - xpad)-0.09, top=1 - ypad, bottom=ypad, wspace=0.6, hspace=0.6)
    hms = [hm_s, hm_g]
    for i in range(len(hm_s)):
        plot_s = plot_heatmap(hm_s[i],axs,cbar_ax_size, row=0, col=i,vmax=size_max,map="viridis")
        plot_g = plot_heatmap(hm_g[i],axs,cbar_ax_rate, row=1, col=i,vmax=rate_max,map='plasma')
    plt.gcf().subplots_adjust(bottom=0.15)
    return fig


def plot_sobol_results(si_size, si_gen, names):
    sns.set_style("white")
    sns.set_context("notebook")
    fig, axs = plt.subplots(2, 1, facecolor="w")
    plt.gcf().subplots_adjust(bottom=0.25)

    si = [si_size, si_gen]
    output = ["(Size)", "(Rate)"]
    for i, ax in enumerate(axs):
        sns.barplot(names, si[i]["ST"], ax=ax, color='#00B2EE')
        if i == 0:
            plt.setp(ax, xticks = [])
        plt.setp(ax, ylabel=("Total-Effect "+output[i]))
        plt.xticks(rotation="vertical")
    ylims = [np.around(ax.get_ylim()[1],1) for ax in axs]
    for ax in axs:
        ax.set_ylim(0, max(ylims)*1.1)
    return fig


def plot_flow_heatmaps(size_df, rate_df, feat_denorm):
    oil_flow = feat_denorm["oil_flow"]
    water_flow = feat_denorm["water_flow"]
    oil_idx = min_dist_idx(oil_flow, size_df.columns)
    water_idx = min_dist_idx(water_flow, size_df.index)

    tick_spacing = int(np.floor(len(size_df.columns) / 10))
    dx = 0.15
    dy = 1
    figsize = plt.figaspect(float(dx * 2) / float(dy * 1))
    fig, axs = plt.subplots(1, 2, facecolor="w", figsize=figsize)
    plt.subplots_adjust(wspace=0.3)
    sns.set_style("white")
    sns.set_context("notebook")
    sns.set(font_scale=1.25)
    sns.heatmap(size_df, cmap="viridis", vmin=0, ax=axs[0], xticklabels=tick_spacing,
                yticklabels=tick_spacing, cbar_kws={'label': 'Droplet Size'})
    axs[0].scatter(oil_idx, water_idx, marker="*", color="r", s=200)

    plt.setp(axs[0], xlabel="Oil Flow Rate (ml/hr)", ylabel="Water Flow Rate (uL/min)")
    sns.heatmap(rate_df, cmap="viridis", vmin=0, ax=axs[1], xticklabels=tick_spacing,
                yticklabels=tick_spacing, cbar_kws={'label': 'Generation Rate'})
    plt.setp(axs[1], xlabel="Oil Flow Rate (ml/hr)", ylabel="Water Flow Rate (uL/min)")
    axs[1].scatter(oil_idx, water_idx, marker="*", color="r", s=200)
    return fig