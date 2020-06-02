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


def plot_heatmap(data, axs,cbar_ax,label, row=0, col=0, vmax=None, map="magma"):
    if col==6:
        plot = sns.heatmap(data, ax=axs[row][col],
                           cbar=True, vmin=-vmax, vmax=vmax,
                           cmap = map, cbar_ax=cbar_ax, cbar_kws={'label': label+"\n(% change)"})
    else:
        plot = sns.heatmap(data, ax=axs[row][col],
                            cbar=False, vmin=-vmax, vmax=vmax,
                           cmap = map)
    axs[row][col].scatter(len(data.columns)/2, len(data.columns)/2, marker="*", color='w', s=100)
    plt.setp(axs[row][col], xlabel= str.capitalize(data.columns.name.replace("_", " ")) + "\n(% change)",
             ylabel= str.capitalize(data.index.name.replace("_", " ")) + "\n(% change)")
    return plot


def plot_heatmaps(hm_s, hm_g):
    size_max = max([np.abs(df).max().max() for df in hm_s])
    rate_max = max([np.abs(df).max().max() for df in hm_g])

    dx = 0.7
    dy = 0.8
    figsize = plt.figaspect(float(dx * 2) / float(dy * 8))

    fig, axs = plt.subplots(2,len(hm_s)+1, constrained_layout=True,figsize=figsize, facecolor="w")
    axs[0, -1].axis('off')
    axs[1, -1].axis('off')
    # fig.subplots_adjust(right =0.9)
    ca_pos_size = axs[0][-1].get_position()
    ca_pos_rate = axs[1][-1].get_position()
    cbar_ax_size = fig.add_axes([ca_pos_size.x0+0.1, ca_pos_size.y0+0.15, 0.01, 0.3])
    cbar_ax_rate = fig.add_axes([ca_pos_rate.x0+0.1, ca_pos_rate.y0+0.075, 0.01, 0.3])
    hms = [hm_s, hm_g]
    for i in range(len(hm_s)):
        plot_s = plot_heatmap(hm_s[i],axs,cbar_ax_size, "Droplet Size", row=0, col=i,vmax=size_max,map="viridis")
        plot_g = plot_heatmap(hm_g[i],axs,cbar_ax_rate, "Generation Rate", row=1, col=i,vmax=rate_max,map='plasma')
    return fig


def plot_sobol_results(si_size, si_gen, names):
    sns.set_style("white")
    sns.set_context("notebook")
    fig, axs = plt.subplots(2, 1, facecolor="w")
    plt.gcf().subplots_adjust(bottom=0.25)

    si = [si_size, si_gen]
    output = ["(Size)", "(Rate)"]
    colors = ["#1f968b", "#c03a83"]
    for i, ax in enumerate(axs):
        sns.barplot(names, si[i]["ST"], ax=ax, color=colors[i])
        if i == 0:
            plt.setp(ax, xticks = [])
        plt.setp(ax, ylabel=("Total-Effect "+output[i]))
        plt.xticks(rotation="vertical")
    ylims = [np.around(ax.get_ylim()[1],1) for ax in axs]
    for ax in axs:
        ax.set_ylim(0, max(ylims)*1.1)
    return fig


def plot_flow_heatmaps(size_df, rate_df, feat_denorm):
    tick_spacing = int(np.floor(len(size_df.columns) / 10))
    dx = 0.15
    dy = 1
    figsize = plt.figaspect(float(dx * 2) / float(dy * 1))
    fig, axs = plt.subplots(1, 2, facecolor="w", figsize=figsize)
    plt.subplots_adjust(wspace=0.3, bottom=0.15)
    sns.set_style("white")
    sns.set_context("notebook")
    sns.set(font_scale=1.25)
    sns.heatmap(size_df, cmap="viridis", ax=axs[0], xticklabels=tick_spacing,
                yticklabels=tick_spacing, cbar_kws={'label': 'Droplet Size (um)'})
    axs[0].tick_params(axis='x', labelrotation=30)
    axs[0].scatter(len(size_df.columns)/2, len(size_df.columns)/2, marker="*", color="w", s=200)
    plt.setp(axs[0], xlabel="Oil Flow Rate (ml/hr)", ylabel="Water Flow Rate (uL/min)")

    sns.heatmap(rate_df, cmap="plasma", ax=axs[1], xticklabels=tick_spacing,
                yticklabels=tick_spacing, cbar_kws={'label': 'Generation Rate (Hz)'})
    axs[1].tick_params(axis='x', labelrotation=30)
    plt.setp(axs[1], xlabel="Oil Flow Rate (ml/hr)", ylabel="Water Flow Rate (uL/min)")
    axs[1].scatter(len(size_df.columns)/2, len(size_df.columns)/2, marker="*", color="w", s=200)
    return fig