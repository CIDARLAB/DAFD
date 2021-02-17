import matplotlib.pyplot as plt
import seaborn as sns
from DAFD.tolerance_study.tol_utils import *


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
    figsize = plt.figaspect(float(dx * 4) / float(dy * 8))

    fig, axs = plt.subplots(4, 5, constrained_layout=True,figsize=figsize, facecolor="w")

    axs[0, 0].axis('off')
    axs[2, 0].axis('off')
    axs[0, -1].axis('off')
    axs[1, -1].axis('off')
    axs[2, -1].axis('off')
    axs[3, -1].axis('off')

    # fig.subplots_adjust(right =0.9)
    ca_pos_size = axs[0][-1].get_position()
    ca_pos_rate = axs[1][-1].get_position()
    cbar_ax_size = fig.add_axes([ca_pos_size.x0+0.1, ca_pos_size.y0+0.15, 0.01, 0.3])
    cbar_ax_rate = fig.add_axes([ca_pos_rate.x0+0.1, ca_pos_rate.y0+0.075, 0.01, 0.3])
    hms = [hm_s, hm_g]
    for i in range(len(hm_s)):
        plot_s = plot_heatmap(hm_s[i],axs,cbar_ax_size, "Droplet Size", row=int(np.floor((i+1)/4)), col=(i+1)%4, vmax=size_max,map="viridis")
        plot_g = plot_heatmap(hm_g[i],axs,cbar_ax_rate, "Generation Rate", row=int(np.floor((i+1)/4))+2, col=(i+1)%4,vmax=rate_max,map='plasma')
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
    SMALL_SIZE = 12
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    plt.setp(axs[0], xlabel="Oil Flow Rate (ml/hr)", ylabel="Water Flow Rate (uL/min)", )

    sns.heatmap(rate_df, cmap="plasma", ax=axs[1], xticklabels=tick_spacing,
                yticklabels=tick_spacing, cbar_kws={'label': 'Generation Rate (Hz)'})
    axs[1].tick_params(axis='x', labelrotation=30)
    plt.setp(axs[1], xlabel="Oil Flow Rate (ml/hr)", ylabel="Water Flow Rate (uL/min)")
    axs[1].scatter(len(size_df.columns)/2, len(size_df.columns)/2, marker="*", color="w", s=200)
    return fig

def plot_heatmap_grid(data, axs,cbar_ax,label, row=0, col=0, vmax=None, map="magma"):
    SMALL_SIZE = 9
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    x_tick_spacing = int(np.floor(len(data.columns) / 5))
    y_tick_spacing = int(np.floor(len(data.columns) / 5))
    if col == 3:
        plot = sns.heatmap(data, ax=axs[row][col],
                           cbar=True, vmin=-vmax, vmax=vmax,xticklabels=x_tick_spacing, yticklabels=y_tick_spacing,
                           cmap = map, cbar_ax=cbar_ax, cbar_kws={'label': label+"\n(% change)"})
    else:
        plot = sns.heatmap(data, ax=axs[row][col],
                            cbar=False, vmin=-vmax, vmax=vmax,xticklabels=x_tick_spacing, yticklabels=y_tick_spacing,
                           cmap = map)
    axs[row][col].scatter(len(data.columns)/2, len(data.columns)/2, marker="*", color='w', s=100)
    plt.setp(axs[row][col], xlabel= str.capitalize(data.columns.name.replace("_", " ")) + "\n(% change)",
             ylabel= str.capitalize(data.index.name.replace("_", " ")) + "\n(% change)")
    return plot


def plot_heatmaps_grid(hm_s, hm_g, include_pcs=False, si=None, names=None):
    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    size_max = max([np.abs(df).max().max() for df in hm_s])
    rate_max = max([np.abs(df).max().max() for df in hm_g])

    figsize = plt.figaspect(566/839) * 1.5

    fig, axs = plt.subplots(4, 4, figsize=figsize, facecolor="w")
    cbar_ax_size = fig.add_axes([0.93, 0.625, 0.01, 0.3])
    cbar_ax_rate = fig.add_axes([0.93, 0.15, 0.01, 0.3])
    for i in range(len(hm_s)):
        plot_s = plot_heatmap_grid(hm_s[i],axs,cbar_ax_size, "Droplet Size", row=int(np.floor((i+1)/4)), col=(i+1)%4, vmax=size_max,map="viridis")
        plot_g = plot_heatmap_grid(hm_g[i],axs,cbar_ax_rate, "Generation Rate", row=int(np.floor((i+1)/4))+2, col=(i+1)%4,vmax=rate_max,map='plasma')
    plt.subplots_adjust(left=0.05, bottom=0.12, right=0.9, top=0.96, wspace=0.29, hspace=0.44)

    if include_pcs:
        edited_names = []
        for name in names:
            name = str.capitalize(name)
            name = name.replace("_", "\n")
            edited_names.append(name)
        output = ["(Size)", "(Rate)"]
        colors = ["#1f968b", "#c03a83"]
        bar_ax = [axs[0][0], axs[2][0]]
        sns.set_style("white")
        for i, ax in enumerate(bar_ax):
            sns.barplot(edited_names, si[i]["ST"], ax=ax, color=colors[i])
            # if i == 0:
            #     plt.setp(ax, xticks = [])
            plt.setp(ax, ylabel=("Total-Effect "+output[i]))
            bar_ax[i].tick_params(axis='x', labelrotation=90)
            ax.set_ylim(0, 1.1)
    return fig
def plot_half_heatmaps_grid(hm, output, include_pcs=False, si=None, names=None):
    if output == "Droplet Size":
        cmap = "viridis"
    else:
        cmap = "plasma"
    SMALL_SIZE = 9
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    hm_max = max([np.abs(df).max().max() for df in hm])
    #rate_max = max([np.abs(df).max().max() for df in hm_g])

    figsize = plt.figaspect(566/839/2)

    fig, axs = plt.subplots(2, 4, figsize=figsize, facecolor="w")
    cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
    #cbar_ax_rate = fig.add_axes([0.93, 0.15, 0.01, 0.3])
    for i in range(len(hm)):
        plot_s = plot_heatmap_grid(hm[i],axs,cbar_ax, output, row=int(np.floor((i+1)/4)), col=(i+1)%4, vmax=hm_max,map=cmap)
    #    plot_g = plot_heatmap_grid(hm_g[i],axs,cbar_ax_rate, "Generation Rate", row=int(np.floor((i+1)/4))+2, col=(i+1)%4,vmax=rate_max,map='plasma')
    plt.subplots_adjust(left=0.05, bottom=0.14, right=0.9, top=0.96, wspace=0.29, hspace=0.44)

    if include_pcs:
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('font', size=24)  # controls default text sizes

        edited_names = []
        for name in names:
            name = str.capitalize(name)
            name = name.replace("_", "\n")
            edited_names.append(name)
        if output == "Droplet Size":
            yax_str = "(Size)"
            color = "#1f968b"
        else:
            yax_str = "(Rate)"
            color = "#c03a83"
        sns.set_style("white")
        ax = axs[0][0]
        sns.barplot(edited_names, si["ST"], ax=ax, color=color)
        plt.setp(ax, ylabel=("Total-Effect "+yax_str))
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_ylim(0, 1.1)
    return fig