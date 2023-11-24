import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *

import awkward as ak

from utils_plotting import plot_binned_errorbar
from scipy.stats import bootstrap, binned_statistic


def ratio_hist(ax, df, mask, key, bins=20, hist_range=(-3, 3)):
    hist_a, bins = np.histogram(df[mask][key], bins=bins, range=hist_range)
    hist_b, bins = np.histogram(df[key], bins=bins, range=hist_range)

    return hist_a / hist_b, bins


df = uproot_to_pandas(
    uproot.open(f"{snakemake.input[1]}:tracksummary"),
).rename(columns={"event_nr": "event_id"})
print(df.keys())

df = df[ df.event_id < 100 ].copy()


def plot_binned(ax, binning_data, stat_data, bins, stat, **kwargs):
    vals, bins, _ = binned_statistic(binning_data, stat_data, stat, bins)
    print(bins)
    
    ax.step(bins[:-1], vals, **kwargs)


hole_mask =  df.nHoles > 0
outlier_mask = df.nOutliers > 0

fig, axes = plt.subplots(3,2,figsize=(12,8))

color = "tab:grey"
opts = dict(fmt="none", color=color)

eta_bins = np.linspace(-3,3,20)
pt_bins = np.linspace(0,100,20)

xbins = [eta_bins, pt_bins]
binning_keys = ["t_eta", "t_pT"]
labels = ["$\eta$", "$p_T$ [GeV]"]
ax_cols = [axes[:,0], axes[:,1] ]

for axrow, bin_key, label, bins in zip(ax_cols, binning_keys, labels, xbins):
    x = bins[:-1]+0.5*np.diff(bins)
    w = np.diff(bins)
    
    vals, errs, _ = plot_binned_errorbar(axrow[0], df[bin_key], df.nHoles, bins, 
                                            np.mean, label="mean holes", **opts)
    axrow[0].bar(x, vals, width=w, color=color, alpha=0.3)
    
    vals, errs, bins = plot_binned_errorbar(axrow[1], df[bin_key], df.nOutliers, bins, 
                                            np.mean, label="mean outliers", **opts)
    axrow[1].bar(x, vals, width=w, color=color, alpha=0.3)
    
    vals, errs, bins = plot_binned_errorbar(axrow[2], df[bin_key], df.nMeasurements, bins, 
                                            np.mean, label="mean measurements", **opts)
    axrow[2].bar(x, vals, width=w, color=color, alpha=0.3)
    
    axrow[0].set_ylim(0,0.1)
    axrow[1].set_ylim(0,1)
    axrow[2].set_ylim(10,15)
    
    if "eta" in bin_key:
        for ax in axrow:
            ax.set_xticks(np.arange(-3,4))
    
    for ax in axrow:
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel(label)
    
    axrow[0].set_title("$\\langle\\mathrm{holes}\\rangle$ vs " + label[:6])
    axrow[1].set_title("$\\langle\\mathrm{outliers}\\rangle$ vs " + label[:6])
    axrow[2].set_title("$\\langle\\mathrm{measurements}\\rangle$ vs " + label[:6])

fig.tight_layout()

if snakemake.config["plt_show"]:
    plt.show()

fig.savefig(snakemake.output[0])
