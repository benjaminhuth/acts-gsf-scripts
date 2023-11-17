import os
from pathlib import Path

import uproot
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.ndimage import gaussian_filter1d

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *
from gsfanalysis.utils import *
from gsfanalysis.tracksummary_plots import *
from gsfanalysis.core_tail_utils import *


plt.rcParams['font.family'] = "serif"


def hist2d_smoothed(ax, a, b, bins, cmap, norm=None):
    H, xedges, yedges = np.histogram2d(a, b, bins=bins)

    ax.imshow(
        H.T,
        interpolation="gaussian",
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=norm,
        cmap=cmap,
        aspect="auto",
    )

    H = H.astype(int)

    x = xedges[:-1] + np.diff(xedges)
    y = yedges[:-1] + np.diff(yedges)

    accumulated_vals = [
        np.concatenate([count * [v] for v, count in zip(y, H[i, :])])
        for i in range(H.shape[0])
    ]
    means = np.array([np.mean(array) for array in accumulated_vals])
    # modes = [ y[np.argmax(H[i,:])] for i in range(H.shape[0]) ]
    stddevs = np.array([np.std(array) for array in accumulated_vals])

    ax.plot(x, gaussian_filter1d(means, sigma=2), color="black")
    ax.plot(x, gaussian_filter1d(means + stddevs, sigma=2), color="grey")
    ax.plot(x, gaussian_filter1d(means - stddevs, sigma=2), color="grey")
    # ax.plot(x, , color="black")


# fmt: off
summary_gsf = uproot_to_pandas(uproot.open(snakemake.input[0] + ":tracksummary"))
summary_kf = uproot_to_pandas(uproot.open(snakemake.input[1] + ":tracksummary"))
# fmt: on

summary_gsf, summary_kf = select_particles_and_unify_index(summary_gsf, summary_kf)

# Go to 1/MeV
for df in [summary_gsf, summary_kf]:
    df["res_eQOP_fit"] *= 1000

gsf_cmap = LinearSegmentedColormap.from_list(
    "gsf", [to_rgba("tab:orange", alpha=0), to_rgba("tab:orange", alpha=1)]
)

kf_cmap = LinearSegmentedColormap.from_list(
    "kf", [to_rgba("tab:blue", alpha=0), to_rgba("tab:blue", alpha=1)]
)

# fmt: off
bins = (50, 20)
res_qop_cut = 5 # 1/MeV
hline_args = dict(colors=["black"],linestyles=["dotted"],linewidths=[1,],)
# fmt: on

plt.rcParams['font.family'] = "serif"
fig, axes = plt.subplots(2, 2, figsize=(10,7))

binning_keys = ["t_pT", "t_eta"]
xlabels = ["$p_T$ [GeV]", "$\eta$"]

dfs = [summary_gsf, summary_kf]
titles = ["GSF (12 components)", "KF"]
cmaps = [gsf_cmap, kf_cmap]

for axrow, binning, xlabel in zip(axes, binning_keys, xlabels):    
    for ax, df, title, cmap in zip(axrow, dfs, titles, cmaps):
        mask = abs(df.res_eQOP_fit) < res_qop_cut
        
        hist2d_smoothed(
            ax,
            df[binning][mask],
            df.res_eQOP_fit[mask],
            bins=bins,
            # norm=LogNorm(),
            cmap=cmap,
        )
        # fmt: off
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$q/p_{fit} - q/p_{true}\quad[MeV^{-1}]$")
        ax.hlines([0,], *ax.get_xlim(), **hline_args)
        # fmt: on

fig.tight_layout()

if snakemake.config["plt_show"]:
    plt.show()

fig.savefig(snakemake.output[0])
