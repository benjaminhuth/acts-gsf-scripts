import os
from pathlib import Path

import uproot
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.ndimage import gaussian_filter1d

from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *
from gsfanalysis.utils import *
from gsfanalysis.tracksummary_plots import *
from gsfanalysis.core_tail_utils import *


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


summary_gsf = uproot_to_pandas(
    uproot.open(snakemake.input[0] + ":tracksummary"),
)

summary_kf = uproot_to_pandas(
    uproot.open(snakemake.input[1] + ":tracksummary"),
)

summary_gsf, summary_kf = remove_outliers_and_unify_index(summary_gsf, summary_kf)


gsf_cmap = LinearSegmentedColormap.from_list(
    "gsf", [to_rgba("tab:orange", alpha=0), to_rgba("tab:orange", alpha=1)]
)

kf_cmap = LinearSegmentedColormap.from_list(
    "kf", [to_rgba("tab:blue", alpha=0), to_rgba("tab:blue", alpha=1)]
)

bins = (50, 20)
res_qop_cut = 0.005
hline_args = dict(
    colors=[
        "black",
    ],
    linestyles=[
        "dotted",
    ],
    linewidths=[
        1,
    ],
)

figsize = (10, 5)
fig, ax = plt.subplots(2, 2)

# GSF - res QOP vs pT
mask = abs(summary_gsf.res_eQOP_fit) < res_qop_cut
hist2d_smoothed(
    ax[0, 0],
    summary_gsf.t_pT[mask],
    summary_gsf.res_eQOP_fit[mask],
    bins=bins,
    # norm=LogNorm(),
    cmap=gsf_cmap,
)
ax[0, 0].set_title("GSF 20 cmp")
ax[0, 0].set_xlabel("pT [GeV]")
ax[0, 0].set_ylabel("$q/p_{fit} - q/p_{true}$")
ax[0, 0].hlines(
    [
        0,
    ],
    *ax[0, 0].get_xlim(),
    **hline_args
)


# KF - res QOP vs pT
mask = abs(summary_kf.res_eQOP_fit) < res_qop_cut
hist2d_smoothed(
    ax[0, 1],
    summary_kf.t_pT[mask],
    summary_kf.res_eQOP_fit[mask],
    bins=bins,
    # norm=LogNorm(),
    cmap=kf_cmap,
)
ax[0, 1].set_title("KF")
ax[0, 1].set_xlabel("pT [GeV]")
ax[0, 1].set_ylabel("$q/p_{fit} - q/p_{true}$")
ax[0, 1].hlines(
    [
        0,
    ],
    *ax[0, 1].get_xlim(),
    **hline_args
)


# GSF - res QOP vs eta
mask = abs(summary_gsf.res_eQOP_fit) < res_qop_cut
hist2d_smoothed(
    ax[1, 0],
    summary_gsf.t_eta[mask],
    summary_gsf.res_eQOP_fit[mask],
    bins=bins,
    # norm=LogNorm(),
    cmap=gsf_cmap,
)
ax[1, 0].set_title("GSF 20 cmp")
ax[1, 0].set_xlabel("$\eta$")
ax[1, 0].set_xlim(-3.1, 3.1)
ax[1, 0].set_xticks(np.arange(-3, 4))
ax[1, 0].set_ylabel("$q/p_{fit} - q/p_{true}$")
ax[1, 0].hlines(
    [
        0,
    ],
    *ax[1, 0].get_xlim(),
    **hline_args
)


# KF - res QOP vs eta
mask = abs(summary_kf.res_eQOP_fit) < res_qop_cut
hist2d_smoothed(
    ax[1, 1],
    summary_kf.t_eta[mask],
    summary_kf.res_eQOP_fit[mask],
    bins=bins,
    # norm=LogNorm(),
    cmap=kf_cmap,
)
ax[1, 1].set_title("KF")
ax[1, 1].set_xlabel("$\eta$")
ax[1, 1].set_xticks(np.arange(-3, 4))
ax[1, 1].set_xlim(-3.1, 3.1)
ax[1, 1].set_ylabel("$q/p_{fit} - q/p_{true}$")
ax[1, 1].hlines(
    [
        0,
    ],
    *ax[1, 1].get_xlim(),
    **hline_args
)

fig.tight_layout()

if snakemake.config["plt_show"]:
    plt.show()

fig.savefig(snakemake.output[0])
