import os
from pathlib import Path

import uproot
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *
from gsfanalysis.comparison_plot import *

print("matplotlibrc file:",matplotlib.matplotlib_fname())
print(plt.rcParams['font.family'], flush=True)

summary_gsf12 = uproot_to_pandas(
    uproot.open(snakemake.input[0] + ":tracksummary"),
)

summary_kf = uproot_to_pandas(
    uproot.open(snakemake.input[1] + ":tracksummary"),
)

summary_gsf1 = uproot_to_pandas(
    uproot.open(snakemake.input[2] + ":tracksummary"),
)


summary_gsf12, summary_kf, summary_gsf1 = select_particles_and_unify_index(summary_gsf12, summary_kf, summary_gsf1)

# Go to 1/MeV
for df in [summary_gsf12, summary_kf, summary_gsf1]:
    df["res_eQOP_fit"] *= 1000


gsf_vs_kf = [
    (summary_gsf12, "GSF", "tab:orange"),
    (summary_kf, "KF", "tab:blue"),
]

gsf_vs_gsf = [
    (summary_gsf12, "GSF (12)", "tab:orange"),
    (summary_gsf1, "GSF (1)", "tab:green"),
]

latex = {
    "res_eQOP_fit": "${qop}_{fit} - {qop}_{true} \quad [MeV^{-1}]$",
    "res_ePNORM_fit": "$({p}_{fit} - {p}_{true}) \;/\; p_{true}$",
    "pull_eQOP_fit": "$({qop}_{fit} - {qop}_{true}) \;/\; \sigma_{qop,fit}$",
}

title = {
    "res_eQOP_fit": "residuals q/p",
    "res_ePNORM_fit": "normalized residuals p",
    "pull_eQOP_fit": "pulls q/p",
}

legend_opts = dict(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
figsize = (10, 5)

# Residuals GSF vs KF
fig1, axes = plt.subplots(1, 2, figsize=figsize)
keys = ["res_eQOP_fit", "res_ePNORM_fit"]
ranges = [(-80,80), (-1,1)]


for ax, key, r in zip(axes, keys, ranges):
    plot_sets(ax, gsf_vs_kf, key, r, assymetric_interval=True, density=True)
    ax.set_xlabel(latex[key])
    ax.legend(**legend_opts)
    ax.set_title(f"GSF (12) vs. KF: {title[key]}")
    ax.set_ylabel("density [a.u.]")
    ax.yaxis.set_major_locator(plt.NullLocator())

# axes[0].set_xlim(-71,11)
fig1.tight_layout()

# plt.show()
# assert Falsedf


# Residuals GSF vs GSF
fig2, axes = plt.subplots(1, 2, figsize=figsize)
keys = ["res_eQOP_fit", "res_ePNORM_fit"]
ranges = [(-15,15), (-1,1)]

for ax, key, r in zip(axes, keys, ranges):
    plot_sets(ax, gsf_vs_gsf, key, r, assymetric_interval=True, density=True)
    ax.set_xlabel(latex[key])
    ax.legend(**legend_opts)
    ax.set_title(f"GSF (12) vs. GSF (1): {title[key]}")
    ax.set_ylabel("density [a.u.]")
    ax.yaxis.set_major_locator(plt.NullLocator())

fig2.tight_layout()

# Pulls
fig3, axes = plt.subplots(1, 2, figsize=figsize)
key = "pull_eQOP_fit"

plot_sets(axes[0], gsf_vs_kf, key, (-80,80), assymetric_interval=True, density=True)
axes[0].set_xlabel(latex[key])
axes[0].set_title(f"GSF (12) vs. KF: {title[key]}")
# axes[0].set_xlim(-71,11)

plot_sets(axes[1], gsf_vs_gsf, key, (-3,3), assymetric_interval=True, density=True)
axes[1].set_xlabel(latex[key])
axes[1].set_title(f"GSF (12) vs. GSF (1): {title[key]}")

for ax in axes:
    ax.legend(**legend_opts)
    ax.set_ylabel("density [a.u.]")
    ax.yaxis.set_major_locator(plt.NullLocator())


fig3.tight_layout()

if snakemake.config["plt_show"]:
    plt.show()

for i, fig in enumerate([fig1, fig2, fig3]):
    fig.savefig(snakemake.output[i], bbox_inches='tight')


