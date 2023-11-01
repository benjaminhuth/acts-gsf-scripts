import os
from pathlib import Path

import uproot
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar

from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *
from gsfanalysis.utils import *
from gsfanalysis.tracksummary_plots import *
from gsfanalysis.core_tail_utils import *

figsize = (10, 5)

summary_gsf = uproot_to_pandas(
    uproot.open(snakemake.input[0] + ":tracksummary"),
)

summary_kf = uproot_to_pandas(
    uproot.open(snakemake.input[1] + ":tracksummary"),
)

summary_gsf, summary_kf = select_particles_and_unify_index(summary_gsf, summary_kf)

# TODO not yet clear how this can be integrated
# core_quantile = 0.95
# summary_gsf = add_core_to_df_quantile(
#     summary_gsf, "res_eQOP_fit", core_quantile
# )

summary_gsf["res_eTHETA_fit"] *= 1000.0
summary_kf["res_eTHETA_fit"] *= 1000.0

summary_gsf["res_ePHI_fit"] *= 1000.0
summary_kf["res_ePHI_fit"] *= 1000.0

clip_map = {
    "d_0": (-0.6, 0.6),
    "z": (-0.60, 0.60),
    "\\varphi": (-30, 30),
    "\\theta": (-1.0, 1.0),
    "q/p": (-1, 0.5),
    "t": (-50, 50),
    "p": (-10, 10),
    "p norm": (-1, 1),
}

fig, ax = make_full_residual_plot(
    [summary_kf, summary_gsf],
    ["KF", "GSF"],
    clip_map=clip_map,
    log=True,
    p_pnorm=False,
    angle_unit="mrad"
)
for a in ax.flatten():
    a.set_ylim(max(10, a.get_ylim()[0]), a.get_ylim()[1])

fig.set_size_inches(*figsize)
fig.tight_layout()

if snakemake.config["plt_show"]:
    plt.show()

fig.savefig(snakemake.output[0])
