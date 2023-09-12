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

summary_gsf, summary_kf = remove_outliers_and_unify_index(
    summary_gsf, summary_kf
)

# TODO not yet clear how this can be integrated
# core_quantile = 0.95
# summary_gsf = add_core_to_df_quantile(
#     summary_gsf, "res_eQOP_fit", core_quantile
# )

clip_map = {
    "d_0": (-5,5),
    "z": (-10,10),
    "\\varphi": (-0.2,0.2),
    "\\theta": (-0.01,0.01),
    "q/p": (-4,4),
    "t": (-5000,5000),
    "p": (-10,10),
    "p norm": (-2,2)
}

fig, ax = make_full_residual_plot(
    [summary_kf, summary_gsf], ["KF", "GSF"], clip_map=clip_map
)
fig.set_size_inches(*figsize)
fig.tight_layout()
fig.savefig(snakemake.output[0])

