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
from gsfanalysis.comparison_plot import *

figsize = (10, 5)

summary_a = uproot_to_pandas(
    uproot.open(snakemake.input[0] + ":tracksummary"),
)

summary_b = uproot_to_pandas(
    uproot.open(snakemake.input[1] + ":tracksummary"),
)

summary_a, summary_b = remove_outliers_and_unify_index(
    summary_a, summary_b
)


name_a, color_a = snakemake.params["config_a"]
name_b, color_b = snakemake.params["config_b"]

gsf_vs_kf = [
    (summary_a, name_a, color_a),
    (summary_b, name_b, color_b),
]

fig, ax = make_gsf_detailed_comparison_plots(
    gsf_vs_kf, assymetric_interval=True
)
fig.suptitle(snakemake.params["suptitle"])
fig.tight_layout()
fig.savefig(snakemake.output[0])
