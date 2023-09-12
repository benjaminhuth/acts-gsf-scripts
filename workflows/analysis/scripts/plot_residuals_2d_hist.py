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


gsf_cmap = LinearSegmentedColormap.from_list(
    "gsf", [to_rgba("tab:orange", alpha=0), to_rgba("tab:orange", alpha=1)]
)

kf_cmap = LinearSegmentedColormap.from_list(
    "kf", [to_rgba("tab:blue", alpha=0), to_rgba("tab:blue", alpha=1)]
)

print(summary_gsf.keys())

print(max(summary_gsf.t_pT))

bins=(30,30)
cut = 0.05

mask = abs(summary_gsf.res_eQOP_fit) < cut

fig, ax = plt.subplots(1,2)

# ax[0].hist2d(summary_gsf.t_pT[mask], summary_gsf.res_eQOP_fit[mask],
#              bins=bins,
#              # norm=LogNorm(),
#              cmap=gsf_cmap)


h, xe, ye = np.histogram2d(summary_gsf.t_pT[mask], summary_gsf.res_eQOP_fit[mask], bins=bins)

ax[0].imshow(h, cmap=gsf_cmap, interpolation="gaussian", extent=[0,10,-cut, cut])

mask = abs(summary_kf.res_eQOP_fit) < cut


ax[1].hist2d(summary_kf.t_pT[mask], summary_kf.res_eQOP_fit[mask],
             bins=bins,
             # norm=LogNorm(),
             cmap=kf_cmap)
plt.show()



