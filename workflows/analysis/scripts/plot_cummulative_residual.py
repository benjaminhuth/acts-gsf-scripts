import os
from pathlib import Path

import uproot
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from gsfanalysis.pandas_import import *
from gsfanalysis.comparison_plot import *

summary_gsf12 = uproot_to_pandas(
    uproot.open(snakemake.input[0] + ":tracksummary"),
)

summary_kf = uproot_to_pandas(
    uproot.open(snakemake.input[1] + ":tracksummary"),
)


summary_gsf12, summary_kf = select_particles_and_unify_index(summary_gsf12, summary_kf)


KEY="res_eQOP_fit"

def plot(ax, df, **kwargs):
    absvals = abs(df[KEY])
    m = max(absvals)

    xs = np.logspace(np.log10(1e-3), np.log10(m), 20)
    ys = [100.0*sum(absvals < x)/len(df) for x in xs ]

    p = ax.plot(xs, ys, **kwargs)[0]
    ax.vlines(m, ymin=90, ymax=100, ls="--", color=p._color)
    ax.set_xscale('log')

fig, ax = plt.subplots(1,figsize=(12,5))
plot(ax, summary_gsf12, label="GSF(12)", color="tab:orange")
plot(ax, summary_kf, label="KF", color="tab:blue")

ax.set_title("Cummulative Residual distribution")
ax.set_xlabel("|res_qop|")
ax.set_ylabel("% below")

# plt.show()
fig.savefig(snakemake.output[0])

