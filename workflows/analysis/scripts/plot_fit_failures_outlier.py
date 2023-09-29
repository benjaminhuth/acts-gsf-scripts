import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *

import awkward as ak

def ratio_hist(ax, df, mask, key, bins = 20, hist_range=(-3,3)):
    hist_a, bins = np.histogram(df[ mask ][key], bins=bins, range=hist_range)
    hist_b, bins = np.histogram(df[key], bins=bins, range=hist_range)

    hist_a / hist_b

    ax.bar(bins[:-1], hist_a / hist_b, width=np.diff(bins))


summary_gsf = uproot_to_pandas(
    uproot.open(f"{snakemake.input[1]}:tracksummary"),
).rename(columns={"event_nr": "event_id"})

particles = ak.to_dataframe(uproot.open(f"{snakemake.input[0]}:particles").arrays()).reset_index(drop=True)

print(particles.keys())
print(summary_gsf.keys())

print(len(summary_gsf)/len(particles))


summary_gsf["has_outliers"] = summary_gsf.nOutliers > 0



plt.show()
