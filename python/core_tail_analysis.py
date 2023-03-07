#!/bin/python3

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import uproot
import pprint

from gsfanalysis.tracksummary_plots import make_full_residual_plot
from gsfanalysis.pandas_import import uproot_to_pandas
from gsfanalysis.core_tail_utils import *
from gsfanalysis.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="where the root/ and csv/ dirs are")
args = vars(parser.parse_args())

inputDir = Path(args["input_dir"])
assert inputDir.exists()

with open(inputDir / "config.json", 'r') as f:
    run_config = json.load(f)

summary_gsf, states_gsf = uproot_to_pandas(
    uproot.open(str(inputDir / "root/tracksummary_gsf.root:tracksummary")),
    uproot.open(str(inputDir / "root/trackstates_gsf.root:trackstates")),
)

summary_kf, states_kf = uproot_to_pandas(
    uproot.open(str(inputDir / "root/tracksummary_kf.root:tracksummary")),
    uproot.open(str(inputDir / "root/trackstates_kf.root:trackstates")),
)

summary_gsf_no_outliers = summary_gsf[ summary_gsf["nOutliers"] == 0 ]
n_outlier_tracks = len(summary_gsf) - len(summary_gsf_no_outliers)
print("Outlier tracks: {:.2%}".format(n_outlier_tracks/len(summary_gsf)))

# Remove outliers states
common_idx = summary_gsf_no_outliers.index.intersection(summary_kf.index)
summary_kf = summary_kf.loc[common_idx, :]
summary_gsf = summary_gsf.loc[common_idx, :]

# fig, ax = make_full_residual_plot([summary_kf, summary_gsf], ["KF", "GSF"])
# fig.tight_layout()

#############
# Core Tail #
#############

summary_gsf = add_core_to_df_quantile(summary_gsf, "res_eQOP_fit", 0.95)
summary_kf = add_core_to_df_quantile(summary_kf, "res_eQOP_fit", 0.95)

for rkey, pkey, coor in zip(
    ["res_eQOP_fit", "res_eLOC0_fit"],
    ["pull_eQOP_fit", "pull_eLOC0_fit"],
    [("qop", "1/GeV"), ("d_0", "mm")]
):
    fig, axes = plt.subplots(2,2)
    plot_core_tail_residuals(summary_gsf, rkey, coor, (fig, axes[0,:]))
    fig.suptitle("GSF Residuals ${}$".format(coor[0]))
    
    plot_core_tail_pulls(summary_gsf, pkey, coor, (fig, axes[1,:]))
    fig.suptitle("GSF Residuals ${}$".format(coor[0]))
    
    axes = add_gsf_run_infos(axes, run_config)
    axes = add_commit_hash(axes, run_config)
    fig.tight_layout()

plt.show()
