#!/bin/python3

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import uproot
import pprint

from gsfanalysis.tracksummary_plots import make_full_residual_plot
from gsfanalysis.pandas_import import uproot_to_pandas, remove_outliers_and_unify_index
from gsfanalysis.core_tail_utils import *
from gsfanalysis.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="where the root/ and csv/ dirs are")
args = vars(parser.parse_args())

inputDir = Path(args["input_dir"])
assert inputDir.exists()

with open(inputDir / "config.json", "r") as f:
    run_config = json.load(f)

summary_gsf, states_gsf = uproot_to_pandas(
    uproot.open(str(inputDir / "root/tracksummary_gsf.root:tracksummary")),
    uproot.open(str(inputDir / "root/trackstates_gsf.root:trackstates")),
)

summary_kf, states_kf = uproot_to_pandas(
    uproot.open(str(inputDir / "root/tracksummary_kf.root:tracksummary")),
    uproot.open(str(inputDir / "root/trackstates_kf.root:trackstates")),
)

nTracks = len(summary_gsf)
summary_gsf, summary_kf = remove_outliers_and_unify_index(summary_gsf, summary_kf)
print("Outliers: {:.2%}".format(1.0 - (len(summary_gsf) / nTracks)))

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
    [("qop", "1/GeV"), ("d_0", "mm")],
):
    fig, axes = plt.subplots(2, 2)
    plot_core_tail_residuals(
        summary_gsf, rkey, ["core", "tail", "all"], coor, (fig, axes[0, 0])
    )
    plot_core_tail_residuals(summary_gsf, rkey, ["core"], coor, (fig, axes[0, 1]))

    plot_core_tail_pulls(summary_gsf, pkey, "all", coor, (fig, axes[1, 0]))
    plot_core_tail_pulls(summary_gsf, pkey, "core", coor, (fig, axes[1, 1]))

    fig.suptitle("GSF Residuals ${}$".format(coor[0]))

    # axes = add_gsf_run_infos(axes, run_config)
    axes = add_run_infos(axes, run_config)
    fig.tight_layout()

plt.show()
