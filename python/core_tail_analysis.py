#!/bin/python3

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import uproot
import pprint

from gsfanalysis.tracksummary_plots import *
from gsfanalysis.pandas_import import *
from gsfanalysis.core_tail_utils import *
from gsfanalysis.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="where the root/ and csv/ dirs are")
args = vars(parser.parse_args())

inputDir = Path(args["input_dir"])
assert inputDir.exists()

with open(inputDir / "config.json", "r") as f:
    run_config = json.load(f)

summary_gsf_all, states_gsf = uproot_to_pandas(
    uproot.open(str(inputDir / "root/tracksummary_gsf.root:tracksummary")),
    uproot.open(str(inputDir / "root/trackstates_gsf.root:trackstates")),
)

summary_kf_all, states_kf = uproot_to_pandas(
    uproot.open(str(inputDir / "root/tracksummary_kf.root:tracksummary")),
    uproot.open(str(inputDir / "root/trackstates_kf.root:trackstates")),
)

summary_gsf, summary_kf = select_particles_and_unify_index(
    summary_gsf_all.copy(), summary_kf_all.copy()
)

summary_gsf = add_core_to_df_quantile(summary_gsf, "res_eQOP_fit", 0.95)
summary_kf = add_core_to_df_quantile(summary_kf, "res_eQOP_fit", 0.95)
summary_gsf_core = summary_gsf[summary_gsf["is_core"]]
print_basic_statistics(
    [summary_gsf_all, summary_gsf, summary_gsf_core, summary_kf],
    [
        "GSF (all)".rjust(12),
        "GSF (no outliers)".rjust(18),
        "GSF (no outliers)".rjust(25),
        "KF".rjust(12),
    ],
)

# Residual plot
fig, ax = make_full_residual_plot(
    [summary_kf, summary_gsf], ["KF", "GSF"], log=True, clip_std=10
)
fig.suptitle("Note: clip at +-10 stddev")
fig.tight_layout()

fig, ax = make_full_residual_plot(
    [summary_kf, summary_gsf], ["KF", "GSF"], log=False, clip_std=2
)
fig.suptitle("Note: clip at +-3 stddev")
fig.tight_layout()

#############
# Core Tail #
#############


for summary_df, fitter in zip([summary_gsf, summary_kf], ["GSF", "KF"]):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle(fitter)
    for rkey, pkey, coor, ax in zip(
        ["res_eQOP_fit", "res_eLOC0_fit"],
        ["pull_eQOP_fit", "pull_eLOC0_fit"],
        [("qop", "1/GeV"), ("d_0", "mm")],
        axes,
    ):
        plot_core_tail_residuals(
            summary_df, rkey, ["core", "tail", "all"], coor, (fig, ax[0])
        )
        ax[0].set_title("RES core & tail ${}$".format(coor[0]))
        plot_core_tail_residuals(summary_df, rkey, ["core"], coor, (fig, ax[1]))
        ax[1].set_title("RES core ${}$".format(coor[0]))

        plot_core_tail_pulls(summary_df, pkey, "all", coor, (fig, ax[2]))
        ax[2].set_title("PULL core & tail ${}$".format(coor[0]))
        plot_core_tail_pulls(summary_df, pkey, "core", coor, (fig, ax[3]))
        ax[3].set_title("PULL core ${}$".format(coor[0]))

    # axes = add_run_infos(axes, run_config)
    axes = add_commit_hash(axes, run_config)
    fig.tight_layout()

plt.show()
