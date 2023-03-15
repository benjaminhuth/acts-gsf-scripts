#!/bin/python3

from pathlib import Path
from pprint import pprint
import argparse
import json

import matplotlib.pyplot as plt
import uproot

from gsfanalysis.tracksummary_plots import *
from gsfanalysis.pandas_import import *
from gsfanalysis.core_tail_utils import *
from gsfanalysis.utils import *


def short_analysis(inputDir: Path):
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

    dirname = inputDir.name

    print("#" * (len(dirname) + 4))
    print("#", inputDir.name, "#")
    print("#" * (len(dirname) + 4), "\n")

    print("components:", run_config["gsf"]["maxComponents"])
    print("weight cutoff:", run_config["gsf"]["weightCutoff"])
    print("reduction:", run_config["gsf"]["finalReductionMethod"], "\n")

    if "particles" in run_config:
        total_tracks = run_config["events"] * run_config["particles"]
        for df, fitter in zip([summary_gsf, summary_kf], ["GSF", "KF"]):
            errors = total_tracks - len(df)
            print(
                "Error tracks {}: {} ({:.1%}))".format(fitter, errors, errors / len(df))
            )
        print("")

    gsf_no_outliers = summary_gsf[summary_gsf["nOutliers"] == 0]
    print_basic_statistics(
        [summary_gsf, gsf_no_outliers, summary_kf],
        ["GSF (all)".rjust(12), "GSF (no outliers)".rjust(18), "KF".rjust(12)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="where the root/ and csv/ dirs are")
    args = vars(parser.parse_args())

    inputDir = Path(args["input_dir"])
    assert inputDir.exists()

    short_analysis(inputDir)
