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

    if (inputDir / "gsf.json").exists():
        with open(inputDir / "gsf.json") as f:
            gsf_config = json.load(f)
    # Old format with all in one json
    else:
        gsf_config = run_config["gsf_config"]

    summary_gsf = uproot_to_pandas(
        uproot.open(str(inputDir / "root/tracksummary_gsf.root:tracksummary")),
    )

    summary_kf = uproot_to_pandas(
        uproot.open(str(inputDir / "root/tracksummary_kf.root:tracksummary")),
    )

    dirname = inputDir.name

    print("#" * (len(dirname) + 4))
    print("#", inputDir.name, "#")
    print("#" * (len(dirname) + 4), "\n")

    print("components:", gsf_config["maxComponents"])
    print("weight cutoff:", gsf_config["weightCutoff"])
    print("reduction:", gsf_config["componentMergeMethod"], "\n")

    if "particles" in run_config:
        total_tracks = run_config["events"] * run_config["particles"]
        for df, fitter in zip([summary_gsf, summary_kf], ["GSF", "KF"]):
            errors = total_tracks - len(df)
            print(
                "Error tracks {}: {} ({:.1%})".format(
                    fitter, errors, errors / total_tracks
                )
            )
        print("")

    gsf_no_outliers = summary_gsf[summary_gsf["nOutliers"] == 0]
    print_basic_statistics(
        [summary_gsf, gsf_no_outliers, summary_kf],
        ["GSF (all)".rjust(12), "GSF (no outliers)".rjust(18), "KF".rjust(12)],
    )

    print("WORST")
    print_worst(summary_gsf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="where the root/ and csv/ dirs are")
    args = vars(parser.parse_args())

    inputDir = Path(args["input_dir"])
    assert inputDir.exists()

    short_analysis(inputDir)
