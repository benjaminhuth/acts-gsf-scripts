#!/bin/python3

from pathlib import Path
from typing import Union, Optional
import math
import pprint
import json
import os
import sys
from tempfile import NamedTemporaryFile
import subprocess
import textwrap
import shutil
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uproot

import acts
import acts.examples

import gsf_utils

u = acts.UnitConstants

#####################
# Cmd line handling #
#####################

assert "ACTS_ROOT" in os.environ and Path(os.environ["ACTS_ROOT"]).exists()

# fmt: off
parser = argparse.ArgumentParser(description='Run GSF')
parser.add_argument("--config_file", type=str, default="")
parser.add_argument("--detector", choices=["telescope", "odd", "sphenix"], default="odd", help="which detector geometry to use")
parser.add_argument('-p','--pick', help='pick track', type=int, default=-1)
parser.add_argument('-n','--events', help='number of events', type=int, default=1)
parser.add_argument('-j','--jobs', help='number of jobs', type=int, default=-1)
parser.add_argument('-s','--skip', help='number of skipped events', type=int, default=0)
parser.add_argument('-c','--components', help='number of GSF components', type=int, default=4)
parser.add_argument('--cutoff',help='weight cutoff of GSF components', type=float, default=1.e-4)
parser.add_argument('--surfaces', help='number of telescope surfaces', type=int, default=10)
parser.add_argument('--particles', help='particles per event', type=int, default=1000)
parser.add_argument('--pmin', help='minimum momentum for particle gun', type=float, default=4.0)
parser.add_argument('--pmax', help='maximum momentum for particle gun', type=float, default=4.0)
parser.add_argument('--erroronly', help='set loglevel to show only errors (except sequencer)', default=False, action="store_true")
parser.add_argument('--debug', help='set loglevel to show debug', default=False, action="store_true")
parser.add_argument('-v', '--verbose', help='set loglevel to VERBOSE (except for sequencer)', default=False, action="store_true")
parser.add_argument('--fatras', help='use FATRAS instead of Geant4', default=False, action="store_true")
parser.add_argument('--particle_smearing', help='value used for initial particle smearing', type=float, default=None)
parser.add_argument('--telescope_digi_smearing', help='stddev for the pixel smearing', type=float, default=0.01)
parser.add_argument('--plt_show', help='Call plt.show() in the end', action="store_true", default=False)
parser.add_argument('--disable_fatras_interactions', help="no  interactions in FATRAS", default=False, action="store_true")
parser.add_argument('-o', '--output', help="Override default output dir")
parser.add_argument('--no_states', help="Don't write trackstates", default=False, action="store_true")
parser.add_argument('--no_kalman', help="Don't simulate kalman fitter", default=False, action="store_true")
parser.add_argument('--skip_analysis', default=False, action="store_true")
parser.add_argument('--seeding', default="smeared", choices=["smeared", "truth", "estimated"])
args = vars(parser.parse_args())
# fmt: on

if len(args["config_file"]) > 0:
    print(args["config_file"])
    input_config = Path(args["config_file"])
    assert input_config.exists()

    with open(input_config) as f:
        file_args = json.load(f)

    # only override what was specified on command line (the non-defaults)
    override = {
        args[key]
        for key in args.keys()
        if args[key] != parser.get_default(key) and key != "config_file"
    }
    if len(override) > 0:
        print("Override input file with:")
        pprint.pprint(override)
        file_args = {**file_args, **override}

    args = file_args


if args["output"] is not None:
    outputDir = Path(args["output"])
else:
    outputDir = Path.cwd() / "output_{}".format(args["detector"])


##################
# Build geometry #
##################

gsf_environment = gsf_utils.GsfEnvironment(args)

s = acts.examples.Sequencer(
    events=args["events"],
    numThreads=args["jobs"] if args["fatras"] else 1,
    outputDir=str(outputDir),
    skip=args["skip"],
    logLevel=acts.logging.INFO,
)

s.addWriter(
    acts.examples.CsvTrackingGeometryWriter(
        level=gsf_environment.defaultLogLevel,
        trackingGeometry=gsf_environment.trackingGeometry,
        outputDir=str(outputDir / "csv"),
        writePerEvent=False,
    )
)

gsfConfig = gsf_environment.run_sequencer(s, outputDir)
del s

# Bring geometry file to top for convenience
shutil.copyfile(outputDir / "csv/detectors.csv", Path.cwd() / "detectors.csv")

# Dump detailed GSF config seperately
with open(outputDir / "gsf.json", "w") as f:
    json.dump(gsfConfig, f, indent=4)

# Save configuration
with open(outputDir / "config.json", "w") as f:
    actsCommitHash = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=os.environ["ACTS_ROOT"],
        )
        .stdout.decode("utf-8")
        .rstrip()
    )
    args["acts-commit-hash"] = actsCommitHash

    json.dump(args, f, indent=4)

if args["skip_analysis"]:
    exit()

############
# Analysis #
############

from default_analysis import default_analysis
from short_analysis import short_analysis

# pdfreport = PdfPages(outputDir / "report.pdf")
# main_direction = "x" if args["detector"] == "telescope" else "r"
#
# default_analysis(
#     outputDir,
#     main_direction,
#     pmax=args["pmax"],
#     pick_track=args["pick"],
#     pdfreport=pdfreport,
# )

# pdfreport.close()

if args["pick"] == -1:
    short_analysis(outputDir)

# if args["plt_show"]:
#     plt.show()
