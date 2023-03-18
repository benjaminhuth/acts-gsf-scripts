import tempfile
import os
import random
import argparse
import json
import multiprocessing
from pathlib import Path
from functools import partial
from contextlib import redirect_stdout
from datetime import datetime
from pprint import pprint

import pandas as pd
import numpy as np
from scipy.stats import bootstrap

from gsfanalysis.pandas_import import *
from gsfanalysis.core_tail_utils import *


def analyze_iteration(args, summary_gsf, timing):
    result_row = {}
    result_row["components"] = components
    result_row["weight_cutoff"] = weight_cutoff

    fitter_timing = timing[timing["identifier"] == "Algorithm:TrackFittingAlgorithm"]
    assert len(fitter_timing) == 1
    result_row["timing"] = float(fitter_timing["time_perevent_s"])

    summary_gsf = add_core_to_df_quantile(
        summary_gsf, "res_eQOP_fit", args["core_quantile"]
    )
    result_row["n_tracks"] = len(summary_gsf)

    summary_gsf_no_outliers = summary_gsf[summary_gsf["nOutliers"] == 0]
    result_row["n_outliers"] = len(summary_gsf) - len(summary_gsf_no_outliers)

    result_row["core_quantile"] = args["core_quantile"]

    local_coors = ["LOC0", "LOC1", "PHI", "THETA", "QOP"]

    if args["filter_outliers"]:
        summary_gsf = summary_gsf_no_outliers

    summary_gsf_core = summary_gsf[summary_gsf["is_core"]]

    for df, suffix in zip([summary_gsf, summary_gsf_core], ["", "_core"]):
        for coor in local_coors:
            res_key = f"res_e{coor}_fit"
            pull_key = f"pull_e{coor}_fit"

            for key, stat in [
                (res_key, np.mean),
                (res_key, rms),
                (pull_key, np.mean),
                (pull_key, np.std),
            ]:
                result_key = key.replace("fit", stat.__name__) + suffix

                # val = stat(df[key])
                # bootstrap_res = bootstrap((df[key],), stat)

                result_row[result_key] = stat(df[key])
                # result_row[result_key + "_err"] = bootstrap_res.standard_error

    return pd.DataFrame(result_row)


def gsf_initializer(args):
    from gsf_utils import GsfEnvironment

    global gsf_env
    gsf_env = GsfEnvironment(args)


def gsf_subprocess(args, pars):
    import acts.examples

    components, weight_cutoff = pars

    args["components"] = components
    args["cutoff"] = weight_cutoff

    print(
        datetime.now().strftime("%H:%M:%S"),
        multiprocessing.current_process().name,
        "PARS:",
        components,
        weight_cutoff,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        global gsf_env

        s = acts.examples.Sequencer(
            events=args["events"],
            numThreads=1,
            outputDir=tmp_dir,
            skip=0,
            logLevel=gsf_env.defaultLogLevel,
        )

        gsf_env.args = args
        gsf_env.run_sequencer(s, Path(tmp_dir))

        timing = pd.read_csv(os.path.join(tmp_dir, "timing.tsv"), sep="\t")

        summary_gsf = uproot_to_pandas(
            uproot.open(
                os.path.join(tmp_dir, "root/tracksummary_gsf.root:tracksummary")
            )
        )

    return analyze_iteration(args, summary_gsf, timing)


def run_in_pool(args, pars):
    # Try to make the load a bit more balanced this way...
    random.shuffle(pars)

    with multiprocessing.Pool(
        args["jobs"], initializer=partial(gsf_initializer, args)
    ) as p:
        dfs = p.map(partial(gsf_subprocess, args), pars, 1)

    return pd.concat(dfs, ignore_index=True)


def run_sequential(args, pars):
    import acts.examples
    from gsf_utils import GsfEnvironment

    result_df = pd.DataFrame()

    gsf_env = GsfEnvironment(args)

    for components, weight_cutoff in pars:
        with tempfile.TemporaryDirectory() as tmp_dir:
            s = acts.examples.Sequencer(
                events=args["events"],
                numThreads=args["jobs"],
                outputDir=tmp_dir,
                skip=0,
                logLevel=gsf_env.defaultLogLevel,
            )

            args["components"] = components
            args["cutoff"] = weight_cutoff

            gsf_env.args = args
            gsf_env.run_sequencer(s, Path(tmp_dir))

            del s

            timing = pd.read_csv(os.path.join(tmp_dir, "timing.tsv"), sep="\t")

            summary_gsf = uproot_to_pandas(
                uproot.open(
                    os.path.join(tmp_dir, "root/tracksummary_gsf.root:tracksummary")
                )
            )

        iteration_df = analyze_iteration(args, summary_gsf, timing)
        result_df = pd.concat([result_df, iteration_df], ignore_index=True)

        tmp_dir

    return result_df


if __name__ == "__main__":
    assert "ACTS_ROOT" in os.environ and Path(os.environ["ACTS_ROOT"]).exists()

    # fmt: off
    parser = argparse.ArgumentParser(description='Run GSF sweep')
    parser.add_argument('-n','--events', type=int, default=3)
    parser.add_argument('-j','--jobs', type=int, default=3)
    parser.add_argument('--particles', type=int, default=1000)
    parser.add_argument('--pmin', type=float, default=0.5)
    parser.add_argument('--pmax', type=float, default=20)
    parser.add_argument('--filter_outliers', action="store_true", default=False)
    parser.add_argument('--core_quantile', type=float, default=0.95)
    parser.add_argument('--fatras', action="store_true", default=False)
    parser.add_argument('-o', '--output', type=str, default="output_sweep")
    args = vars(parser.parse_args())
    # fmt: on

    # Add additional fixed args
    args["detector"] = "odd"
    args["erroronly"] = True
    args["verbose"] = False
    args["debug"] = False
    args["seeding"] = "smeared"
    args["pick"] = -1
    args["disable_fatras_interactions"] = False
    args["no_kalman"] = True
    args["fatras"] = True
    args["no_states"] = True

    pprint(args)

    # components = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    # weight_cutoff = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1]

    # components = [1, 4, 8, 12, 16]
    # weight_cutoff = [1.0e-8, 1.0e-4, 1.0e-2]

    components = [4, 4]
    weight_cutoff = [1.0e-4, 1.0e-4]

    print("Sweep components:", components)
    print("Sweep weight cutoffs:", weight_cutoff)

    # Combine to grid
    pars = []
    for c in components:
        for wc in weight_cutoff:
            pars.append((c, wc))

    if args["fatras"]:
        result_df = run_sequential(args, pars)
    else:
        result_df = run_in_pool(args, pars)

    print(result_df)

    out_dir = Path(args["output"])
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / "config.json", "w") as f:
        args["component_sweep"] = components
        args["weight_cutoff_sweep"] = weight_cutoff
        json.dump(args, f, indent=4)

    result_df.to_csv(out_dir / "result.csv", index=False)
