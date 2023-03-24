import tempfile
import os
import random
import argparse
import warnings
import json
import multiprocessing
from pathlib import Path
from functools import partial
from contextlib import redirect_stdout
from datetime import datetime
from pprint import pprint

from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

from gsfanalysis.pandas_import import *
from gsfanalysis.core_tail_utils import *
import gsfanalysis.statistics as stats


def symmetric_q95(x):
    return np.quantile(abs(x - stats.mode(x)), 0.95)


def symmetric_q68(x):
    return np.quantile(abs(x - stats.mode(x)), 0.68)


def analyze_iteration(args, summary_gsf, timing):
    result_row = {}
    result_row["components"] = args["components"]
    result_row["weight_cutoff"] = args["cutoff"]

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

    local_coors = ["LOC0", "LOC1", "PHI", "THETA", "QOP", "P", "PNORM"]
    local_coors = ["QOP", "P", "PNORM"]

    if args["filter_outliers"]:
        summary_gsf = summary_gsf_no_outliers

    summary_gsf_core = summary_gsf[summary_gsf["is_core"]]

    for df, suffix in zip([summary_gsf, summary_gsf_core], ["", "_core"]):
        for coor in local_coors:
            res_key = f"res_e{coor}_fit"
            pull_key = f"pull_e{coor}_fit"

            for key, stat in [
                (res_key, np.mean),
                (res_key, stats.rms),
                (res_key, stats.mode),
                (res_key, symmetric_q95),
                (res_key, symmetric_q68),
                (pull_key, np.mean),
                (pull_key, np.std),
            ]:
                # Don't have pulls for P and PNORM
                if "eP" in key and "pull" in key:
                    continue

                result_key = key.replace("fit", stat.__name__) + suffix

                values = df[key]
                sanitize_threshold = 1.0e8
                sanitize_mask = values.between(-sanitize_threshold, sanitize_threshold)

                if sum(sanitize_mask) < len(df):
                    print(
                        f"WARNING   unreasonable high/low values encountered for {key} / {stat.__name__}"
                    )
                    print(f"WARNING   {df[~sanitize_mask][key].to_numpy()}")
                    print(
                        f"WARNING   clip these to +-{sanitize_threshold} to prevent overflow"
                    )

                    values = np.clip(values, -sanitize_threshold, sanitize_threshold)

                # print(key, stat.__name__)
                val = stat(values)
                result_row[result_key] = val

                err = bootstrap((values,), stat).standard_error
                result_row[result_key + "_err"] = err
                # print("-> err",err)

    return pd.DataFrame({key: [result_row[key]] for key in result_row.keys()})


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
        "{} - {} - cmps: {}, wc: {}".format(
            datetime.now().strftime("%H:%M:%S"),
            multiprocessing.current_process().name,
            components,
            weight_cutoff,
        )
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        global gsf_env

        s = acts.examples.Sequencer(
            events=args["events"],
            numThreads=min(args["events"], 5) if args["fatras"] else 1,
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
        print(
            "{} - cmps: {}, wc: {}".format(
                datetime.now().strftime("%H:%M:%S"),
                components,
                weight_cutoff,
            )
        )

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
    parser.add_argument("--detector", choices=["telescope", "odd"], default="odd")
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
    args["surfaces"] = 10
    args["digi_smearing"] = 0.01
    args["erroronly"] = True
    args["verbose"] = False
    args["debug"] = False
    args["seeding"] = "smeared"
    args["pick"] = -1
    args["disable_fatras_interactions"] = False
    args["no_kalman"] = True
    args["fatras"] = True
    args["no_states"] = True

    print("Config:")
    pprint(args)

    # components = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    # weight_cutoff = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1]

    components = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    weight_cutoff = [1.0e-8, 1.0e-6, 1.0e-4]

    # components = [1, 4, 8, 12, 16]
    # weight_cutoff = [1.0e-8, 1.0e-4, 1.0e-2]

    # components = [12]
    # weight_cutoff = [1.0e-6]

    print("Sweep components:", components)
    print("Sweep weight cutoffs:", weight_cutoff)

    # Combine to grid
    pars = []
    for c in components:
        for wc in weight_cutoff:
            pars.append((c, wc))

    print("Grid size:", len(pars))

    if args["jobs"] == 1:  # args["fatras"] and False:
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
