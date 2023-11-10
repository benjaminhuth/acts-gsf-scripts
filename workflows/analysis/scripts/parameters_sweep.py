from collections import namedtuple
from pathlib import Path
import tempfile
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

from gsfanalysis.pandas_import import *
from gsfanalysis.core_tail_utils import *
import gsfanalysis.statistics as stats

import acts
import acts.examples

from utils_fitting import run_fitting

def symmetric_q95(x):
    return np.quantile(abs(x - stats.mode(x)), 0.95)


def symmetric_q68(x):
    return np.quantile(abs(x - stats.mode(x)), 0.68)

athena_dir = Path("/home/benjamin/Documents/athena")
gsf_data_dir = athena_dir / "Tracking/TrkFitter/TrkGaussianSumFilter/Data"
low_bhapprox = gsf_data_dir / "GeantSim_LT01_cdf_nC6_O5.par"
high_bhapprox = gsf_data_dir / "GeantSim_GT01_cdf_nC6_O5.par"

def run_configuration(pars):
    components, weight_cutoff, componentMergeMethod, reductionAlgorithm = pars
    print(f"-> cmps: {components}, wc: {weight_cutoff}, mm: {componentMergeMethod}, ra: {reductionAlgorithm}")
    
    # Be robust for refactored name or current name in main
    MergeMethodEnum = None
    try:
        MergeMethodEnum = acts.examples.ComponentMergeMethod
        mergeMethodKey = "componentMergeMethod"
    except:
        MergeMethodEnum = acts.examples.FinalReductionMethod
        mergeMethodKey = "finalReductionMethod"

    try:
        componentMergeMethod = vars(MergeMethodEnum)[componentMergeMethod]
    except:
        print(f"WARNING: cannot use merge method '{componentMergeMethod}', not implemented")
        return pd.DataFrame()

    opts = {
        "maxComponents": components,
        "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
            str(low_bhapprox),
            str(high_bhapprox),
        ),
        mergeMethodKey: componentMergeMethod,
        "weightCutoff": weight_cutoff,
        "level": acts.logging.ERROR,
        "mixtureReductionAlgorithm" : vars(acts.examples.MixtureReductionAlgorithm)[reductionAlgorithm],
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run_fitting(
                "gsf",
                acts.examples.makeGsfFitterFunction,
                opts,
                tmp,
                snakemake.input[0],
                snakemake.input[1],
                snakemake.params["n_events"],
            )

            timing = pd.read_csv(tmp / "timing.tsv", sep="\t")

            summary_gsf = uproot_to_pandas(
                uproot.open(str(tmp / "tracksummary_gsf.root:tracksummary"))
            )

        result_row = {}
        result_row["components"] = components
        result_row["weight_cutoff"] = weight_cutoff
        result_row["component_merge_method"] = str(opts[mergeMethodKey])[21:]
        result_row["mixture_reduction"] = str(reductionAlgorithm)

        fitter_timing = timing[timing["identifier"] == "Algorithm:TrackFittingAlgorithm"]
        assert len(fitter_timing) == 1
        result_row["timing"] = float(fitter_timing["time_perevent_s"].iloc[0])

        result_row["n_tracks"] = len(summary_gsf)
        result_row["n_outliers"] = sum(summary_gsf["nOutliers"] > 0)
        result_row["n_holes"] = sum(summary_gsf["nHoles"] > 0)

        local_coors = ["QOP", "P", "PNORM"]

        if snakemake.params["apply_selection"]:
            print("INFO: Apply selection in sweep!")
            summary_gsf = select_particles_and_unify_index(summary_gsf.copy())

        for coor in local_coors:
            res_key = f"res_e{coor}_fit"
            pull_key = f"pull_e{coor}_fit"

            for key, stat in [
                (res_key, np.mean),
                (res_key, np.std),
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

                result_key = key.replace("fit", stat.__name__)

                values = summary_gsf[key]
                sanitize_threshold = 1.0e8
                sanitize_mask = values.between(-sanitize_threshold, sanitize_threshold)

                if sum(sanitize_mask) < len(summary_gsf):
                    # fmt: off
                    print(f"WARNING   unreasonable high/low values encountered for {key} / {stat.__name__}")
                    print(f"WARNING   {summary_gsf[~sanitize_mask][key].to_numpy()}")
                    print(f"WARNING   clip these to +-{sanitize_threshold} to prevent overflow")
                    # fmt: on

                    values = np.clip(values, -sanitize_threshold, sanitize_threshold)

                # print(key, stat.__name__)
                val = stat(values)
                result_row[result_key] = val

                err = bootstrap((values,), stat).standard_error
                result_row[result_key + "_err"] = err
                # print("-> err",err)

        iteration_df = pd.DataFrame({key: [result_row[key]] for key in result_row.keys()})
        # result_df = pd.concat([result_df, iteration_df], ignore_index=True)
        return iteration_df
    except:
        print("FAILED",pars)
        return pd.DataFrame()


pars = []

components = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
print("Sweep components:", components)
for c in components:
    pars.append((c, 1.0e-6, "maxWeight", "KLDistanceQoP"))

# 1.e-6 already covered above
weight_cutoff = [1.0e-8, 1.0e-4, 1.0e-2, 1.0e-1]
print("Sweep weight cutoffs:", weight_cutoff)
for w in weight_cutoff:
    pars.append((12, w, "maxWeight", "KLDistanceQoP"))

pars.append((12, 1.0e-6, "mean", "KLDistanceQoP"))
# pars.append((12, 1.0e-6, "mode", "KLDistanceQoP"))

pars.append((12, 1.0e-6, "maxWeight", "weightCut"))
pars.append((12, 1.0e-6, "maxWeight", "KLDistanceFull"))

print("Grid size:", len(pars))

with multiprocessing.Pool(snakemake.config["sim_jobs"]) as pool:
    result_dfs = pool.map(run_configuration, pars)

result_df = pd.concat(result_dfs, ignore_index=True)
result_df.to_csv(snakemake.output[0], index=False)
