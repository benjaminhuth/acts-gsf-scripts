from collections import namedtuple
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

from gsfanalysis.pandas_import import *
from gsfanalysis.core_tail_utils import *
import gsfanalysis.statistics as stats

from utils import *


def symmetric_q95(x):
    return np.quantile(abs(x - stats.mode(x)), 0.95)


def symmetric_q68(x):
    return np.quantile(abs(x - stats.mode(x)), 0.68)


athena_dir = Path("/home/benjamin/Documents/athena")
gsf_data_dir = athena_dir / "Tracking/TrkFitter/TrkGaussianSumFilter/Data"
low_bhapprox = gsf_data_dir / "GeantSim_LT01_cdf_nC6_O5.par"
high_bhapprox = gsf_data_dir / "GeantSim_GT01_cdf_nC6_O5.par"

components = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
weight_cutoff = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2]

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

result_df = pd.DataFrame()

for components, weight_cutoff in pars:
    print("-> cmps: {}, wc: {}".format(components, weight_cutoff))

    opts = {
        "maxComponents": components,
        "abortOnError": False,
        "disableAllMaterialHandling": False,
        "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
            str(low_bhapprox),
            str(high_bhapprox),
        ),
        "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
        "weightCutoff": weight_cutoff,
        "level": acts.logging.ERROR,
    }

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

    fitter_timing = timing[timing["identifier"] == "Algorithm:TrackFittingAlgorithm"]
    assert len(fitter_timing) == 1
    result_row["timing"] = float(fitter_timing["time_perevent_s"].iloc[0])

    summary_gsf = add_core_to_df_quantile(
        summary_gsf, "res_eQOP_fit", snakemake.params["core_quantile"]
    )
    result_row["core_quantile"] = snakemake.params["core_quantile"]

    result_row["n_tracks"] = len(summary_gsf)

    summary_gsf_no_outliers = summary_gsf[summary_gsf["nOutliers"] == 0]
    result_row["n_outliers"] = len(summary_gsf) - len(summary_gsf_no_outliers)

    local_coors = ["LOC0", "LOC1", "PHI", "THETA", "QOP", "P", "PNORM"]
    local_coors = ["QOP", "P", "PNORM"]

    if snakemake.params["filter_outliers"]:
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
                    # fmt: off
                    print(f"WARNING   unreasonable high/low values encountered for {key} / {stat.__name__}")
                    print(f"WARNING   {df[~sanitize_mask][key].to_numpy()}")
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
    result_df = pd.concat([result_df, iteration_df], ignore_index=True)

result_df.to_csv(snakemake.output[0], index=False)
