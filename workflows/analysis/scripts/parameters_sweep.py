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

bethe_heitler_approxes = {
    "GeantSim_CDF": ("GeantSim_LT01_cdf_nC6_O5.par", "GeantSim_GT01_cdf_nC6_O5.par", 0.1, 0.3),
    "CDF_C6_O5": ("BetheHeitler_cdf_nC6_O5.par", "BetheHeitler_cdf_nC6_O5.par", 0.1, 0.2),
    "CDFMOM_C6_O5": ("BetheHeitler_cdfmom_nC6_O5.par", "BetheHeitler_cdfmom_nC6_O5.par", 0.1, 0.2),
    "KL_C6_O5": ("BetheHeitler_kl_nC6_O5.par", "BetheHeitler_kl_nC6_O5.par", 0.1, 0.2),
}


def run_configuration(pars):
    components, weight_cutoff, componentMergeMethod, reductionAlgorithm, bha = pars
    print(f"-> cmps: {components}, wc: {weight_cutoff}, mm: {componentMergeMethod}, ra: {reductionAlgorithm}")

    configDir = Path("./config")
    lowPars, highPars, lowLimit, highLimit = bethe_heitler_approxes[bha]
    lowPars = configDir / lowPars
    highPars = configDir / highPars
    
    assert lowPars.exists()
    assert highPars.exists()

    opts = {
        "maxComponents": components,
        "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
            lowParametersPath=str(lowPars),
            highParametersPath=str(highPars),
            lowLimit=lowLimit,
            highLimit=highLimit,
        ),
        "componentMergeMethod": vars(acts.examples.ComponentMergeMethod)[componentMergeMethod],
        "weightCutoff": weight_cutoff,
        "level": acts.logging.ERROR,
        "mixtureReductionAlgorithm" : vars(acts.examples.MixtureReductionAlgorithm)[reductionAlgorithm],
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
    result_row["component_merge_method"] = str(opts["componentMergeMethod"])[21:]
    result_row["mixture_reduction"] = str(reductionAlgorithm)
    result_row["bethe_heitler_approx"] = bha

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


pars = []

components = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
print("Sweep components:", components)
for c in components:
    pars.append((c, 1.0e-4, "maxWeight", "KLDistance", "GeantSim_CDF"))

# 1.e-6 already covered above
weight_cutoff = [1.0e-8, 1.0e-6, 1.0e-2, 1.0e-1]
print("Sweep weight cutoffs:", weight_cutoff)
for w in weight_cutoff:
    pars.append((12, w, "maxWeight", "KLDistance", "GeantSim_CDF"))

pars.append((12, 1.0e-4, "mean", "KLDistance", "GeantSim_CDF"))
pars.append((12, 1.0e-4, "maxWeight", "weightCut", "GeantSim_CDF"))

for k in bethe_heitler_approxes.keys():
    if k != "GeantSim_CDF":
        pars.append((12, 1.0e-4, "maxWeight", "KLDistance", k))


print("Grid size:", len(pars))

with multiprocessing.Pool(snakemake.config["sim_jobs"]) as pool:
    result_dfs = pool.map(run_configuration, pars)

result_df = pd.concat(result_dfs, ignore_index=True)

assert len(result_df) > 0

result_df.to_csv(snakemake.output[0], index=False)
