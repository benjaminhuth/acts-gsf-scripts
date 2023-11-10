from pathlib import Path

import acts
import acts.examples

from utils_fitting import run_fitting

athena_dir = Path("/home/benjamin/Documents/athena")
gsf_data_dir = athena_dir / "Tracking/TrkFitter/TrkGaussianSumFilter/Data"
low_bhapprox = gsf_data_dir / "GeantSim_LT01_cdf_nC6_O5.par"
high_bhapprox = gsf_data_dir / "GeantSim_GT01_cdf_nC6_O5.par"

MergeMethodEnum = None
try:
    MergeMethodEnum = acts.examples.ComponentMergeMethod
    mergeMethodKey = "componentMergeMethod"
except:
    MergeMethodEnum = acts.examples.FinalReductionMethod
    mergeMethodKey = "finalReductionMethod"

gsfOptions = {
    "maxComponents": snakemake.params["components"],
    "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
        str(low_bhapprox),
        str(high_bhapprox),
    ),
    mergeMethodKey: MergeMethodEnum.maxWeight,
    "weightCutoff": snakemake.params["weight_cutoff"],
    "level": acts.logging.ERROR,
    "mixtureReductionAlgorithm": acts.examples.MixtureReductionAlgorithm.KLDistance,
}

outputDir = Path(snakemake.output[0]).parent
run_fitting(
    "gsf",
    acts.examples.makeGsfFitterFunction,
    gsfOptions,
    outputDir,
    snakemake.input[0],
    snakemake.input[1],
    seeding=snakemake.params["seeding"],
)
