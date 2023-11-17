from pathlib import Path

import acts
import acts.examples

from utils_fitting import run_fitting


bha = acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
    lowParametersPath="config/GeantSim_LT01_cdf_nC6_O5.par",
    highParametersPath="config/GeantSim_GT01_cdf_nC6_O5.par",
    lowLimit=0.1,
    highLimit=0.3,
)

# bha = acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
#     lowParametersPath="config/BetheHeitler_cdf_nC6_O5.par",
#     highParametersPath="config/BetheHeitler_cdf_nC6_O5.par",
#     lowLimit=0.1,
#     highLimit=0.2,
# )

MergeMethodEnum = None
try:
    MergeMethodEnum = acts.examples.ComponentMergeMethod
    mergeMethodKey = "componentMergeMethod"
except:
    MergeMethodEnum = acts.examples.FinalReductionMethod
    mergeMethodKey = "finalReductionMethod"

gsfOptions = {
    "maxComponents": snakemake.params["components"],
    "betheHeitlerApprox": bha,
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
