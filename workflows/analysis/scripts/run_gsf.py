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

if "use_non_geant_bha" in vars(snakemake.params) and snakemake.params["use_non_geant_bha"]:
    print("INFO    Use 'BetheHeitler_cdf_nC6_O5.par' instead of Geant4 based approximation")
    bha = acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
        lowParametersPath="config/BetheHeitler_cdf_nC6_O5.par",
        highParametersPath="config/BetheHeitler_cdf_nC6_O5.par",
        lowLimit=0.1,
        highLimit=0.2,
    )

gsfOptions = {
    "maxComponents": snakemake.params["components"],
    "betheHeitlerApprox": bha,
    "componentMergeMethod": acts.examples.ComponentMergeMethod.maxWeight,
    "weightCutoff": snakemake.params["weight_cutoff"],
    # "momentumCutoff": 0.1,
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
