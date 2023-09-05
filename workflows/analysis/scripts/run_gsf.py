from pathlib import Path

import acts
import acts.examples

from utils import run_fitting

athena_dir = Path("/home/benjamin/Documents/athena")
gsf_data_dir = athena_dir / "Tracking/TrkFitter/TrkGaussianSumFilter/Data"
low_bhapprox = gsf_data_dir / "GeantSim_LT01_cdf_nC6_O5.par"
high_bhapprox = gsf_data_dir / "GeantSim_GT01_cdf_nC6_O5.par"

gsfOptions = {
    "maxComponents": snakemake.params["components"],
    "abortOnError": False,
    "disableAllMaterialHandling": False,
    "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
        str(low_bhapprox),
        str(high_bhapprox),
    ),
    "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
    "weightCutoff": snakemake.params["weight_cutoff"],
    "level": acts.logging.ERROR,
}

run_fitting("gsf", acts.examples.makeGsfFitterFunction, gsfOptions, snakemake)

