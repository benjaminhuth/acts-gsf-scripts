from pathlib import Path

import acts
import acts.examples

from utils_fitting import run_fitting

kalmanOptions = {
    "multipleScattering": True,
    "energyLoss": True,
    "reverseFilteringMomThreshold": 0.0,
    "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
    "level": acts.logging.INFO,
}

outputDir = Path(snakemake.output[0]).parent
run_fitting(
    "kf",
    acts.examples.makeKalmanFitterFunction,
    kalmanOptions,
    outputDir,
    snakemake.input[0],
    snakemake.input[1],
    seeding=snakemake.params["seeding"],
)
