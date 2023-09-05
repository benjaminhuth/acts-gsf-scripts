from pathlib import Path

import acts
import acts.examples

from utils import run_fitting

kalmanOptions = {
    "multipleScattering": True,
    "energyLoss": True,
    "reverseFilteringMomThreshold": 0.0,
    "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
    "level": acts.logging.INFO,
}

run_fitting("kf", acts.examples.makeKalmanFitterFunction, kalmanOptions, snakemake)

