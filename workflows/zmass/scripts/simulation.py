import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import textwrap
import math
from pprint import pprint
import argparse

import acts
import acts.examples

from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.odd import getOpenDataDetector

import numpy as np


oddDir = Path(os.environ["ODD_DIR"])

oddMaterialMap = oddDir / "data/odd-material-maps.root"
digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
seedingSel = oddDir / "config/odd-seeding-config.json"

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(
    oddDir, mdecorator=oddMaterialDeco
)

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

outputDir = Path(snakemake.output[0]).parent
outputDir.mkdir(exist_ok=True, parents=True)
# (outputDir / "csv").mkdir(exist_ok=True, parents=True)

s = acts.examples.Sequencer(
    events=snakemake.config["events"],
    numThreads=1,
    outputDir=str(outputDir),
    trackFpes=False,
)

# From https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/800xxx/800279/mc.Py8EG_AZ_Zee.py
pythiaStrings = [
    "WeakSingleBoson:ffbar2gmZ = on",
    "23:onMode = off",
    "23:onIfAny = 11 -11",
    "PhaseSpace:mHatMin = 60.",
    "BeamRemnants:primordialKThard = 1.713",
    "SpaceShower:pT0Ref = 0.586",
    "SpaceShower:alphaSvalue = 0.12374",
    "MultipartonInteractions:pT0Ref = 2.18",
    "TimeShower:QEDshowerByL=off",
    "TimeShower:QEDshowerByOther=off",
]

addPythia8(
    s,
    hardProcess=pythiaStrings,
    npileup=0,
    vtxGen=acts.examples.GaussianVertexGenerator(
        stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
        mean=acts.Vector4(0, 0, 0, 0),
    ),
    rnd=rnd,
    # outputDirCsv=str(outputDir / "csv"),
    printPythiaEventListing="short" if snakemake.config["events"] == 1 else None,
    # printParticles=True
)

addGeant4(
    s,
    detector,
    trackingGeometry,
    field,
    preSelectParticles=ParticleSelectorConfig(
        absZ=(0, 1e4),
        rho=(0, 1e3),
        m=(0.510*u.MeV, 0.512*u.MeV),
        pt=(1 * u.GeV, None),
        removeNeutral=True,
    ),
    postSelectParticles=ParticleSelectorConfig(
        #absZ=(0, 25*u.mm),
        # rho=(0, 2*u.mm),
        # m=(0.510*u.MeV, 0.512*u.MeV),
        # pt=(1 * u.GeV, None),
        removeNeutral=True,
    ),
    # outputDirRoot=str(outputDir),
    rnd=rnd,
    killVolume=acts.Volume.makeCylinderVolume(r=1.1 * u.m, halfZ=3.0 * u.m),
    killAfterTime=25 * u.ns,
)

# s.addAlgorithm(
#     acts.examples.ParticlesPrinter(
#         level=acts.logging.INFO,
#         inputParticles="particles_initial_selected",
#     )
# )

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    # outputDirRoot=outputDir,
    # outputDirCsv=outputDir,
    rnd=rnd,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    seedingAlgorithm=SeedingAlgorithm.TruthEstimated,
    truthEstimatedSeedingAlgorithmConfigArg=TruthEstimatedSeedingAlgorithmConfigArg(
        deltaR=(0.0, np.inf)
    ),
    truthSeedRanges=TruthSeedRanges(rho=(0.0, 1.0), nHits=(3, None)),
    initialVarInflation=6 * [100],
    geoSelectionConfigFile=seedingSel,
    inputParticles="particles_selected",
    logLevel=acts.logging.INFO,
)

s.addAlgorithm(
    acts.examples.TruthTrackFinder(
        level=acts.logging.INFO,
        inputParticles="truth_seeded_particles",
        inputMeasurementParticlesMap="measurement_particles_map",
        outputProtoTracks="prototracks",
    )
)

# KF
kalmanOptions = {
    "multipleScattering": True,
    "energyLoss": True,
    "reverseFilteringMomThreshold": 0.0,
    "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
    "level": acts.logging.INFO,
}
kalmanFitter = acts.examples.makeKalmanFitterFunction(
    trackingGeometry, field, **kalmanOptions
)

# GSF
low_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_LT01_cdf_nC6_O5.par"
high_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_GT01_cdf_nC6_O5.par"

gsfOptions = {
    "maxComponents": 12,
    "abortOnError": False,
    "disableAllMaterialHandling": False,
    "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
        low_bhapprox,
        high_bhapprox,
    ),
    "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
    "weightCutoff": 1.0e-4,
    "level": acts.logging.FATAL,
}

gsfFitter = acts.examples.makeGsfFitterFunction(
    trackingGeometry, field, **gsfOptions
)

for fitter, function in zip(["gsf", "kf"], [gsfFitter, kalmanFitter]):
    trajectories = "trajectories_" + fitter
    tracks = "tracks_" + fitter

    s.addAlgorithm(
        acts.examples.TrackFittingAlgorithm(
            level=acts.logging.INFO,
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputProtoTracks="prototracks",
            inputInitialTrackParameters="estimatedparameters",
            outputTracks=tracks,
            calibrator=acts.examples.makePassThroughCalibrator(),
            fit=function,
        )
    )

    s.addAlgorithm(
        acts.examples.TracksToTrajectories(
            level=acts.logging.WARNING,
            inputTracks=tracks,
            outputTrajectories=trajectories,
        )
    )

    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=acts.logging.ERROR,
            inputTrajectories=trajectories,
            inputParticles="truth_seeded_particles",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "tracksummary_{}.root".format(fitter)),
        )
    )

    s.addWriter(
        acts.examples.RootTrajectoryStatesWriter(
            level=acts.logging.ERROR,
            inputTrajectories=trajectories,
            inputParticles="truth_seeded_particles",
            inputSimHits="simhits",
            inputMeasurementParticlesMap="measurement_particles_map",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(outputDir / "trackstates_{}.root".format(fitter)),
        )
    )

s.run()
