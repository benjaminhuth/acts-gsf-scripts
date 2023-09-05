from pathlib import Path
import os

import numpy as np

import acts
import acts.examples

from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.odd import getOpenDataDetector

oddDir = Path(os.environ["ODD_DIR"])
assert oddDir.exists()

defaultLogLevel = acts.logging.ERROR

def setup():
    oddMaterialMap = oddDir / "data/odd-material-maps.root"

    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    detector, trackingGeometry, decorators = getOpenDataDetector(
        oddDir, mdecorator=oddMaterialDeco, logLevel=defaultLogLevel
    )

    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))
    rnd = acts.examples.RandomNumbers(seed=42)

    return detector, trackingGeometry, field, rnd


def run_fitting(fitter, fitter_factory, fitter_options, snakemake):
    outputDir = Path(snakemake.output[0]).parent
    _, trackingGeometry, field, rnd = setup()

    s = acts.examples.Sequencer(
        events=snakemake.config["n_events"],
        numThreads=-1,
        outputDir=outputDir,
        trackFpes=False,
        logLevel=defaultLogLevel,
    )

    s.addReader(
        acts.examples.RootParticleReader(
            level=defaultLogLevel,
            particleCollection="particles_selected",
            filePath=snakemake.input[0],
        )
    )

    s.addReader(
        acts.examples.RootSimHitReader(
            level=defaultLogLevel,
            filePath=snakemake.input[1],
            simHitCollection="simhits",
        )
    )

    digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"

    addDigitization(
        s=s,
        trackingGeometry=trackingGeometry,
        field=field,
        rnd=rnd,
        logLevel=defaultLogLevel,
        digiConfigFile=digiConfigFile,
    )

    seedingSel = oddDir / "config/odd-seeding-config.json"

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
        logLevel=defaultLogLevel,
    )

    s.addAlgorithm(
        acts.examples.TruthTrackFinder(
            level=defaultLogLevel,
            inputParticles="truth_seeded_particles",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputProtoTracks="prototracks",
        )
    )

    trajectories = "trajectories_" + fitter
    tracks = "tracks_" + fitter

    function = fitter_factory(trackingGeometry, field, **fitter_options)

    s.addAlgorithm(
        acts.examples.TrackFittingAlgorithm(
            level=defaultLogLevel,
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
            level=defaultLogLevel,
            inputTracks=tracks,
            outputTrajectories=trajectories,
        )
    )

    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=defaultLogLevel,
            inputTrajectories=trajectories,
            inputParticles="truth_seeded_particles",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "tracksummary_{}.root".format(fitter)),
        )
    )

    s.addWriter(
        acts.examples.RootTrajectoryStatesWriter(
            level=defaultLogLevel,
            inputTrajectories=trajectories,
            inputParticles="truth_seeded_particles",
            inputSimHits="simhits",
            inputMeasurementParticlesMap="measurement_particles_map",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(outputDir / "trackstates_{}.root".format(fitter)),
        )
    )

    s.run()
