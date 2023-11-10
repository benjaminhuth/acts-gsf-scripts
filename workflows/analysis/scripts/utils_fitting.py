from pathlib import Path
import os

import numpy as np

import acts
import acts.examples

from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.odd import getOpenDataDetector

u = acts.UnitConstants

oddDir = Path(os.environ["ODD_DIR"])
assert oddDir.exists()

defaultLogLevel = acts.logging.ERROR

seedingSel = oddDir / "config/odd-seeding-config.json"
assert seedingSel.exists()

digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
assert digiConfigFile.exists()

oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert oddMaterialMap.exists()


def setup():
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    detector, trackingGeometry, decorators = getOpenDataDetector(
        oddDir, mdecorator=oddMaterialDeco, logLevel=defaultLogLevel
    )

    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))
    rnd = acts.examples.RandomNumbers(seed=42)

    return detector, trackingGeometry, field, rnd


def run_fitting(
    fitter,
    fitter_factory,
    fitter_options,
    outputDir,
    inputParticles,
    inputHits,
    n_events=None,
    seeding="truth_estimated",
):
    assert seeding in ["truth_estimated", "smeared"]
    detector, trackingGeometry, field, rnd = setup()

    s = acts.examples.Sequencer(
        events=n_events,
        numThreads=-1,
        outputDir=outputDir,
        trackFpes=False,
        logLevel=defaultLogLevel,
    )

    s.addReader(
        acts.examples.RootParticleReader(
            level=defaultLogLevel,
            particleCollection="particles_initial",
            filePath=inputParticles,
        )
    )

    s.addReader(
        acts.examples.RootSimHitReader(
            level=defaultLogLevel,
            filePath=inputHits,
            simHitCollection="simhits",
        )
    )

    addDigitization(
        s=s,
        trackingGeometry=trackingGeometry,
        field=field,
        rnd=rnd,
        logLevel=defaultLogLevel,
        digiConfigFile=digiConfigFile,
        # outputDirCsv=outputDir / "csv",
    )

    addSeedingTruthSelection(
        s,
        "particles_initial",
        "particles_initial_selected",
        truthSeedRanges=TruthSeedRanges(rho=(0, 2 * u.mm), nHits=(3, None)),
        logLevel=defaultLogLevel,
    )

    if seeding == "truth_estimated":
        spacePoints = addSpacePointsMaking(
            s,
            trackingGeometry,
            seedingSel,
            logLevel=defaultLogLevel,
        )

        s.addAlgorithm(
            acts.examples.TruthSeedingAlgorithm(
                level=defaultLogLevel,
                inputParticles="particles_initial_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                inputSpacePoints=[spacePoints],
                outputParticles="truth_seeded_particles",
                outputProtoTracks="truth_seeded_prototracks",
                outputSeeds="seeds",
            )
        )

        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=defaultLogLevel,
                inputParticles="truth_seeded_particles",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="prototracks",
            )
        )

        s.addAlgorithm(
            acts.examples.TrackParamsEstimationAlgorithm(
                level=defaultLogLevel,
                inputSeeds="seeds",
                inputProtoTracks="prototracks",
                outputTrackParameters="track_parameters",
                outputProtoTracks="prototracks_with_params",
                trackingGeometry=trackingGeometry,
                magneticField=field,
                initialVarInflation=[100.0] * 6,
                particleHypothesis=acts.ParticleHypothesis.electron,
            )
        )
    elif seeding == "smeared":
        s.addAlgorithm(
            acts.examples.ParticleSmearing(
                level=defaultLogLevel,
                randomNumbers=rnd,
                inputParticles="particles_initial_selected",
                outputTrackParameters="track_parameters",
                initialVarInflation=[100.0] * 6,
                particleHypothesis=acts.ParticleHypothesis.electron,
            )
        )

        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=defaultLogLevel,
                inputParticles="particles_initial_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="prototracks_with_params",
            )
        )

        s.addWhiteboardAlias("truth_seeded_particles", "particles_initial_selected")
    else:
        raise ValueError(f"invalid seeding config '{seeding}'")
    tracks = "tracks_" + fitter

    function = fitter_factory(trackingGeometry, field, **fitter_options)

    s.addAlgorithm(
        acts.examples.TrackFittingAlgorithm(
            level=defaultLogLevel,
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputProtoTracks="prototracks_with_params",
            inputInitialTrackParameters="track_parameters",
            outputTracks=tracks,
            calibrator=acts.examples.makePassThroughCalibrator(),
            fit=function,
        )
    )

    s.addWriter(
        acts.examples.RootTrackSummaryWriter(
            level=defaultLogLevel,
            inputTracks=tracks,
            inputParticles="truth_seeded_particles",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "tracksummary_{}.root".format(fitter)),
        )
    )

    s.addWriter(
        acts.examples.RootTrackStatesWriter(
            level=defaultLogLevel,
            inputTracks=tracks,
            inputParticles="truth_seeded_particles",
            inputSimHits="simhits",
            inputMeasurementParticlesMap="measurement_particles_map",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(outputDir / "trackstates_{}.root".format(fitter)),
        )
    )

    s.run()


