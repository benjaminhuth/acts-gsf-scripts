#!/usr/bin/env python3
import os
from pathlib import Path
import pathlib
import argparse
import pprint

import acts
from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.odd import getOpenDataDetector

u = acts.UnitConstants

parser = argparse.ArgumentParser(description='Run GSF OpenDataDetector')
parser.add_argument('-n','--events', type=int, default=1)
parser.add_argument('-j','--jobs', help='number of jobs', type=int, default=1)
parser.add_argument('-i','--inputdir', type=str, default="")
parser.add_argument('-c','--components', help='number of GSF components', type=int, default=12)
args = vars(parser.parse_args())

print("Options:")
pprint.pprint(args)

inputDir = Path(args["inputdir"])
assert inputDir.exists()

#######
# ODD #
#######

acts_root = Path(os.environ["ACTS_ROOT"])

oddDir = acts_root / "thirdparty/OpenDataDetector"
# acts.examples.dump_args_calls(locals())  # show python binding calls
oddMaterialMap = oddDir / "data/odd-material-maps.root"
digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(
    oddDir, mdecorator=oddMaterialDeco
)

outputDir = pathlib.Path.cwd()
outputDir.mkdir(exist_ok=True, parents=True)

rnd = acts.examples.RandomNumbers(seed=42)
field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
defaultLogLevel=acts.logging.INFO

#############
# Sequencer #
#############

s = acts.examples.Sequencer(
    events=args["events"], numThreads=args["jobs"], logLevel=defaultLogLevel, trackFpes=False,
)



for d in decorators:
    s.addContextDecorator(d)

s.addReader(
    acts.examples.CsvParticleReader(
        level=defaultLogLevel,
        inputDir=str(inputDir),
        inputStem="particles_initial",
        outputParticles="particles",
    )
)

s.addReader(
    acts.examples.CsvSimHitReader(
        level=defaultLogLevel,
        inputDir=str(inputDir),
        inputStem="hits",
        outputSimHits="simhits",
    )
)

# s.addReader(
#     acts.examples.CsvMeasurementReader(
#         level=defaultLogLevel,
#         inputDir = str(inputDir),
#         outputMeasurements = "measurements",
#         outputMeasurementSimHitsMap = "measurement_simhit_map",
#         outputSourceLinks = "sourcelinks",
#         inputSimHits = "simhits",
#         outputMeasurementParticlesMap = "measurement_particles_map",
#     )
# )
digiFile = (Path(__file__).parent.parent.parent / "config/odd/odd-digi-smearing-config.json").resolve()
assert digiFile.exists()
addDigitization(
    s=s,
    trackingGeometry=trackingGeometry,
    field=field,
    rnd=rnd,
    # outputDirRoot=outputDir,
    # outputDirCsv=(outputDir / "csv")
    logLevel=acts.logging.INFO,
    digiConfigFile=digiFile,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    # inputParticles="particles_initial",
    # truthEstimatedSeedingAlgorithmConfigArg=TruthEstimatedSeedingAlgorithmConfigArg(
    #     deltaR=(0.0, 10000.0)
    # ),
    # truthSeedRanges=TruthSeedRanges(rho=(0.0, 1.0), nHits=(3, None)),
    # geoSelectionConfigFile=oddDir / "config/odd-seeding-config.json",
    # initialVarInflation=6 * [100],
)

s.addAlgorithm(
    acts.examples.TruthTrackFinder(
        level=defaultLogLevel,
        inputParticles="truth_seeds_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
        outputProtoTracks="prototracks",
    )
)

gsfOptions = {
    "maxComponents": args["components"],
    "abortOnError": False,
    "disableAllMaterialHandling": False,
    "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
        "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_LT01_cdf_nC6_O5.par",
        "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_GT01_cdf_nC6_O5.par"),
    "componentMergeMethod": acts.examples.ComponentMergeMethod.maxWeight,
    "weightCutoff": 1.e-8,
    "level": acts.logging.ERROR,
}
pprint.pprint(gsfOptions)

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=acts.logging.INFO,
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="prototracks",
        inputInitialTrackParameters="estimatedparameters",
        outputTracks="gsf_trajectories",
        fit=acts.examples.makeGsfFitterFunction(
            trackingGeometry, field, **gsfOptions
        ),
        calibrator=acts.examples.makePassThroughCalibrator(),
    )
)

s.run()
