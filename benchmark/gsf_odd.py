#!/usr/bin/env python3
import os
from pathlib import Path
import pathlib
import argparse
import pprint
import acts

u = acts.UnitConstants

from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.odd import getOpenDataDetector


parser = argparse.ArgumentParser(description='Run GSF OpenDataDetector')
parser.add_argument('-n','--events', help='number of events', type=int, default=1)
parser.add_argument('-j','--jobs', help='number of jobs', type=int, default=-1)
parser.add_argument('-c','--components', help='number of GSF components', type=int, default=12)
parser.add_argument('--pmin', help='minimum momentum for particle gun', type=float, default=1.0)
parser.add_argument('--pmax', help='maximum momentum for particle gun', type=float, default=20.0)
parser.add_argument('--erroronly', help='set loglevel to show only errors (except sequencer)', default=False, action="store_true")
parser.add_argument('-v', '--verbose', help='set loglevel to VERBOSE (except for sequencer)', default=False, action="store_true")
parser.add_argument('--debug', help='debug logging', default=False, action="store_true")
parser.add_argument('--multiplicity', type=int, default=1000)
args = vars(parser.parse_args())

defaultLogLevel = acts.logging.FATAL if args["erroronly"] else acts.logging.ERROR
defaultLogLevel = acts.logging.VERBOSE if args["verbose"] else defaultLogLevel
defaultLogLevel = acts.logging.DEBUG if args["debug"] else defaultLogLevel


print("Options:")
pprint.pprint(args)

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

##################
# Rnd and Bfield #
##################

rnd = acts.examples.RandomNumbers(seed=42)
field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))


#############
# Sequencer #
#############

s = acts.examples.Sequencer(
    events=args["events"], numThreads=args["jobs"], logLevel=acts.logging.INFO, trackFpes=False,
)


for d in decorators:
    s.addContextDecorator(d)

s.addWriter(
    acts.examples.CsvTrackingGeometryWriter(
        level=acts.logging.INFO,
        trackingGeometry=trackingGeometry,
        outputDir=str(outputDir / "csv"),
        writePerEvent=False,
    )
)

addParticleGun(
    s,
    EtaConfig(-3.0, 3.0),
    ParticleConfig(2, acts.PdgParticle.eElectron, False),
    PhiConfig(0.0, 360.0 * u.degree),
    MomentumConfig(args["pmin"] * u.GeV, args["pmax"] * u.GeV),
    multiplicity=args["multiplicity"],
    vtxGen=acts.examples.GaussianVertexGenerator(stddev=acts.Vector4(0, 0, 0, 0), mean=acts.Vector4(0, 0, 0, 0)),
    rnd=rnd,
    printParticles=False,
    logLevel=defaultLogLevel
)

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    outputDirCsv="csv",
    logLevel=defaultLogLevel,
    enableInteractions=True,
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    rnd=rnd,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    inputParticles="particles_initial",
    truthSeedRanges=TruthSeedRanges(nHits=(9,None)),
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
    "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
    "weightCutoff": 1.e-8,
    "level": defaultLogLevel,
}
pprint.pprint(gsfOptions)

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=defaultLogLevel,
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
