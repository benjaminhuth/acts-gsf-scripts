from pathlib import Path
import math
import pprint
import os

import argparse

import acts
import acts.examples
from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.geant4 import GdmlDetectorConstruction

import numpy as np

from gdml_telescope import *

u = acts.UnitConstants

#####################
# Cmd line handling #
#####################

parser = argparse.ArgumentParser(description='Run GSF sPHENIX')
parser.add_argument('-p','--pick', help='pick track', type=int, default=-1)
parser.add_argument('-n','--events', help='number of events', type=int, default=10)
parser.add_argument('-j','--jobs', help='number of jobs', type=int, default=-1)
parser.add_argument('-s','--skip', help='number of skipped events', type=int, default=0)
parser.add_argument('-c','--components', help='number of GSF components', type=int, default=4)
parser.add_argument('--surfaces', help='number of telescope surfaces', type=int, default=5)
parser.add_argument('--pmin', help='minimum momentum for particle gun', type=float, default=1.0)
parser.add_argument('--pmax', help='maximum momentum for particle gun', type=float, default=20.0)
args = vars(parser.parse_args())

##################
# Build geometry #
##################

surface_distance = 50
surface_thickness = 1
surface_width = 1000

# to match geant4 geometry
x_offset = args["surfaces"] * surface_distance / 2

telescopeConfig = {
    "positions": np.arange(-x_offset, args["surfaces"]*surface_distance, surface_distance).tolist(),
    # "offsets": foo,
    "bounds": [surface_width, surface_width], #[args["surfaces"]*10, args["surfaces"]*10],
    "thickness": 1,
    "surfaceType": 0,
    "binValue": 0,
}

# pprint.pprint(telescopeConfig)


detector, trackingGeometry, decorators = acts.examples.TelescopeDetector.create(
    **telescopeConfig
)

# Geant 4
gdml_file = "gdml/telescope.gdml"
make_gdml(gdml_file, args["surfaces"], surface_width, surface_thickness, surface_distance)
g4detector = GdmlDetectorConstruction(gdml_file)

###################
# Setup sequencer #
###################

outputDir = Path(".")
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)
use_geant = True

if use_geant:
    args["jobs"] = 1

s = acts.examples.Sequencer(
    events=args["events"],
    numThreads=args["jobs"],
    outputDir=str(outputDir),
    skip=args["skip"]
)

(outputDir / "csv").mkdir(exist_ok=True, parents=True)
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
    MomentumConfig(args['pmin'] * u.GeV, args['pmax'] * u.GeV, transverse=True),
    EtaConfig(-0.01, 0.01, uniform=True),
    PhiConfig(-0.01, 0.01),
    ParticleConfig(1, acts.PdgParticle.eElectron, randomizeCharge=False),
    rnd=rnd,
    vtxGen=acts.examples.GaussianVertexGenerator(
        stddev=acts.Vector4(0, 0, 0, 0), mean=acts.Vector4(0, 0, 0, 0)
    ),
    multiplicity=1000,
)

if use_geant:
    addGeant4(
        s,
        g4detector,
        trackingGeometry,
        field,
        rnd,
        outputDirCsv="csv",
        materialMappings=["G4_Si"],
    )
else:
    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        outputDirCsv="csv"
    )

'''
addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile="telescope-digi-smearing-config.json",
    # outputDirRoot=outputDir,
    rnd=rnd,
    outputDirCsv=None, #"csv"
)

addSeeding(
    s,
    trackingGeometry,
    field,
    seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
)

s.addAlgorithm(
    acts.examples.TruthTrackFinder(
        level=acts.logging.INFO,
        inputParticles="truth_seeds_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
        outputProtoTracks="prototracks",
    )
)

kalmanOptions = {
    "multipleScattering": True,
    "energyLoss": True,
    "reverseFilteringMomThreshold": 0.0,
    "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
}

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=acts.logging.VERBOSE if args["pick"] != -1 else acts.logging.INFO,
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="prototracks",
        inputInitialTrackParameters="estimatedparameters",
        outputTrajectories="trajectories",
        directNavigation=False,
        pickTrack=args["pick"],
        trackingGeometry=trackingGeometry,
        dFit=acts.examples.TrackFittingAlgorithm.makeKalmanFitterFunction(
            field, **kalmanOptions
        ),
        fit=acts.examples.TrackFittingAlgorithm.makeKalmanFitterFunction(
            trackingGeometry, field, **kalmanOptions
        ),
    )
)

gsfOptions = {
    "maxComponents": args["components"],
    "abortOnError": False,
    "disableAllMaterialHandling": False,
#    "finalReductionMethod": acts.examples.FinalReductionMethod.mean,
}
pprint.pprint(gsfOptions)

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=acts.logging.VERBOSE if args["pick"] != -1 else acts.logging.INFO,
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="prototracks",
        inputInitialTrackParameters="estimatedparameters",
        outputTrajectories="gsf_trajectories",
        directNavigation=False,
        pickTrack=args["pick"],
        trackingGeometry=trackingGeometry,
        fit=acts.examples.TrackFittingAlgorithm.makeGsfFitterFunction(
            trackingGeometry, field, **gsfOptions
        ),
    )
)

s.addWriter(
    acts.examples.RootTrajectorySummaryWriter(
        level=acts.logging.INFO,
        inputTrajectories="gsf_trajectories",
        inputParticles="truth_seeds_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
        filePath=str(outputDir / "root/tracksummary_gsf.root"),
    )
)

s.addWriter(
    acts.examples.RootTrajectorySummaryWriter(
        level=acts.logging.INFO,
        inputTrajectories="trajectories",
        inputParticles="truth_seeds_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
        filePath=str(outputDir / "root/tracksummary_kf.root"),
    )
)
'''
s.run()
