from pathlib import Path
import math
import pprint
import os

import argparse

import acts
import acts.examples
from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.geant4 import GdmlDetectorConstructionFactory

import numpy as np
u = acts.UnitConstants

#####################
# Cmd line handling #
#####################

parser = argparse.ArgumentParser(description='Run GSF sPHENIX')
parser.add_argument('-p','--pick', help='pick track', type=int, default=-1)
parser.add_argument('-n','--events', help='number of events', type=int, default=1)
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

positions = np.arange(-x_offset, args["surfaces"]*surface_distance, surface_distance).tolist()
telescopeConfig = {
    "positions": positions,
    # "offsets": foo,
    "bounds": [surface_width, surface_width], #[args["surfaces"]*10, args["surfaces"]*10],
    "thickness": 1,
    "surfaceType": 0,
    "binValue": 0,
    "stereos": len(positions)*[0.0],
}

# pprint.pprint(telescopeConfig)


detector, trackingGeometry, decorators = acts.examples.TelescopeDetector.create(
    **telescopeConfig
)

# Geant 4
gdml_file = "gdml/telescope.gdml"
g4factory = GdmlDetectorConstructionFactory(gdml_file)

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
    multiplicity=1,
)

addGeant4(
    s,
    detector=None,
    g4DetectorConstructionFactory=g4factory,
    trackingGeometry=trackingGeometry,
    field=field,
    rnd=rnd,
    outputDirCsv="csv",
    materialMappings=["G4_Si"],
    logLevel=acts.logging.VERBOSE,
)

s.run()
