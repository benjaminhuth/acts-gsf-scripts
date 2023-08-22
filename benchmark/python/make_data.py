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
parser.add_argument('-n','--events', help='number of events', type=int, default=1)
parser.add_argument('-j','--jobs', help='number of jobs', type=int, default=-1)
parser.add_argument('-o','--output', help='output dir', type=str, default="")
parser.add_argument('--pmin', help='minimum momentum for particle gun', type=float, default=1.0)
parser.add_argument('--pmax', help='maximum momentum for particle gun', type=float, default=20.0)
parser.add_argument('-v', '--verbose', help='set loglevel to VERBOSE (except for sequencer)', default=False, action="store_true")
parser.add_argument('--multiplicity', type=int, default=1000)
args = vars(parser.parse_args())

defaultLogLevel = acts.logging.VERBOSE if args["verbose"] else acts.logging.INFO

print("Options:")
pprint.pprint(args)

#######
# ODD #
#######

acts_root = Path(os.environ["ACTS_ROOT"])

oddDir = acts_root / "thirdparty/OpenDataDetector"

oddMaterialMap = oddDir / "data/odd-material-maps.root"
digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(
    oddDir, mdecorator=oddMaterialDeco
)

outputDir = pathlib.Path(args["output"])
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
    logLevel=defaultLogLevel,
    enableInteractions=True,
    outputDirCsv=outputDir,
    postSelectParticles=ParticleSelectorConfig(
        removeSecondaries=True,
    ),
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    rnd=rnd,
    outputDirCsv=outputDir,
)

s.run()

