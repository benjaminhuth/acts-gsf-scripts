#!/usr/bin/env python3
import os
from pathlib import Path
import pathlib
import pprint

import acts
from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.odd import getOpenDataDetector

u = acts.UnitConstants

defaultLogLevel = acts.logging.VERBOSE if snakemake.config["verbose"] else acts.logging.ERROR

outputDir = Path(snakemake.output[0]).parent

#######
# ODD #
#######

oddDir = Path(os.environ["ODD_DIR"])

oddMaterialMap = oddDir / "data/odd-material-maps.root"
digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(
    oddDir, mdecorator=oddMaterialDeco
)

##################
# Rnd and Bfield #
##################

rnd = acts.examples.RandomNumbers(seed=42)
field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

#############
# Sequencer #
#############

s = acts.examples.Sequencer(
    events=snakemake.config["n_events"], numThreads=1, logLevel=defaultLogLevel, trackFpes=False,
)

for d in decorators:
    s.addContextDecorator(d)

addParticleGun(
    s,
    EtaConfig(-3.0, 3.0),
    ParticleConfig(snakemake.config["particles_per_vertex"], acts.PdgParticle.eElectron, False),
    PhiConfig(0.0, 360.0 * u.degree),
    MomentumConfig(snakemake.config["p_min"] * u.GeV, snakemake.config["p_max"] * u.GeV),
    multiplicity=snakemake.config["multiplicity"],
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

s.run()

