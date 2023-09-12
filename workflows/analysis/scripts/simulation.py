from pathlib import Path
import math

import acts
import acts.examples

from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.odd import getOpenDataDetector

import numpy as np

from utils import setup

u = acts.UnitConstants

defaultLogLevel = acts.logging.ERROR

detector, trackingGeometry, field, rnd = setup()

outputDir = Path(snakemake.output[0]).parent
outputDir.mkdir(exist_ok=True, parents=True)
# (outputDir / "csv").mkdir(exist_ok=True, parents=True)

s = acts.examples.Sequencer(
    events=snakemake.config["n_events"],
    numThreads=1,
    outputDir=str(outputDir),
    trackFpes=False,
    logLevel=acts.logging.INFO,
)


realistic_stddev = acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns)
momCfg  = MomentumConfig(snakemake.config["p_min"] * u.GeV, snakemake.config["p_max"] * u.GeV, transverse=True),
etaCfg = EtaConfig(-3, 3),
phiCfg = PhiConfig(0, 2 * math.pi),

addParticleGun(
    s,
    ParticleConfig(snakemake.config["particles_per_vertex"], acts.PdgParticle.eElectron, randomizeCharge=False),
    rnd=rnd,
    vtxGen=acts.examples.GaussianVertexGenerator(
        mean=acts.Vector4(0, 0, 0, 0),
        stddev=realistic_stddev,
    ),
    multiplicity=snakemake.config["multiplicity"],
    logLevel=defaultLogLevel,
)

addGeant4(
    s,
    detector,
    trackingGeometry,
    field,
    postSelectParticles=ParticleSelectorConfig(
        absZ=(0, 25*u.mm),
        rho=(0, 2*u.mm),
        removeNeutral=True,
        removeSecondaries=True,
    ),
    outputDirRoot=outputDir,
    rnd=rnd,
    killVolume=acts.Volume.makeCylinderVolume(r=1.1 * u.m, halfZ=3.0 * u.m),
    killAfterTime=25 * u.ns,
    logLevel=defaultLogLevel,
)

s.run()
