from pathlib import Path
import math
import random
import string
import multiprocessing
import subprocess
from functools import partial
import tempfile
import uproot
import os
import numpy as np


def run_geant4_sim(args, config):
    outputDir, events, skip = args

    import acts
    import acts.examples

    from acts.examples.simulation import (
        MomentumConfig,
        EtaConfig,
        PhiConfig,
        ParticleConfig,
        addParticleGun,
        addGeant4,
        ParticleSelectorConfig,
    )
    from acts.examples.odd import getOpenDataDetector

    from utils import setup

    u = acts.UnitConstants

    defaultLogLevel = acts.logging.ERROR

    detector, trackingGeometry, field, rnd = setup()
    outputDir.mkdir(exist_ok=True, parents=True)
    # (outputDir / "csv").mkdir(exist_ok=True, parents=True)

    s = acts.examples.Sequencer(
        events=events,
        skip=skip,
        numThreads=1,
        outputDir=str(outputDir),
        trackFpes=False,
        logLevel=acts.logging.INFO,
    )

    realistic_stddev = acts.Vector4(
        0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns
    )
    momCfg = MomentumConfig(
        config["p_min"] * u.GeV,
        config["p_max"] * u.GeV,
        transverse=snakemake.params["p_transverse"],
    )
    etaCfg = EtaConfig(-3, 3, uniform=snakemake.params["uniform_eta"])
    phiCfg = PhiConfig(0, 2 * math.pi)
    particleCfg = ParticleConfig(
        config["particles_per_vertex"],
        acts.PdgParticle(snakemake.params["pdg"]),
        randomizeCharge=False,
    )

    addParticleGun(
        s,
        rnd=rnd,
        momentumConfig=momCfg,
        etaConfig=etaCfg,
        phiConfig=phiCfg,
        particleConfig=particleCfg,
        multiplicity=config["multiplicity"],
        vtxGen=acts.examples.GaussianVertexGenerator(
            mean=acts.Vector4(0, 0, 0, 0),
            stddev=realistic_stddev,
        ),
        logLevel=defaultLogLevel,
        outputDirRoot=outputDir,
    )

    addGeant4(
        s,
        detector,
        trackingGeometry,
        field,
        postSelectParticles=ParticleSelectorConfig(
            absZ=(0, 200 * u.mm),
            rho=(0, 2 * u.mm),
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


n_events = snakemake.params["n_events"]
jobs = min(snakemake.config["sim_jobs"], n_events)
output_dir = Path(snakemake.output[0]).parent
output_dir.mkdir(exist_ok=True, parents=True)
chunks = np.array_split(np.arange(n_events), jobs)

skips = [c[0] for c in chunks]
events = [len(c) for c in chunks]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    outdirs = [tmp / f"subdir{i}" for i in range(jobs)]

    with multiprocessing.Pool(jobs) as p:
        p.map(
            partial(run_geant4_sim, config={**snakemake.config, **snakemake.params}),
            zip(outdirs, events, skips),
            1,
        )

    for filename in [Path(f).name for f in snakemake.output]:
        files = [d / filename for d in outdirs]
        subprocess.run(["hadd", output_dir / filename] + files)

# test consistency
unique_event_ids = np.unique(
    uproot.open(str(output_dir / "hits.root:hits")).arrays(library="pd").event_id
)
assert len(unique_event_ids) == n_events
