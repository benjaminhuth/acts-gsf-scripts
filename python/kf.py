from pathlib import Path
import os
import argparse

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector
from acts.examples.simulation import *
from acts.examples.reconstruction import *

u = acts.UnitConstants


def runTruthTrackingKalman(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    outputDir: Path,
    digiConfigFile: Path,
    directNavigation=False,
    reverseFilteringMomThreshold=0 * u.GeV,
    s: acts.examples.Sequencer = None,
):
    from acts.examples.simulation import (
        addParticleGun,
        EtaConfig,
        ParticleConfig,
        addFatras,
        addDigitization,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TruthSeedRanges,
        addKalmanTracks,
    )

    s = s or acts.examples.Sequencer(
        events=100, numThreads=-1, logLevel=acts.logging.INFO
    )

    rnd = acts.examples.RandomNumbers()
    outputDir = Path(outputDir)

    addParticleGun(
        s,
        EtaConfig(-3.0, 3.0),
        ParticleConfig(1, acts.PdgParticle.eMuon, False),
        multiplicity=1000,
        vtxGen=acts.examples.GaussianVertexGenerator(
            mean=acts.Vector4(0, 0, 0, 0),
            stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
        ),
        rnd=rnd,
        outputDirRoot=None,
    )

    addFatras(s, trackingGeometry, field, rnd=rnd, enableInteractions=True)

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
        rnd=rnd,
        truthSeedRanges=TruthSeedRanges(
            pt=(500 * u.MeV, None),
            nHits=(9, None),
        ),
    )

    addKalmanTracks(
        s,
        trackingGeometry,
        field,
        directNavigation,
        reverseFilteringMomThreshold,
        energyLoss=True,
        multipleScattering=True,
    )

    # Output
    # s.addWriter(
    #     acts.examples.RootTrajectoryStatesWriter(
    #         level=acts.logging.INFO,
    #         inputTrajectories="trajectories",
    #         inputParticles="truth_seeds_selected",
    #         inputSimHits="simhits",
    #         inputMeasurementParticlesMap="measurement_particles_map",
    #         inputMeasurementSimHitsMap="measurement_simhits_map",
    #         filePath=str(outputDir / "trackstates_fitter.root"),
    #     )
    # )

    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=acts.logging.INFO,
            inputTrajectories="trajectories",
            inputParticles="truth_seeds_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "tracksummary_fitter.root"),
        )
    )

    # s.addWriter(
    #     acts.examples.TrackFinderPerformanceWriter(
    #         level=acts.logging.INFO,
    #         inputProtoTracks="sorted_truth_particle_tracks"
    #         if directNavigation
    #         else "truth_particle_tracks",
    #         inputParticles="truth_seeds_selected",
    #         inputMeasurementParticlesMap="measurement_particles_map",
    #         filePath=str(outputDir / "performance_track_finder.root"),
    #     )
    # )
    #
    # s.addWriter(
    #     acts.examples.TrackFitterPerformanceWriter(
    #         level=acts.logging.INFO,
    #         inputTrajectories="trajectories",
    #         inputParticles="truth_seeds_selected",
    #         inputMeasurementParticlesMap="measurement_particles_map",
    #         filePath=str(outputDir / "performance_track_fitter.root"),
    #     )
    # )

    return s


assert "ACTS_ROOT" in os.environ and Path(os.environ["ACTS_ROOT"]).exists()

parser = argparse.ArgumentParser(description="Run GSF")
parser.add_argument("output", help="Override default output dir", type=str)
parser.add_argument("-n", "--events", help="number of events", type=int, default=1)
parser.add_argument("-j", "--jobs", help="number of jobs", type=int, default=-1)
args = vars(parser.parse_args())

if args["output"] is not None:
    output_dir = Path(args["output"])
else:
    output_dir = Path.cwd() / "output/kf"

acts_dir = Path(os.environ["ACTS_ROOT"])
odd_dir = acts_dir / "thirdparty/OpenDataDetector"

detector, trackingGeometry, _ = getOpenDataDetector(odd_dir)
field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args["events"], numThreads=args["jobs"], logLevel=acts.logging.INFO
)

runTruthTrackingKalman(
    trackingGeometry,
    field,
    digiConfigFile=odd_dir / "config/odd-digi-smearing-config.json",
    outputDir=output_dir,
    s=s,
).run()
