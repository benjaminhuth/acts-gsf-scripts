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


@acts.examples.NamedTypeArgs(
    preselectParticles=ParticleSelectorConfig,
)
def myAddFatras(
    s: acts.examples.Sequencer,
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    outputDirCsv: Optional[Union[Path, str]] = None,
    outputDirRoot: Optional[Union[Path, str]] = None,
    rnd: Optional[acts.examples.RandomNumbers] = None,
    preselectParticles: Optional[ParticleSelectorConfig] = ParticleSelectorConfig(),
    logLevel: Optional[acts.logging.Level] = None,
) -> None:
    customLogLevel = acts.examples.defaultLogging(s, logLevel)

    # Preliminaries
    rnd = rnd or acts.examples.RandomNumbers()

    # Selector
    if preselectParticles is not None:
        particles_selected = "particles_selected"
        addParticleSelection(
            s,
            preselectParticles,
            inputParticles="particles_input",
            outputParticles=particles_selected,
        )
    else:
        particles_selected = "particles_input"

    # Simulation
    alg = acts.examples.FatrasSimulation(
        level=customLogLevel(),
        inputParticles=particles_selected,
        outputParticlesInitial="particles_initial",
        outputParticlesFinal="particles_final",
        outputSimHits="simhits",
        randomNumbers=rnd,
        trackingGeometry=trackingGeometry,
        magneticField=field,
        generateHitsOnSensitive=True,
        emEnergyLossRadiation=True,
    )

    # Sequencer
    s.addAlgorithm(alg)

    # Output
    addSimWriters(
        s,
        alg.config.outputSimHits,
        outputDirCsv,
        outputDirRoot,
        logLevel=logLevel,
    )

    return s

#####################
# Cmd line handling #
#####################

parser = argparse.ArgumentParser(description='Run GSF Telescope')
parser.add_argument('-p','--pick', help='pick track', type=int, default=-1)
parser.add_argument('-n','--events', help='number of events', type=int, default=10)
parser.add_argument('-j','--jobs', help='number of jobs', type=int, default=-1)
parser.add_argument('-s','--skip', help='number of skipped events', type=int, default=0)
parser.add_argument('-c','--components', help='number of GSF components', type=int, default=4)
parser.add_argument('--surfaces', help='number of telescope surfaces', type=int, default=5)
parser.add_argument('--pmin', help='minimum momentum for particle gun', type=float, default=1.0)
parser.add_argument('--pmax', help='maximum momentum for particle gun', type=float, default=20.0)
parser.add_argument('--erroronly', help='set loglevel to show only errors (except sequencer)', default=False, action="store_true")
parser.add_argument('-v', '--verbose', help='set loglevel to VERBOSE (except for sequencer)', default=False, action="store_true")
parser.add_argument('--visualize', help='visualize the GDML geometry and exit', default=False, action="store_true")
parser.add_argument('--fatras', help='use fatras instead of geant4', default=False, action="store_true")
args = vars(parser.parse_args())

##################
# Build geometry #
##################

surface_distance = 50
surface_thickness = 1
surface_width = 1000

telescopeConfig = {
    "positions": np.arange(0, args["surfaces"]*surface_distance, surface_distance).tolist(),
    # "offsets": foo,
    "bounds": [surface_width, surface_width], #[args["surfaces"]*10, args["surfaces"]*10],
    "thickness": surface_thickness,
    "surfaceType": 0,
    "binValue": 0,
}

# pprint.pprint(telescopeConfig)


detector, trackingGeometry, decorators = acts.examples.TelescopeDetector.create(
    **telescopeConfig
)

# Geant 4
gdml_file = "gdml/telescope.gdml"
make_gdml(gdml_file, args["surfaces"], surface_width, surface_thickness, surface_distance, args["visualize"])

if args["visualize"]:
    exit(0);

g4detector = GdmlDetectorConstruction(gdml_file)

###################
# Setup sequencer #
###################

outputDir = Path(".")
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)
use_geant = False if args["fatras"] else True
defaultLogLevel = acts.logging.FATAL if args["erroronly"] else acts.logging.INFO
defaultLogLevel = acts.logging.VERBOSE if args["verbose"] else defaultLogLevel

if use_geant:
    args["jobs"] = 1

s = acts.examples.Sequencer(
    events=args["events"],
    numThreads=args["jobs"],
    outputDir=str(outputDir),
    skip=args["skip"],
    logLevel=acts.logging.INFO,
)

(outputDir / "csv").mkdir(exist_ok=True, parents=True)
s.addWriter(
    acts.examples.CsvTrackingGeometryWriter(
        level=defaultLogLevel,
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
    logLevel=defaultLogLevel,
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
        logLevel=defaultLogLevel,
    )
else:
    myAddFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        outputDirCsv="csv",
        logLevel=defaultLogLevel,
    )

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile="telescope-digi-smearing-config.json",
    # outputDirRoot=outputDir,
    rnd=rnd,
    outputDirCsv=None, #"csv"
    logLevel=defaultLogLevel,
)

addParticleSelection(
    s,
    ParticleSelectorConfig(removeNeutral=True),
    inputParticles="particles_initial",
    outputParticles="particles_initial_selected_post_sim",
    logLevel=defaultLogLevel,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    inputParticles="particles_initial_selected_post_sim",
    seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    logLevel=defaultLogLevel,
    truthSeedRanges=TruthSeedRanges(rho=(0.0, 1.0), nHits=(6,None)),
)

s.addAlgorithm(
    acts.examples.TruthTrackFinder(
        level=defaultLogLevel,
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
        level=acts.logging.VERBOSE if args["pick"] != -1 else defaultLogLevel,
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="prototracks",
        inputInitialTrackParameters="estimatedparameters",
        outputTrajectories="trajectories",
        directNavigation=False,
        pickTrack=args["pick"],
        trackingGeometry=trackingGeometry,
        fit=acts.examples.makeKalmanFitterFunction(
            trackingGeometry, field, **kalmanOptions
        ),
    )
)

gsfOptions = {
    "maxComponents": args["components"],
    "abortOnError": False,
    "disableAllMaterialHandling": False,
    "lowBetheHeitlerPath": "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_LT01_cdf_nC6_O5.par",
    "highBetheHeitlerPath": "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_GT01_cdf_nC6_O5.par",
    "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
}
pprint.pprint(gsfOptions)

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=acts.logging.VERBOSE if args["pick"] != -1 else defaultLogLevel,
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="prototracks",
        inputInitialTrackParameters="estimatedparameters",
        outputTrajectories="gsf_trajectories",
        directNavigation=False,
        pickTrack=args["pick"],
        trackingGeometry=trackingGeometry,
        fit=acts.examples.makeGsfFitterFunction(
            trackingGeometry, field, **gsfOptions
        ),
    )
)

s.addWriter(
    acts.examples.RootTrajectorySummaryWriter(
        level=acts.logging.ERROR,
        inputTrajectories="gsf_trajectories",
        inputParticles="truth_seeds_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
        filePath=str(outputDir / "root/tracksummary_gsf.root"),
    )
)

s.addWriter(
    acts.examples.RootTrajectoryStatesWriter(
        level=acts.logging.ERROR,
        inputTrajectories="gsf_trajectories",
        inputParticles="truth_seeds_selected",
        inputSimHits="simhits",
        inputMeasurementParticlesMap="measurement_particles_map",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        filePath=str(outputDir / "root/trackstates_gsf.root"),
    )
)

s.addWriter(
    acts.examples.RootTrajectorySummaryWriter(
        level=acts.logging.ERROR,
        inputTrajectories="trajectories",
        inputParticles="truth_seeds_selected",
        inputMeasurementParticlesMap="measurement_particles_map",
        filePath=str(outputDir / "root/tracksummary_kf.root"),
    )
)

s.addWriter(
    acts.examples.RootTrajectoryStatesWriter(
        level=acts.logging.ERROR,
        inputTrajectories="trajectories",
        inputParticles="truth_seeds_selected",
        inputSimHits="simhits",
        inputMeasurementParticlesMap="measurement_particles_map",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        filePath=str(outputDir / "root/trackstates_kf.root"),
    )
)

s.run()
