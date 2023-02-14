from pathlib import Path
from typing import Union, Optional
import math
import pprint
import os
from tempfile import NamedTemporaryFile
import textwrap
import shutil

import argparse

import acts
import acts.examples
from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.geant4 import GdmlDetectorConstruction

import numpy as np

u = acts.UnitConstants

#####################
# Cmd line handling #
#####################

parser = argparse.ArgumentParser(description='Run GSF Telescope')
parser.add_argument("detector", choices=["telescope", "odd", "sphenix"], help="which detector geometry to use")
parser.add_argument('-p','--pick', help='pick track', type=int, default=-1)
parser.add_argument('-n','--events', help='number of events', type=int, default=1)
parser.add_argument('-j','--jobs', help='number of jobs', type=int, default=-1)
parser.add_argument('-s','--skip', help='number of skipped events', type=int, default=0)
parser.add_argument('-c','--components', help='number of GSF components', type=int, default=4)
parser.add_argument('--surfaces', help='number of telescope surfaces', type=int, default=10)
parser.add_argument('--pmin', help='minimum momentum for particle gun', type=float, default=4.0)
parser.add_argument('--pmax', help='maximum momentum for particle gun', type=float, default=4.0)
parser.add_argument('--erroronly', help='set loglevel to show only errors (except sequencer)', default=False, action="store_true")
parser.add_argument('--debug', help='set loglevel to show debug', default=False, action="store_true")
parser.add_argument('-v', '--verbose', help='set loglevel to VERBOSE (except for sequencer)', default=False, action="store_true")
parser.add_argument('--fatras', help='use FATRAS instead of Geant4', default=False, action="store_true")
parser.add_argument('--particle_smearing', help='value used for initial particle smearing', type=float, default=None)
parser.add_argument('--smearing', help='stddev for the pixel smearing', type=float, default=0.01)
parser.add_argument('--plt_show', help='Call plt.show() in the end', action="store_true", default=False)
parser.add_argument('--disable_fatras_interactions', help="no  interactions in FATRAS", default=False, action="store_true")
parser.add_argument('-o', '--output', help="Override default output dir")
args = vars(parser.parse_args())

##################
# Build geometry #
##################

if args["detector"] == "telescope":
    surface_distance = 50
    surface_thickness = 0.5
    surface_width = 1000

    telescopeConfig = {
        "positions": np.arange(0, args["surfaces"]*surface_distance, surface_distance).tolist(),
        # "offsets": [0,0],
        "bounds": [surface_width, surface_width], #[args["surfaces"]*10, args["surfaces"]*10],
        "thickness": surface_thickness,
        "surfaceType": 0,
        "binValue": 0,
    }

    detector, trackingGeometry, decorators = acts.examples.TelescopeDetector.create(
        **telescopeConfig
    )
    
elif args["detector"] == "odd":
    from acts.examples.odd import getOpenDataDetector
    
    oddDir = Path(os.environ["ACTS_ROOT"]) / "thirdparty/OpenDataDetector"

    oddMaterialMap = oddDir / "data/odd-material-maps.root"
    # digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
    digiConfigFile = "config/odd/odd-digi-smearing-config.json"
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

    detector, trackingGeometry, decorators = getOpenDataDetector(
        oddDir, mdecorator=oddMaterialDeco
    )
    
elif args["detector"] == "sphenix":
    raise "Not yet supported"

###################
# Setup sequencer #
###################

if args["output"] is not None:
    outputDir = Path(args["output"])
else:
    outputDir = Path.cwd() / "output_{}".format(args["detector"])
    
(outputDir / "root").mkdir(parents=True, exist_ok=True)
(outputDir / "csv").mkdir(exist_ok=True, parents=True)

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)
use_geant = False if args["fatras"] else True
defaultLogLevel = acts.logging.FATAL if args["erroronly"] else acts.logging.INFO
defaultLogLevel = acts.logging.DEBUG if args["debug"] else defaultLogLevel
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

s.addWriter(
    acts.examples.CsvTrackingGeometryWriter(
        level=defaultLogLevel,
        trackingGeometry=trackingGeometry,
        outputDir=str(outputDir / "csv"),
        writePerEvent=False,
    )
)

realistic_stddev = acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns)
addParticleGun(
    s,
    MomentumConfig(args['pmin'] * u.GeV, args['pmax'] * u.GeV, transverse=False),
    EtaConfig(-0.01, 0.01) if args["detector"] == "telescope" else EtaConfig(-3, 3),
    PhiConfig(-0.01, 0.01) if args["detector"] == "telescope" else PhiConfig(0, 2*np.pi),
    ParticleConfig(1, acts.PdgParticle.eElectron, randomizeCharge=False),
    rnd=rnd,
    vtxGen=acts.examples.GaussianVertexGenerator(
        mean=acts.Vector4(0, 0, 0, 0),
        stddev=acts.Vector4(0, 0, 0, 0), # if args["detector"] == "telescope" else realistic_stddev,
    ),
    multiplicity=1000,
    logLevel=defaultLogLevel,
)

if use_geant:
    addGeant4(
        s,
        detector,
        trackingGeometry,
        field,
        rnd,
        outputDirCsv="csv",
        logLevel=acts.logging.INFO,
    )
else:
    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        outputDirCsv="csv",
        logLevel=defaultLogLevel,
        enableInteractions=not args["disable_fatras_interactions"],
    )
    
digitization_args = {
    "s": s,
    "trackingGeometry": trackingGeometry,
    "field": field,
    # outputDirRoot=outputDir,
    "rnd": rnd,
    "outputDirCsv": None, #"csv"
    "logLevel": defaultLogLevel,
}

if args["detector"] == "telescope":
    with NamedTemporaryFile() as fp:
        content = """
            {{
                "acts-geometry-hierarchy-map" : {{
                    "format-version" : 0,
                    "value-identifier" : "digitization-configuration"
                }},
                "entries": [
                    {{
                        "volume" : 0,
                        "value" : {{
                            "smearing" : [
                                {{"index" : 0, "mean" : 0.0, "stddev" : {}, "type" : "Gauss"}},
                                {{"index" : 1, "mean" : 0.0, "stddev" : {}, "type" : "Gauss"}}
                            ]
                        }}
                    }}
                ]
            }}""".format(args["smearing"], args["smearing"])
        content = textwrap.dedent(content)
        fp.write(str.encode(content))
        fp.flush()

        addDigitization(
            **digitization_args,
            digiConfigFile=fp.name,
        )
else:
    addDigitization(
        **digitization_args,
        digiConfigFile=digiConfigFile,
    )
    

addParticleSelection(
    s,
    ParticleSelectorConfig(
        removeNeutral=True,
        # removeEarlyEnergyLoss=True,
        # removeEarlyEnergyLossThreshold=0.001
    ),
    inputParticles="particles_initial",
    # inputSimHits="simhits",
    outputParticles="particles_initial_selected_post_sim",
    logLevel=defaultLogLevel,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    inputParticles="particles_initial_selected_post_sim",
    # particleSmearingSigmas=10*[ args["particle_smearing"] ],
    seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    logLevel=defaultLogLevel,
    truthSeedRanges=TruthSeedRanges(rho=(0.0, 1.0), nHits=(6,None)),
    initialVarInflation=6*[100.0],
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
        outputTrajectories="kf_trajectories",
        outputTracks="kf_tracks",
        directNavigation=False,
        pickTrack=args["pick"],
        trackingGeometry=trackingGeometry,
        fit=acts.examples.makeKalmanFitterFunction(
            trackingGeometry, field, **kalmanOptions
        ),
    )
)

low_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_LT01_cdf_nC6_O5.par"
high_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_GT01_cdf_nC6_O5.par"

gsfOptions = {
    "maxComponents": args["components"],
    "abortOnError": False,
    "disableAllMaterialHandling": False,
    "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
        low_bhapprox,
        high_bhapprox,
    ),
    "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
    "weightCutoff": 1.e-4,
    "level": acts.logging.VERBOSE if args["pick"] != -1 or args["verbose"] else acts.logging.ERROR,
    # "minimalMomentumThreshold": 0.,
}
pprint.pprint(gsfOptions)

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=defaultLogLevel,
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="prototracks",
        inputInitialTrackParameters="estimatedparameters",
        outputTrajectories="gsf_trajectories",
        outputTracks="gsf_tracks",
        directNavigation=False,
        pickTrack=args["pick"],
        trackingGeometry=trackingGeometry,
        fit=acts.examples.makeGsfFitterFunction(
            trackingGeometry, field, **gsfOptions
        ),
    )
)

for trajectories, postfix in zip(("gsf_trajectories", "kf_trajectories"), ("gsf", "kf")):
    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=acts.logging.WARNING,
            inputTrajectories=trajectories,
            inputParticles="truth_seeds_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "root/tracksummary_{}.root".format(postfix)),
        )
    )

    s.addWriter(
        acts.examples.RootTrajectoryStatesWriter(
            level=acts.logging.WARNING,
            inputTrajectories=trajectories,
            inputParticles="truth_seeds_selected",
            inputSimHits="simhits",
            inputMeasurementParticlesMap="measurement_particles_map",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(outputDir / "root/trackstates_{}.root".format(postfix)),
        )
    )

s.run()
del s

# Bring geometry file to top for convenience
shutil.copyfile(outputDir / "csv/detectors.csv", Path.cwd() / "detectors.csv")

# Save configuration
import json
with open(outputDir/"config.json", "w") as f:
    gsfConfig = gsfOptions.copy()

    del gsfConfig["betheHeitlerApprox"]
    del gsfConfig["level"]
    gsfConfig["finalReductionMethod"] = str(gsfConfig["finalReductionMethod"])
    gsfConfig["low_bhapprox"] = low_bhapprox
    gsfConfig["high_bhapprox"] = high_bhapprox

    config = args.copy()
    config["gsf"] = gsfConfig
    json.dump(config, f, indent=4)

# Analysis
import analysis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uproot
import time

summary_gsf, states_gsf = analysis.uproot_to_pandas(
    uproot.open(str(outputDir / "root/tracksummary_gsf.root:tracksummary")),
    uproot.open(str(outputDir / "root/trackstates_gsf.root:trackstates"))
)

summary_kf, states_kf = analysis.uproot_to_pandas(
    uproot.open(str(outputDir / "root/tracksummary_kf.root:tracksummary")),
    uproot.open(str(outputDir / "root/trackstates_kf.root:trackstates"))
)

pdfreport = PdfPages(outputDir/'report.pdf')

bad = summary_gsf[ ~summary_gsf["res_eLOC0_fit"].between(-100,100) | ~summary_gsf["res_eLOC1_fit"].between(-100,100) ]
keys = ["event_nr", "multiTraj_nr", "nStates", "nMeasurements", "nOutliers",'res_eLOC0_fit', 'res_eLOC1_fit', 'res_ePHI_fit', 'res_eTHETA_fit', 'res_eQOP_fit', 'res_eT_fit']
print(bad[keys])

####################
# Collective plots #
####################
if args["pick"] == -1:
    fig, _ = analysis.ratio_residual_plot(summary_gsf, summary_kf, log_scale=True, bins=50)
    fig.suptitle("Ratio/Res plot (log)")
    fig.tight_layout()
    pdfreport.savefig(fig)

    fig, _ = analysis.ratio_residual_plot(summary_gsf, summary_kf, log_scale=False)
    fig.suptitle("Ratio/Res plot")
    fig.tight_layout()
    pdfreport.savefig(fig)
    
    main_direction = "x" if args["detector"] == "telescope" else "r"

    #analysis.performance_at_trackstates(trackstates_gsf, 'x')
    
    for summary, states, name in zip([summary_gsf, summary_kf], [states_gsf, states_kf], ["GSF", "KF"]):
        fig, _ = analysis.make_full_residual_plot(summary)
        fig.suptitle("{} residuals")
        fig.tight_layout()
        pdfreport.savefig(fig)
        
        fig, _ = analysis.plot_at_track_position(-1, states, name, main_direction, clip_abs=(0,2*args["pmax"]))
        fig.suptitle("{} at first surface".format(name))
        fig.tight_layout()
        pdfreport.savefig(fig)
        
        fig, _ = analysis.plot_at_track_position(0, states, name, main_direction, clip_abs=(0,2*args["pmax"]))
        fig.suptitle("{} at last surface".format(name))
        fig.tight_layout()
        pdfreport.savefig(fig)

        fig, ax = analysis.correlation_scatter_plot(summary, states, clip_res=(-8,8))
        fig.suptitle("Correlation plots {}".format(name))
        fig.tight_layout()
        pdfreport.savefig(fig)
    
#########################
# Single particle plots #
#########################
else:
    pass
    # fig, ax = analysis.single_particle_momentumplot(summary_gsf, states_gsf, "fwd", "bwd")
    # ax.set_title("GSF single particle")
    # fig.tight_layout()
    # pdfreport.savefig(fig)
    # 
    # fig, ax = analysis.single_particle_momentumplot(summary_kf, states_kf, "prt", "flt")
    # ax.set_title("KF single particle")
    # fig.tight_layout()
    # pdfreport.savefig(fig)

pdfreport.close()

if args["plt_show"]:
    plt.show()

