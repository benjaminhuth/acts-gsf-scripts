import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import textwrap
import math
from pprint import pprint

import acts
import acts.examples

from acts.examples.simulation import *
from acts.examples.reconstruction import *
from acts.examples.geant4 import *

import numpy as np


class GsfEnvironment:
    def __init__(self, args):
        self.args = args
        u = acts.UnitConstants

        if self.args["detector"] == "telescope":
            surface_distance = 50
            surface_thickness = 0.5
            surface_width = 1000

            telescopeConfig = {
                "positions": np.arange(
                    0, args["surfaces"] * surface_distance, surface_distance
                ).tolist(),
                # "offsets": [0,0],
                "bounds": [
                    surface_width,
                    surface_width,
                ],  # [args["surfaces"]*10, args["surfaces"]*10],
                "thickness": surface_thickness,
                "surfaceType": 0,
                "binValue": 0,
            }

            (
                self.detector,
                self.trackingGeometry,
                self.decorators,
            ) = acts.examples.TelescopeDetector.create(**telescopeConfig)

        elif args["detector"] == "odd":
            from acts.examples.odd import getOpenDataDetector

            oddDir = Path(os.environ["ACTS_ROOT"]) / "thirdparty/OpenDataDetector"

            oddMaterialMap = oddDir / "data/odd-material-maps.root"
            # digiConfigFile = oddDir / "config/odd-digi-smearing-config.json"
            self.digiConfigFile = "config/odd/odd-digi-smearing-config.json"
            self.seedingSel = oddDir / "config/odd-seeding-config.json"

            oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
            self.detector, self.trackingGeometry, self.decorators = getOpenDataDetector(
                oddDir, mdecorator=oddMaterialDeco
            )

        elif args["detector"] == "sphenix":
            raise "Not yet supported"

        ###################
        # Setup sequencer #
        ###################

        if args["output"] is not None:
            self.outputDir = Path(args["output"])
        else:
            self.outputDir = Path.cwd() / "output_{}".format(args["detector"])

        (self.outputDir / "root").mkdir(parents=True, exist_ok=True)
        (self.outputDir / "csv").mkdir(exist_ok=True, parents=True)

        self.field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))
        self.rnd = acts.examples.RandomNumbers(seed=42)

        self.defaultLogLevel = (
            acts.logging.FATAL if args["erroronly"] else acts.logging.INFO
        )
        self.defaultLogLevel = (
            acts.logging.DEBUG if args["debug"] else self.defaultLogLevel
        )
        self.defaultLogLevel = (
            acts.logging.VERBOSE if args["verbose"] else self.defaultLogLevel
        )

    def run_sequencer(self, s):
        u = acts.UnitConstants

        if self.args["detector"] == "telescope":
            realistic_stddev = acts.Vector4(
                0.0 * u.mm, 0.01 * u.mm, 0.01 * u.mm, 5.0 * u.ns
            )
        else:
            realistic_stddev = acts.Vector4(
                0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns
            )

        addParticleGun(
            s,
            MomentumConfig(
                self.args["pmin"] * u.GeV, self.args["pmax"] * u.GeV, transverse=False
            ),
            EtaConfig(-0.01, 0.01)
            if self.args["detector"] == "telescope"
            else EtaConfig(-3, 3),
            PhiConfig(-0.01, 0.01)
            if self.args["detector"] == "telescope"
            else PhiConfig(0, 2 * math.pi),
            ParticleConfig(1, acts.PdgParticle.eElectron, randomizeCharge=False),
            rnd=self.rnd,
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=realistic_stddev,
            ),
            multiplicity=self.args["particles"],
            logLevel=self.defaultLogLevel,
        )

        if not self.args["fatras"]:
            addGeant4(
                s,
                self.detector,
                self.trackingGeometry,
                self.field,
                self.rnd,
                postSelectParticles=ParticleSelectorConfig(
                    removeNeutral=True,
                    # removeEarlyEnergyLoss=True,
                    # removeEarlyEnergyLossThreshold=0.001
                ),
                logLevel=acts.logging.INFO,
            )
        else:
            addFatras(
                s,
                self.trackingGeometry,
                self.field,
                rnd=self.rnd,
                logLevel=self.defaultLogLevel,
                postSelectParticles=ParticleSelectorConfig(
                    removeNeutral=True,
                    # removeEarlyEnergyLoss=True,
                    # removeEarlyEnergyLossThreshold=0.001
                ),
                enableInteractions=not self.args["disable_fatras_interactions"],
            )

        digitization_args = {
            "s": s,
            "trackingGeometry": self.trackingGeometry,
            "field": self.field,
            # outputDirRoot=outputDir,
            "rnd": self.rnd,
            "outputDirCsv": None,  # "csv"
            "logLevel": self.defaultLogLevel,
        }

        if self.args["detector"] == "telescope":
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
                    }}""".format(
                    self.args["smearing"], self.args["smearing"]
                )
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
                digiConfigFile=self.digiConfigFile,
            )

        # addSeeding(
        #     s,
        #     trackingGeometry,
        #     field,
        #     inputParticles="particles_initial_selected",
        #     # particleSmearingSigmas=10*[ args["particle_smearing"] ],
        #     seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        #     logLevel=defaultLogLevel,
        #     truthSeedRanges=TruthSeedRanges(rho=(0.0, 1.0), nHits=(6, None)),
        #     initialVarInflation=6 * [100.0],
        # )

        seedingAlgorithm = (
            SeedingAlgorithm.TruthEstimated
            if self.args["seeding"] == "estimated"
            else SeedingAlgorithm.TruthSmeared
        )
        particleSmearingSigmas = (
            ParticleSmearingSigmas(10 * [0.0])
            if self.args["seeding"] == "truth"
            else ParticleSmearingSigmas()
        )

        addSeeding(
            s,
            self.trackingGeometry,
            self.field,
            seedingAlgorithm=seedingAlgorithm,
            truthEstimatedSeedingAlgorithmConfigArg=TruthEstimatedSeedingAlgorithmConfigArg(
                deltaR=(0.0, np.inf)
            ),
            truthSeedRanges=TruthSeedRanges(rho=(0.0, 1.0), nHits=(3, None)),
            particleSmearingSigmas=particleSmearingSigmas,
            initialVarInflation=6 * [100],
            geoSelectionConfigFile=self.seedingSel,
            inputParticles="particles_input",
            logLevel=self.defaultLogLevel,
        )

        particles = (
            "truth_seeded_particles"
            if self.args["seeding"] == "estimated"
            else "truth_seeds_selected"
        )

        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=self.defaultLogLevel,
                inputParticles=particles,
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="prototracks",
            )
        )

        kalmanOptions = {
            "multipleScattering": True,
            "energyLoss": True,
            "reverseFilteringMomThreshold": 0.0,
            "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
            "level": acts.logging.INFO,
        }

        s.addAlgorithm(
            acts.examples.TrackFittingAlgorithm(
                level=acts.logging.VERBOSE
                if self.args["pick"] != -1
                else self.defaultLogLevel,
                inputMeasurements="measurements",
                inputSourceLinks="sourcelinks",
                inputProtoTracks="prototracks",
                inputInitialTrackParameters="estimatedparameters",
                outputTracks="tracks_kf",
                directNavigation=False,
                pickTrack=self.args["pick"],
                trackingGeometry=self.trackingGeometry,
                fit=acts.examples.makeKalmanFitterFunction(
                    self.trackingGeometry, self.field, **kalmanOptions
                ),
            )
        )

        low_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_LT01_cdf_nC6_O5.par"
        high_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_GT01_cdf_nC6_O5.par"

        gsfOptions = {
            "maxComponents": self.args["components"],
            "abortOnError": False,
            "disableAllMaterialHandling": False,
            "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
                low_bhapprox,
                high_bhapprox,
            ),
            "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
            "weightCutoff": self.args["cutoff"],
            "level": acts.logging.VERBOSE
            if self.args["pick"] != -1 or self.args["verbose"]
            else acts.logging.ERROR,
            # "minimalMomentumThreshold": 0.,
        }
        pprint(gsfOptions)

        s.addAlgorithm(
            acts.examples.TrackFittingAlgorithm(
                level=self.defaultLogLevel,
                inputMeasurements="measurements",
                inputSourceLinks="sourcelinks",
                inputProtoTracks="prototracks",
                inputInitialTrackParameters="estimatedparameters",
                outputTracks="tracks_gsf",
                directNavigation=False,
                pickTrack=self.args["pick"],
                trackingGeometry=self.trackingGeometry,
                fit=acts.examples.makeGsfFitterFunction(
                    self.trackingGeometry, self.field, **gsfOptions
                ),
            )
        )

        for fitter in ("gsf", "kf"):
            trajectories = "trajectories_" + fitter
            tracks = "tracks_" + fitter

            s.addAlgorithm(
                acts.examples.TracksToTrajectories(
                    level=acts.logging.WARNING,
                    inputTracks=tracks,
                    outputTrajectories=trajectories,
                )
            )

            s.addWriter(
                acts.examples.RootTrajectorySummaryWriter(
                    level=acts.logging.WARNING,
                    inputTrajectories=trajectories,
                    inputParticles=particles,
                    inputMeasurementParticlesMap="measurement_particles_map",
                    filePath=str(
                        self.outputDir / "root/tracksummary_{}.root".format(fitter)
                    ),
                )
            )

            s.addWriter(
                acts.examples.RootTrajectoryStatesWriter(
                    level=acts.logging.WARNING,
                    inputTrajectories=trajectories,
                    inputParticles=particles,
                    inputSimHits="simhits",
                    inputMeasurementParticlesMap="measurement_particles_map",
                    inputMeasurementSimHitsMap="measurement_simhits_map",
                    filePath=str(
                        self.outputDir / "root/trackstates_{}.root".format(fitter)
                    ),
                )
            )

        s.run()

        # Return gsf options as seriazable dict
        gsfConfig = gsfOptions.copy()

        del gsfConfig["betheHeitlerApprox"]
        del gsfConfig["level"]

        gsfConfig["finalReductionMethod"] = str(gsfConfig["finalReductionMethod"])
        gsfConfig["low_bhapprox"] = low_bhapprox
        gsfConfig["high_bhapprox"] = high_bhapprox

        return gsfConfig
