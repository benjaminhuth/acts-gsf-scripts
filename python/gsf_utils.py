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

            self.seedingSel = Path("config/telescope/telescope-seeding-config.json")

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

        assert self.seedingSel.exists()

        ###################
        # Setup sequencer #
        ###################

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

        #######################
        # Setup Geant4 config #
        #######################
        if not args["fatras"]:
            from acts.examples.geant4 import makeGeant4SimulationConfig

            self.g4detectorConstruction = getG4DetectorContruction(self.detector)

            self.g4conf = makeGeant4SimulationConfig(
                level=self.defaultLogLevel,
                detector=self.g4detectorConstruction,
                randomNumbers=self.rnd,
                inputParticles="particles_input",
                trackingGeometry=self.trackingGeometry,
                magneticField=self.field,
                # volumeMappings=,
                # materialMappings=,
            )
            self.g4conf.outputSimHits = "simhits"
            self.g4conf.outputParticlesInitial = "particles_initial"
            self.g4conf.outputParticlesFinal = "particles_final"

    def run_sequencer(self, s, outputDir: Path):
        u = acts.UnitConstants

        (outputDir / "root").mkdir(parents=True, exist_ok=True)
        (outputDir / "csv").mkdir(parents=True, exist_ok=True)

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
            from acts.examples.geant4 import Geant4Simulation

            s.addAlgorithm(
                Geant4Simulation(
                    level=self.defaultLogLevel,
                    config=self.g4conf,
                )
            )

        else:
            addFatras(
                s,
                self.trackingGeometry,
                self.field,
                rnd=self.rnd,
                logLevel=self.defaultLogLevel,
                preSelectParticles=None,
                postSelectParticles=None,
                enableInteractions=not self.args["disable_fatras_interactions"],
            )

        addParticleSelection(
            s,
            inputParticles="particles_initial",
            # inputSimHits="simhits",
            config=ParticleSelectorConfig(
                removeNeutral=True,
                rho=(0, 2 * u.mm),
                absZ=(0, 200 * u.mm),
                # removeEarlyEnergyLoss=True,
                # removeEarlyEnergyLossThreshold=0.001
            ),
            outputParticles="particles_initial_selected",
            logLevel=self.defaultLogLevel,
        )

        def thisAddDigitization(digiConfigFile):
            addDigitization(
                s=s,
                trackingGeometry=self.trackingGeometry,
                field=self.field,
                rnd=self.rnd,
                # outputDirRoot=outputDir,
                # outputDirCsv=(outputDir / "csv")
                logLevel=self.defaultLogLevel,
                digiConfigFile=digiConfigFile,
            )

        if self.args["detector"] == "telescope":
            with NamedTemporaryFile() as fp:
                # fmt: off
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
                        self.args["digi_smearing"],
                        self.args["digi_smearing"]
                    )
                    # fmt: off
                                
                content = textwrap.dedent(content)
                fp.write(str.encode(content))
                fp.flush()
                
                thisAddDigitization(fp.name)
        else:
            thisAddDigitization(self.digiConfigFile)

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
            inputParticles="particles_input",  # "particles_initial_selected",
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

        kalman = False if "no_kalman" in self.args and self.args["no_kalman"] else True

        if kalman:
            kalmanOptions = {
                "multipleScattering": True,
                "energyLoss": True,
                "reverseFilteringMomThreshold": 0.0,
                "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
                "level": self.defaultLogLevel,
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
            else acts.logging.ERROR  # self.defaultLogLevel
            # "minimalMomentumThreshold": 0.,
        }

        if not self.args["erroronly"]:
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

        for fitter in ["gsf", "kf"] if kalman else ["gsf"]:
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
                        outputDir / "root/tracksummary_{}.root".format(fitter)
                    ),
                )
            )

            no_states = "no_states" in self.args and self.args["no_states"]
            if not no_states:
                s.addWriter(
                    acts.examples.RootTrajectoryStatesWriter(
                        level=acts.logging.WARNING,
                        inputTrajectories=trajectories,
                        inputParticles=particles,
                        inputSimHits="simhits",
                        inputMeasurementParticlesMap="measurement_particles_map",
                        inputMeasurementSimHitsMap="measurement_simhits_map",
                        filePath=str(
                            outputDir / "root/trackstates_{}.root".format(fitter)
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
