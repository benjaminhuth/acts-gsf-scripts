import tempfile
import os
import random
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from contextlib import redirect_stdout
from datetime import datetime
from pprint import pprint

import pandas as pd
import numpy as np

from gsfanalysis.pandas_import import *
from gsfanalysis.core_tail_utils import *


def gsf_initializer():
    import acts
    import acts.examples
    from acts.examples.odd import getOpenDataDetector
    from acts.examples.geant4 import makeGeant4SimulationConfig
    from acts.examples.simulation import getG4DetectorContruction

    u = acts.UnitConstants
    defaultLogLevel = acts.logging.FATAL

    oddDir = Path(os.environ["ACTS_ROOT"]) / "thirdparty/OpenDataDetector"

    oddMaterialMap = oddDir / "data/odd-material-maps.root"
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

    global trackingGeometry
    global detector
    detector, trackingGeometry, decorators = getOpenDataDetector(
        oddDir, mdecorator=oddMaterialDeco, logLevel=defaultLogLevel
    )

    g4detectorConstruction = getG4DetectorContruction(detector)

    global field
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2 * u.T))

    global rnd
    rnd = acts.examples.RandomNumbers(seed=42)

    global g4conf
    g4conf = makeGeant4SimulationConfig(
        level=defaultLogLevel,
        detector=g4detectorConstruction,
        randomNumbers=rnd,
        inputParticles="particles_input",
        trackingGeometry=trackingGeometry,
        magneticField=field,
        # volumeMappings=,
        # materialMappings=,
    )
    g4conf.outputSimHits = "simhits"
    g4conf.outputParticlesInitial = "particles_initial"
    g4conf.outputParticlesFinal = "particles_final"


def gsf_subprocess(args, pars):
    import acts
    import acts.examples
    from acts.examples.simulation import (
        addParticleGun,
        MomentumConfig,
        EtaConfig,
        PhiConfig,
        ParticleConfig,
        addGeant4,
        addDigitization,
        addParticleSelection,
        ParticleSelectorConfig,
        addFatras,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TruthSeedRanges,
    )
    from acts.examples.geant4 import Geant4Simulation

    u = acts.UnitConstants
    defaultLogLevel = acts.logging.FATAL

    global trackingGeometry
    global g4conf
    global rnd
    global field

    digiConfigFile = "config/odd/odd-digi-smearing-config.json"

    components, weight_cutoff = pars
    print(
        datetime.now().strftime("%H:%M:%S"),
        multiprocessing.current_process().name,
        "PARS:",
        components,
        weight_cutoff,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        s = acts.examples.Sequencer(
            events=args["events"],
            numThreads=1,
            outputDir=tmp_dir,
            skip=0,
            logLevel=defaultLogLevel,
        )

        # realistic_stddev = acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns)
        addParticleGun(
            s,
            MomentumConfig(
                args["pmin"] * u.GeV, args["pmax"] * u.GeV, transverse=False
            ),
            EtaConfig(-3, 3),
            PhiConfig(0, 2 * np.pi),
            ParticleConfig(1, acts.PdgParticle.eElectron, randomizeCharge=False),
            rnd=rnd,
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0), stddev=acts.Vector4(0, 0, 0, 0)
            ),
            multiplicity=args["particles"],
            logLevel=defaultLogLevel,
        )

        # addFatras(
        #     s,
        #     # detector=detector,
        #     trackingGeometry=trackingGeometry,
        #     field=field,
        #     rnd=rnd,
        #     logLevel=defaultLogLevel,
        # )

        s.addAlgorithm(
            Geant4Simulation(
                level=defaultLogLevel,
                config=g4conf,
            )
        )

        addDigitization(
            s,
            trackingGeometry=trackingGeometry,
            field=field,
            rnd=rnd,
            digiConfigFile=digiConfigFile,
            logLevel=defaultLogLevel,
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
            seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
            logLevel=defaultLogLevel,
            truthSeedRanges=TruthSeedRanges(rho=(0.0, 1.0), nHits=(6, None)),
            initialVarInflation=6 * [100.0],
        )

        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=defaultLogLevel,
                inputParticles="truth_seeds_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="prototracks",
            )
        )

        low_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_LT01_cdf_nC6_O5.par"
        high_bhapprox = "/home/benjamin/Documents/athena/Tracking/TrkFitter/TrkGaussianSumFilter/Data/GeantSim_GT01_cdf_nC6_O5.par"

        gsfOptions = {
            "maxComponents": components,
            "abortOnError": False,
            "disableAllMaterialHandling": False,
            "betheHeitlerApprox": acts.examples.AtlasBetheHeitlerApprox.loadFromFiles(
                low_bhapprox,
                high_bhapprox,
            ),
            "finalReductionMethod": acts.examples.FinalReductionMethod.maxWeight,
            "weightCutoff": weight_cutoff,
            "level": defaultLogLevel,
        }

        s.addAlgorithm(
            acts.examples.TrackFittingAlgorithm(
                level=defaultLogLevel,
                inputMeasurements="measurements",
                inputSourceLinks="sourcelinks",
                inputProtoTracks="prototracks",
                inputInitialTrackParameters="estimatedparameters",
                outputTracks="tracks_gsf",
                directNavigation=False,
                pickTrack=-1,
                trackingGeometry=trackingGeometry,
                fit=acts.examples.makeGsfFitterFunction(
                    trackingGeometry, field, **gsfOptions
                ),
            )
        )

        s.addAlgorithm(
            acts.examples.TracksToTrajectories(
                level=acts.logging.WARNING,
                inputTracks="tracks_gsf",
                outputTrajectories="trajectories_gsf",
            )
        )

        s.addWriter(
            acts.examples.RootTrajectorySummaryWriter(
                level=acts.logging.WARNING,
                inputTrajectories="trajectories_gsf",
                inputParticles="truth_seeds_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=os.path.join(tmp_dir, "tracksummary_gsf.root"),
            )
        )

        s.run()
        del s

        timing = pd.read_csv(os.path.join(tmp_dir, "timing.tsv"), sep="\t")

        summary_gsf = uproot_to_pandas(
            uproot.open(os.path.join(tmp_dir, "tracksummary_gsf.root:tracksummary"))
        )

    result_row = args.copy()
    result_row["components"] = components
    result_row["weight_cutoff"] = weight_cutoff

    print(timing)

    fitter_timing = timing[timing["identifier"] == "Algorithm:TrackFittingAlgorithm"]
    assert len(fitter_timing) == 1
    result_row["timing"] = fitter_timing["time_perevent_s"]

    result_row["n_tracks"] = len(summary_gsf)

    summary_gsf_no_outliers = summary_gsf[summary_gsf["nOutliers"] == 0]
    result_row["n_outliers"] = len(summary_gsf) - len(summary_gsf_no_outliers)

    summary_gsf = add_core_to_df_quantile(
        summary_gsf, "res_eQOP_fit", args["core_quantile"]
    )
    result_row["core_quantile"] = args["core_quantile"]

    local_coors = ["LOC0", "LOC1", "PHI", "THETA", "QOP"]

    for coor in local_coors:
        res_key = f"res_e{coor}_fit"
        pull_key = f"pull_e{coor}_fit"

        result_row[f"res_{coor}_mean"] = np.mean(summary_gsf_no_outliers[res_key])
        result_row[f"res_{coor}_rms"] = rms(summary_gsf_no_outliers[res_key])
        result_row[f"pull_{coor}_mean"] = np.mean(summary_gsf_no_outliers[pull_key])
        result_row[f"pull_{coor}_std"] = np.std(summary_gsf_no_outliers[pull_key])

    return pd.DataFrame(result_row)


if __name__ == "__main__":

    assert "ACTS_ROOT" in os.environ and Path(os.environ["ACTS_ROOT"]).exists()

    # fmt: off
    parser = argparse.ArgumentParser(description='Run GSF sweep')
    parser.add_argument('-n','--events', type=int, default=3)
    parser.add_argument('-j','--jobs', type=int, default=3)
    parser.add_argument('--particles', type=int, default=1000)
    parser.add_argument('--pmin', type=float, default=0.5)
    parser.add_argument('--pmax', type=float, default=20)
    parser.add_argument('--core_quantile', type=float, default=0.95)
    parser.add_argument('-o', '--output', type=str, default="output_sweep/result.csv")
    args = vars(parser.parse_args())
    # fmt: on

    pprint(args)

    # components = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    # weight_cutoff = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1]

    components = [1, 4, 8, 12, 16]
    weight_cutoff = [1.0e-8, 1.0e-4, 1.0e-2]

    # components = [4,4,4,4]
    # weight_cutoff = [1.e-4,1.e-4, 1.e-4, 1.e-4]

    print("Sweep components:", components)
    print("Sweep weight cutoffs:", weight_cutoff)

    # Combine to grid
    pars = []
    for c in components:
        for wc in weight_cutoff:
            pars.append((c, wc))

    # Try to make the load a bit more balanced this way...
    random.shuffle(pars)

    with multiprocessing.Pool(args["jobs"], initializer=gsf_initializer) as p:
        dfs = p.map(partial(gsf_subprocess, args), pars, 1)

    # Print & export result
    result_df = pd.concat(dfs)
    print(result_df)

    out_dir = Path(args["output"])
    out_dir.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(out_dir, index=False)
