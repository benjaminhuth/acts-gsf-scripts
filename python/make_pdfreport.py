#!/bin/python3

import subprocess
import argparse
from pathlib import Path

import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gsfanalysis.pandas_import import *
from gsfanalysis.trackstates_plots import *
from gsfanalysis.tracksummary_plots import *
from gsfanalysis.core_tail_utils import rms


def make_pdfreport(
    inputDir: Path,
    pdfreport,
    main_direction: str,
    pmax=100,
):
    def save_to_pdfreport(fig):
        fig.tight_layout()
        pdfreport.savefig(fig)

    summary_gsf, states_gsf = uproot_to_pandas(
        uproot.open(str(inputDir / "root/tracksummary_gsf.root:tracksummary")),
        uproot.open(str(inputDir / "root/trackstates_gsf.root:trackstates")),
    )

    summary_kf, states_kf = uproot_to_pandas(
        uproot.open(str(inputDir / "root/tracksummary_kf.root:tracksummary")),
        uproot.open(str(inputDir / "root/trackstates_kf.root:trackstates")),
    )

    fig, _ = make_full_residual_plot(
        [summary_kf, summary_gsf], ["KF", "GSF"], clip_std=4
    )
    fig.suptitle("Residuals with Outliers")
    fig.tight_layout()
    save_to_pdfreport(fig)

    summary_gsf, summary_kf = select_particles_and_unify_index(summary_gsf, summary_kf)

    fig, _ = make_full_residual_plot(
        [summary_kf, summary_gsf], ["KF", "GSF"], clip_std=4
    )
    fig.suptitle("Residuals without Outliers")
    fig.tight_layout()
    save_to_pdfreport(fig)

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].hist(np.clip(summary_gsf["res_eQOP_fit"], -2, 2), bins=100)
    ax[0].set_title("residual QOP")
    ax[1].hist(np.clip(summary_gsf["res_ePNORM_fit"], -1, 1), bins=100)
    ax[1].set_title("residual PNORM")
    save_to_pdfreport(fig)

    # analysis.performance_at_trackstates(trackstates_gsf, 'x')

    # for summary, states, name in zip(
    #     [summary_gsf, summary_kf], [states_gsf, states_kf], ["GSF", "KF"]
    # ):
    #     fig, _ = plot_at_track_position(
    #         -1, states, name, main_direction, clip_abs=(0, 2 * pmax)
    #     )
    #     fig.suptitle("{} at first surface".format(name))
    #     fig.tight_layout()
    #     save_to_pdfreport(fig)
    #
    #     fig, _ = plot_at_track_position(
    #         0, states, name, main_direction, clip_abs=(0, 2 * pmax)
    #     )
    #     fig.suptitle("{} at last surface".format(name))
    #     fig.tight_layout()
    #     save_to_pdfreport(fig)
    #
    #     fig, ax = correlation_scatter_plot(summary, clip_res=(-8, 8))
    #     fig.suptitle("Correlation plots {}".format(name))
    #     fig.tight_layout()
    #     save_to_pdfreport(fig)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="GSF/KF Analysis script")
    parser.add_argument("input_dir", help="where the root/ and csv/ dirs are")
    parser.add_argument("--main_direction", help="x for telescope and r for cylindrical", type=str, choices=["r", "x", "y", "z"], default="r")
    parser.add_argument("--disable_meas_holes", help="do not do the measurements holes plot", default=False, action="store_true")
    args = vars(parser.parse_args())
    # fmt: on

    path = Path(args["input_dir"])
    assert path.exists() and (path / "root").exists()

    pdfreport_path = path / "report_{}.pdf".format(path.name)
    pdfreport = PdfPages(pdfreport_path)
    make_pdfreport(path, pdfreport, args["main_direction"])
    pdfreport.close()

    # subprocess.run(["okular", str(pdfreport_path) ])
