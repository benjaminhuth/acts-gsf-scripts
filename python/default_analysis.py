#!/bin/python3

import argparse
from pathlib import Path

import uproot
import matplotlib.pyplot as plt

from gsfanalysis.pandas_import import *
from gsfanalysis.trackstates_plots import *
from gsfanalysis.tracksummary_plots import *
from gsfanalysis.comparison_plot import *
from gsfanalysis.core_tail_utils import rms


def default_analysis(
    outputDir: Path, main_direction: str, pmax=100, pick_track=-1, pdfreport=None
):
    def save_to_pdfreport(fig):
        if pdfreport is not None:
            pdfreport.savefig(fig)

    summary_gsf, states_gsf = uproot_to_pandas(
        uproot.open(str(outputDir / "root/tracksummary_gsf.root:tracksummary")),
        uproot.open(str(outputDir / "root/trackstates_gsf.root:trackstates")),
    )

    summary_kf, states_kf = uproot_to_pandas(
        uproot.open(str(outputDir / "root/tracksummary_kf.root:tracksummary")),
        uproot.open(str(outputDir / "root/trackstates_kf.root:trackstates")),
    )

    print_basic_statistics([summary_gsf, summary_kf], ["GSF", "KF"])

    ####################
    # Collective plots #
    ####################
    if pick_track == -1:
        # fig, _ = ratio_residual_plot(summary_gsf, summary_kf, log_scale=True, bins=50)
        # fig.suptitle("Ratio/Res plot (log)")
        # fig.tight_layout()
        # save_to_pdfreport(fig)
        #
        # fig, _ = ratio_residual_plot(summary_gsf, summary_kf, log_scale=False)
        # fig.suptitle("Ratio/Res plot")
        # fig.tight_layout()
        # save_to_pdfreport(fig)

        fig, _ = make_full_residual_plot([summary_gsf, summary_kf], ["GSF", "KF"])
        fig.suptitle("Residuals")
        fig.tight_layout()
        save_to_pdfreport(fig)

        sets = [
            (summary_gsf, "GSF", "tab:blue"),
            (summary_kf, "KF", "tab:orange"),
        ]

        fig, _ = make_gsf_detailed_comparison_plots(sets)
        fig.suptitle("Comparison plot")
        fig.tight_layout()
        save_to_pdfreport(fig)

        # analysis.performance_at_trackstates(trackstates_gsf, 'x')

        for summary, states, name in zip(
            [summary_gsf, summary_kf], [states_gsf, states_kf], ["GSF", "KF"]
        ):
            fig, _ = plot_at_track_position(
                -1, states, name, main_direction, clip_abs=(0, 2 * pmax)
            )
            fig.suptitle("{} at first surface".format(name))
            fig.tight_layout()
            save_to_pdfreport(fig)

            fig, _ = plot_at_track_position(
                0, states, name, main_direction, clip_abs=(0, 2 * pmax)
            )
            fig.suptitle("{} at last surface".format(name))
            fig.tight_layout()
            save_to_pdfreport(fig)

            fig, ax = correlation_scatter_plot(summary, clip_res=(-8, 8))
            fig.suptitle("Correlation plots {}".format(name))
            fig.tight_layout()
            save_to_pdfreport(fig)

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

    default_analysis(path, args["main_direction"])
    plt.show()
