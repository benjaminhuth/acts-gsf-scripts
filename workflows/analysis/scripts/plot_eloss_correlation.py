import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *


summary_gsf, _ = uproot_to_pandas(
    uproot.open(f"{snakemake.input[0]}:tracksummary"),
    uproot.open(f"{snakemake.input[1]}:trackstates"),
)

summary_kf, _ = uproot_to_pandas(
    uproot.open(f"{snakemake.input[2]}:tracksummary"),
    uproot.open(f"{snakemake.input[3]}:trackstates"),
)

summary_gsf, summary_kf = remove_outliers_and_unify_index(
    summary_gsf.copy(), summary_kf.copy()
)


from gsfanalysis.tracksummary_plots import *

for summary, name in zip(
    [summary_gsf, summary_kf], ["GSF", "KF"]
):
    fig, ax = correlation_scatter_plot(summary, clip_res=(-8, 8))
    fig.suptitle(f"Correlation plots {name}")
    fig.tight_layout()

gsf_cmap = LinearSegmentedColormap.from_list(
    "gsf", [to_rgba("tab:orange", alpha=0), to_rgba("tab:orange", alpha=1)]
)
kf_cmap = LinearSegmentedColormap.from_list(
    "kf", [to_rgba("tab:blue", alpha=0), to_rgba("tab:blue", alpha=1)]
)
bins = [50, 50]


def plot(x_lambda, y_lambda, mask_lambda):
    fig, axes = plt.subplots(1, 2)

    sgsf = summary_gsf[mask_lambda(summary_gsf)]
    skf = summary_kf[mask_lambda(summary_kf)]

    if True:
        gsf_plot = axes[0].hist2d(
            x_lambda(sgsf), y_lambda(sgsf), bins, norm=LogNorm(), cmap=gsf_cmap
        )
        fig.colorbar(gsf_plot[3])

        kf_plot = axes[1].hist2d(
            x_lambda(skf), y_lambda(skf), bins, norm=LogNorm(), cmap=kf_cmap
        )
        fig.colorbar(kf_plot[3])
    else:
        axes[0].scatter(x_lambda(sgsf), y_lambda(sgsf), color="tab:orange")

        axes[1].scatter(x_lambda(skf), y_lambda(skf), color="tab:blue")

    axes[1].set_title("KF")
    axes[0].set_title("GSF")

    for ax in axes:
        ax.set_xlabel("$p_{fit} - p_{true} \quad [GeV]$")

    return fig, axes


####################
# Delta p vs res p #
####################

fig, axes = plot(
    lambda df: np.clip(df.res_eP_fit, -4, 4),
    lambda df: df.t_delta_p,
    lambda df: abs(df.t_delta_p_first_surface) < 0.01,
)

fig.suptitle("$\Delta E$ vs. $res_p$")

for ax in axes:
    ax.set_ylabel("$\Delta E \quad [GeV]$")

fig.tight_layout()
fig.savefig(snakemake.output[0])

##################################
# Delta p FIRST SURFACE vs res p #
##################################

fig2, axes2 = plot(
    lambda df: np.clip(df.res_eP_fit, -4, 4),
    lambda df: df.t_delta_p_first_surface,
    lambda df: abs(df.t_delta_p_first_surface) > 0.01,
)

fig2.suptitle("$\Delta E$ on first surface vs. $res_p$")

for ax in axes2:
    ax.set_ylabel("$\Delta E$ (after first surface) $\quad [GeV]$")

fig2.tight_layout()
fig2.savefig(snakemake.output[1])

if snakemake.config["plt_show"]:
    plt.show()
