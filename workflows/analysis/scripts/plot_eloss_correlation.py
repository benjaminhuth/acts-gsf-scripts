import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *


gsf_cmap = LinearSegmentedColormap.from_list(
    "gsf", [to_rgba("tab:orange", alpha=0), to_rgba("tab:orange", alpha=1)]
)
kf_cmap = LinearSegmentedColormap.from_list(
    "kf", [to_rgba("tab:blue", alpha=0), to_rgba("tab:blue", alpha=1)]
)


def plot(dfgsf, dfkf, x_lambda, y_lambda):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    hist_args = {
        "bins": [50, 50],
        "cmin": 10,
        "norm": LogNorm(),
    }

    if True:
        gsf_plot = axes[0].hist2d(
            x_lambda(dfgsf), y_lambda(dfgsf), cmap=gsf_cmap, **hist_args
        )
        # fig.colorbar(gsf_plot[3])

        kf_plot = axes[1].hist2d(
            x_lambda(dfkf), y_lambda(dfkf), cmap=kf_cmap, **hist_args
        )
        # fig.colorbar(kf_plot[3])
    else:
        axes[0].scatter(x_lambda(dfgsf), y_lambda(dfgsf), color="tab:orange")

        axes[1].scatter(x_lambda(dfkf), y_lambda(dfkf), color="tab:blue")

    axes[1].set_title("KF")
    axes[0].set_title("GSF")

    for ax in axes:
        ax.set_xlabel("$p_{fit} - p_{true} \quad [GeV]$")

    return fig, axes



#############
# Load data #
#############


with open(snakemake.input[0], "rb") as f:
    summary_gsf = pickle.load(f)
    

with open(snakemake.input[1], "rb") as f:
    summary_kf = pickle.load(f)

summary_gsf, summary_kf = select_particles_and_unify_index(
    summary_gsf.copy(), summary_kf.copy(), max_eloss_first_surface=np.inf,
)

assert len(summary_gsf) > 0
assert len(summary_kf) > 0
assert len(summary_gsf) == len(summary_kf)

print(summary_gsf.shape)
print(summary_gsf.head(10))
print(summary_kf.shape)
print(summary_kf.head(10))

assert (summary_gsf.event_nr == summary_kf.event_nr).all()
assert (summary_gsf.track_nr == summary_kf.track_nr).all()


# This is not equal because of different track lenths!!!
# assert (summary_gsf.t_delta_p == summary_kf.t_delta_p).all()
# assert (summary_gsf.t_delta_p_first_surface == summary_kf.t_delta_p_first_surface).all()

####################
# Delta p vs res p #
####################

clip=(-0.8,0.2)

fig, axes = plot(
    summary_gsf.copy(),
    summary_kf.copy(),
    # lambda df: df.res_ePNORM_fit[ df.res_ePNORM_fit.between(*clip) ],
    # lambda df: df.t_delta_p[ df.res_ePNORM_fit.between(*clip) ],
    lambda df: df.res_eQOP_fit[ df.res_eQOP_fit.between(*clip) ],
    lambda df: df.t_delta_p[ df.res_eQOP_fit.between(*clip) ],
)



fig.suptitle("$\Delta E$ vs. $res_p$")

for ax in axes:
    ax.set_ylabel("$\Delta E \quad [GeV]$")
    ax.set_xlim(*clip)
    ax.set_ylim(-10, 0)

fig.tight_layout()
fig.savefig(snakemake.output[0])

##################################
# Delta p FIRST SURFACE vs res p #
##################################

dfgsf = summary_gsf[ abs(summary_gsf.t_delta_p_first_surface) > 0.01 ].copy()
dfkf = summary_kf[ abs(summary_kf.t_delta_p_first_surface) > 0.01 ].copy()

print(len(dfgsf))
print(len(dfkf))
# assert len(dfgsf) == len(dfkf)

fig2, axes2 = plot(
    dfgsf,
    dfkf,
    lambda df: df.res_eQOP_fit[ df.res_eQOP_fit.between(*clip) ],
    lambda df: df.t_delta_p[ df.res_eQOP_fit.between(*clip) ],
    # lambda df: df.res_eP_fit[ df.res_eP_fit.between(*clip) ],
    # lambda df: df.t_delta_p[ df.res_eP_fit.between(*clip) ],
)

fig2.suptitle("$\Delta E$ on first surface vs. $res_p$")

for ax in axes2:
    ax.set_ylabel("$\Delta E$ (after first surface) $\quad [GeV]$")
    ax.set_xlim(*clip)
    ax.set_ylim(-5, 0)

fig2.tight_layout()
fig2.savefig(snakemake.output[1])

if True or snakemake.config["plt_show"]:
    plt.show()
