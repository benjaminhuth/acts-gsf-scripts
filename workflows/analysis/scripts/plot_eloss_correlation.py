import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba, LogNorm

from gsfanalysis.pandas_import import *


gsf_cmap = LinearSegmentedColormap.from_list(
    "gsf", [to_rgba("tab:orange", alpha=0), to_rgba("tab:orange", alpha=1)]
)
kf_cmap = LinearSegmentedColormap.from_list(
    "kf", [to_rgba("tab:blue", alpha=0), to_rgba("tab:blue", alpha=1)]
)


def plot(dfgsf, dfkf, x_lambda, y_lambda, clip, vmax, bins):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    hist_args = {
        "bins": bins,
        "cmin": 1,
        "norm": LogNorm(vmin=1, vmax=vmax),
        "range": clip,
    }

    if True:
        gsf_plot = axes[0].hist2d(
            x_lambda(dfgsf), y_lambda(dfgsf), cmap=gsf_cmap, **hist_args
        )
        fig.colorbar(gsf_plot[3])

        kf_plot = axes[1].hist2d(
            x_lambda(dfkf), y_lambda(dfkf), cmap=kf_cmap, **hist_args
        )
        fig.colorbar(kf_plot[3])
    else:
        axes[0].scatter(x_lambda(dfgsf), y_lambda(dfgsf), color="tab:orange")

        axes[1].scatter(x_lambda(dfkf), y_lambda(dfkf), color="tab:blue")

    axes[1].set_title(f"KF")
    axes[0].set_title(f"GSF")

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


clip_dict = {
    "res_eQOP_fit": (-0.8,0.2),
    "res_eP_fit": (-1,10),
}
y_clip=(-10,0)

label_dict = {
    "res_eQOP_fit": "$p_{fit} - p_{true} \quad [GeV^{-1}]$",
    "res_eP_fit": "$q/p_{fit} - q/p_{true} \quad [GeV]$"
}

title_dict = {
    "res_eQOP_fit": "q/p residual",
    "res_eP_fit": "momentum residual"
}

for i, key in enumerate(["res_eQOP_fit", "res_eP_fit"]):
    clip = clip_dict[key]

    ####################
    # Delta p vs res p #
    ####################

    fig, axes = plot(
        summary_gsf.copy(),
        summary_kf.copy(),
        lambda df: df[key], #[ df[key].between(*clip) ],
        lambda df: df.t_delta_p, #[ df[key].between(*clip) ],
        clip=(clip, y_clip),
        vmax=1000,
        bins=[50,50],
    )

    for ax in axes:
        ax.set_xlabel(label_dict[key])


    fig.suptitle(f"$\Delta E$ vs. {title_dict[key]}")

    for ax in axes:
        ax.set_ylabel("$\Delta E \quad [GeV]$")
        ax.set_xlim(*clip)
        ax.set_ylim(*y_clip)

    fig.tight_layout()
    fig.savefig(snakemake.output[2*i])

    ##################################
    # Delta p FIRST SURFACE vs res p #
    ##################################

    dfgsf = summary_gsf[ abs(summary_gsf.t_delta_p_first_surface) > 0.1 ].copy()
    dfkf = summary_kf[ abs(summary_kf.t_delta_p_first_surface) > 0.1 ].copy()
    

    print(len(dfgsf))
    print(len(dfkf))

    fig2, axes2 = plot(
        dfgsf,
        dfkf,
        lambda df: df[key], #[ df[key].between(*clip) ],
        lambda df: df.t_delta_p_first_surface, #[ df[key].between(*clip) ],
        clip=(clip, y_clip),
        vmax=100,
        bins=[50,50],
    )

    for ax in axes2:
        ax.set_xlabel(label_dict[key])

    fig2.suptitle(f"$\Delta E$ on first surface vs. {title_dict[key]}")

    for ax in axes2:
        ax.set_ylabel("$\Delta E$ (after first surface) $\quad [GeV]$")
        ax.set_xlim(*clip)
        ax.set_ylim(*y_clip)

    fig2.tight_layout()
    fig2.savefig(snakemake.output[2*i+1])

if True or snakemake.config["plt_show"]:
    plt.show()
