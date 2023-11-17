import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from gsfanalysis.parallel_coordinates import parallel_coordinates

n_particles = 10000  # TODO make this more generic


sweep_result = pd.read_csv(snakemake.input[0])

# fmt: off
sweep_result["timing [ms]"] = 1e3 * sweep_result["timing"] / sweep_result["n_tracks"]
sweep_result["outliers [%]"] = 100.0 * sweep_result["n_outliers"] / sweep_result["n_tracks"]
sweep_result["failures [‰]"] = 1000.0 * (n_particles - sweep_result["n_tracks"]) / n_particles
sweep_result["holes [%]"] = 100.0 * sweep_result["n_holes"] / n_particles
# fmt: on

sweep_result = sweep_result.rename(
    columns={
        c: c.replace("symmetric_", "") for c in sweep_result.columns if "symmetric" in c
    }
)

# Go to 1/MeV here
for key in sweep_result.keys():
    if "res_eQOP" in key:
        sweep_result[key] *= 1000.0

sweep_result = sweep_result.rename(
    columns={
        "res_eQOP_mode": "res q/p mode",
        "res_eQOP_q68": "res q/p Q68",
        "res_eQOP_q95": "res q/p Q95",
        "res_eQOP_mode_err": "res q/p mode_err",
        "res_eQOP_q68_err": "res q/p Q68_err",
        "res_eQOP_q95_err": "res q/p Q95_err",
        "weight_cutoff": "weight cutoff",
        "component_merge_method": "component merging",
        "bethe_heitler_approx": "BH Approx",
        "mixture_reduction": "mixture reduction"
    }
)

# Drop some components to not have to much
sweep_result = sweep_result[ sweep_result["components"].isin([1,2,4,8,12,20,32]) ].copy()

columns = [
    "timing [ms]",
    "failures [‰]",
    "outliers [%]",
    "holes [%]",
    "res q/p mode",
    "res q/p Q95",
]

# The default configuration that we vary each
fix = {
    "components": 12,
    "weight cutoff": 1e-6,
    "component merging": "maxWeight",
    "mixture reduction": "KLDistance",
    "BH Approx": "GeantSim_CDF",
}


def make_coordinates_plot(first_col : str, cmap="plasma", figsize=(10, 5)):
    fix_parameters = { k: v for k, v in fix.items() if k != first_col }
    
    conditions = np.ones(len(sweep_result)).astype(bool)
    for key, val in fix_parameters.items():
        conditions = conditions & (sweep_result[key] == val)
    
    sweep_df = sweep_result[conditions].copy().sort_values([first_col], ascending=False).drop_duplicates(first_col)
    
    print(first_col.upper())
    print(sweep_df)
    print()
    
    value_columns = [first_col] + columns
    error_columns = [c + "_err" for c in value_columns if c + "_err" in sweep_df.columns]

    fig, ax = parallel_coordinates(
        sweep_df.loc[:, value_columns],
        sweep_df.loc[:, error_columns],
        "_err",
        jitter_x=True,
        log_columns=["weight cutoff"],
        lw=3,
        cmap=cmap,
        figsize=figsize,
    )

    return fig, ax


# Components
fig1, ax = make_coordinates_plot("components")
fig1.suptitle("Performance metrics for varying component number")
fig1.tight_layout()
fig1.savefig(snakemake.output[0])

# Weight cutoff
fig2, ax = make_coordinates_plot("weight cutoff")
fig2.suptitle("Performance metrics for varying weight cutoffs")
fig2.tight_layout()
fig2.savefig(snakemake.output[1])


# Merging methods
cmap = ListedColormap(matplotlib.colormaps["plasma"](np.linspace(0.4, 0.8, 12)))
fig3, ax = make_coordinates_plot(
    "component merging", figsize=(10, 3), cmap=cmap
)
fig3.suptitle("Performance metrics for varying component merge methods")
fig3.tight_layout()
fig3.savefig(snakemake.output[2])

# Reduction
fig4, ax = make_coordinates_plot("mixture reduction", figsize=(10, 3), cmap=cmap)
fig4.suptitle(f"Performance metrics for varying mixture reduction algorithms")
fig4.tight_layout()
fig4.savefig(snakemake.output[3])

# BHA
fig4, ax = make_coordinates_plot("BH Approx", figsize=(10, 3), cmap=cmap)
fig4.suptitle(f"Performance metrics for varying mixture reduction algorithms")
fig4.tight_layout()
fig4.savefig(snakemake.output[4])




# Make nice summary figure
kMode = "res q/p mode"
kQ95 = "res q/p Q95"

fig, ax = plt.subplots(figsize=(10,6))
ax.errorbar(sweep_result[kMode], sweep_result[kQ95],
            xerr=sweep_result[kMode + "_err"],
            yerr=sweep_result[kQ95 + "_err"],
            fmt="none", elinewidth=1,
            color="grey", capsize=2.0)
ax.set_xlabel("mode of res q/p [$MeV^{-1}$]")
ax.set_ylabel("Q95 of res q/p [$MeV^{-1}$]")
ax.set_ylim(12,20)
ax.set_xlim(-0.2,0.7)
    
chosen = np.ones(len(sweep_result)).astype(bool)
for key, val in fix.items():
    chosen = chosen & (sweep_result[key] == val)
chosen = sweep_result[ chosen ]
assert len(chosen) == 1
chosen = chosen.iloc[0]

lmax = max([len(k) for k in fix.keys()])
ax.errorbar([ chosen[kMode] ], [ chosen[kQ95] ],
            xerr=[ chosen[kMode + "_err"] ],
            yerr=[ chosen[kQ95 + "_err"] ],
            fmt="none", elinewidth=2,
            color="red", capsize=2.0, label="\n".join([f"{k}: {v}" for k, v in fix.items()]))
ax.legend(loc="upper right")
ax.set_title("GSF performance metrics for different configurations")
fig.tight_layout()
fig.savefig(snakemake.output[5])


# maybe show
if snakemake.config["plt_show"]:
    plt.show()
