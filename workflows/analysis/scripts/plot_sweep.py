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
        "component_merge_method": "component merging"
    }
)

columns = [
    "timing [ms]",
    "failures [‰]",
    "outliers [%]",
    "holes [%]",
    "res q/p mode",
    "res q/p Q95",
]


def make_coordinates_plot(first_col, df, cmap="plasma", figsize=(10, 5)):
    value_columns = [first_col] + columns
    error_columns = [c + "_err" for c in value_columns if c + "_err" in df.columns]

    fig, ax = parallel_coordinates(
        df.loc[:, value_columns],
        df.loc[:, error_columns],
        "_err",
        jitter_x=True,
        log_columns=["weight cutoff"],
        lw=3,
        cmap=cmap,
        figsize=figsize,
    )

    return fig, ax


# Components
sweep_components = (
    sweep_result[
        (sweep_result["weight cutoff"] == 1e-4)
        & (sweep_result["component merging"] == "maxWeight")
        & (sweep_result["mixture_reduction"] == "KLDistance")
        & (sweep_result["components"].isin([1, 2, 4, 8, 12, 24, 32]))
    ]
    .copy()
    .sort_values(by=["components"], ascending=False)
)

fig1, ax = make_coordinates_plot("components", sweep_components)
fig1.suptitle("Different number of components with fixed weight cutoff {}".format(1e-4))
fig1.tight_layout()
fig1.savefig(snakemake.output[0])

# Weight cutoffs
sweep_cutoff = (
    sweep_result[
        (sweep_result["components"] == 12)
        & (sweep_result["mixture_reduction"] == "KLDistance")
        & (sweep_result["component merging"] == "maxWeight")
    ]
    .copy()
    .sort_values(by=["weight cutoff"], ascending=False)
)
print("Weight cutoff")
print(sweep_cutoff)
print()

fig2, ax = make_coordinates_plot("weight cutoff", sweep_cutoff)
fig2.suptitle("Different weight cutoffs with fixed component number {}".format(12))
fig2.tight_layout()
fig2.savefig(snakemake.output[1])

# Reduction methods
sweep_merge_meth = (
    sweep_result[
        (sweep_result["weight cutoff"] == 1e-4) & (sweep_result["components"] == 12)
        & (sweep_result["mixture_reduction"] == "KLDistance")
    ]
    .copy().drop_duplicates(["component merging"])
)
print("Component Merge method")
print(sweep_merge_meth)
print()
# no mode
assert len(sweep_merge_meth) == 2

cmap = ListedColormap(matplotlib.colormaps["plasma"](np.linspace(0.4, 0.8, 12)))
fig3, ax = make_coordinates_plot(
    "component merging", sweep_merge_meth, figsize=(10, 3), cmap=cmap
)
fig3.suptitle(f"Merge methods with {12} components and weight-cutoff {1e-4}")
fig3.tight_layout()
fig3.savefig(snakemake.output[2])

sweep_reduction_algorithm = (
    sweep_result[
        (sweep_result["weight cutoff"] == 1e-4) & (sweep_result["components"] == 12)
        & (sweep_result["component merging"] == "maxWeight")
    ]
    .copy().drop_duplicates(["mixture_reduction"])
)
sweep_reduction_algorithm = sweep_reduction_algorithm[ sweep_reduction_algorithm.mixture_reduction != "aggressiveKLDistance" ]
print("Mixture reduction method")
print(sweep_reduction_algorithm)
print()
fig4, ax = make_coordinates_plot(
    "mixture_reduction", sweep_reduction_algorithm, figsize=(10, 3), cmap=cmap
)
fig4.suptitle(f"Reduction algorithms with {12} components and weight-cutoff {1e-4}")
fig4.tight_layout()
fig4.savefig(snakemake.output[3])

# dfgsdfdfgsdfgsfsdf
if snakemake.config["plt_show"]:
    plt.show()
