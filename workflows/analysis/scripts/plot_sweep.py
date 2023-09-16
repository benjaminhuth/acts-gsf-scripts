import pandas as pd
import matplotlib.pyplot as plt

from gsfanalysis.parallel_coordinates import parallel_coordinates


sweep_result = pd.read_csv(snakemake.input[0])
sweep_result["timing_ms"] = 1e3 * sweep_result["timing"]
sweep_result["outlier_ratio"] = sweep_result["n_outliers"] / sweep_result["n_tracks"]
sweep_result = sweep_result.rename(
    columns={
        c: c.replace("symmetric_", "") for c in sweep_result.columns if "symmetric" in c
    }
)

columns = [
    "timing_ms",
    "outlier_ratio",
    "res_eQOP_mode",
    "res_eQOP_q68",
    # "res_eQOP_rms",
    "res_ePNORM_mode",
    "res_ePNORM_q68",
    # "res_ePNORM_q95",
    # "res_ePNORM_rms"
]


def make_coordinates_plot(first_col, df):
    value_columns = [first_col] + columns
    error_columns = [c + "_err" for c in value_columns if c + "_err" in df.columns]

    fig, ax = parallel_coordinates(
        df.loc[:, value_columns],
        df.loc[:, error_columns],
        "_err",
        jitter_x=True,
        log_columns=["weight_cutoff"],
        lw=3,
        cmap="plasma",
        figsize=(10, 5),
    )

    return fig, ax


# Components
sweep_components = (
    sweep_result[
        (sweep_result["weight_cutoff"] == 1e-6)
        & (sweep_result["components"].isin([1, 2, 4, 8, 12, 24, 32]))
    ]
    .copy()
    .sort_values(by=["components"], ascending=False)
)

fig1, ax = make_coordinates_plot("components", sweep_components)
fig1.suptitle("Different number of components with fixed weight cutoff {}".format(1e-6))
fig1.tight_layout()
fig1.savefig(snakemake.output[0])

# Weight cutoffs
sweep_cutoff = (
    sweep_result[sweep_result["components"] == 12]
    .copy()
    .sort_values(by=["weight_cutoff"], ascending=False)
)

fig2, ax = make_coordinates_plot("weight_cutoff", sweep_cutoff)
fig2.suptitle("Different weight cutoffs with fixed component number {}".format(12))
fig2.tight_layout()
fig2.savefig(snakemake.output[1])

if snakemake.config["plt_show"]:
    plt.show()
