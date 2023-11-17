import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


def parallel_coordinates(
    df,
    error_df=None,
    error_suffix="",
    lw=3,
    capsize=10,
    jitter_x=False,
    log_columns=[],
    cmap=None,
    figsize=None,
    enlarge=0.0,
):
    fig, lax = plt.subplots(figsize=figsize)

    ticklabels = {}

    for col in df.columns:
        if (df.dtypes[col] == str) or (df.dtypes[col] == object):
            assert not col in log_columns
            ticklabels[col] = list(df[col])
            df[col] = np.linspace(0, 1, len(df[col])).astype(float)

    if error_df is not None:
        for col in df.columns:
            if not col + error_suffix in error_df:
                error_df = error_df.copy()
                error_df[col + error_suffix] = np.zeros(len(df))

    class Scaler:
        def __init__(self, col, scale=True):
            self.col = col

            self.min = min(
                df[col]
                - (
                    error_df[col + error_suffix]
                    if error_df is not None
                    else np.zeros(len(df))
                )
            )
            self.max = max(
                df[col]
                + (
                    error_df[col + error_suffix]
                    if error_df is not None
                    else np.zeros(len(df))
                )
            )

            # We cannot have a zero range later, so set some arbitrary values
            if self.max == self.min:
                if self.max == 0:
                    self.max = 0.1
                    self.min = -0.1
                else:
                    self.min -= 0.1 * abs(self.min)
                    self.max += 0.1 * abs(self.max)
            elif col in log_columns:
                assert self.min > 0
                assert self.max > 0
                self.min /= 10
                self.max *= 10
            else:
                self.min -= 0.1 * (self.max - self.min)
                self.max += 0.1 * (self.max - self.min)

            assert (self.max - self.min) > 0

        def __call__(self, x):
            s = np.log10 if self.col in log_columns else lambda x: x
            return (s(x) - s(self.min)) / (s(self.max) - s(self.min))

        def only_scale(self, x):
            # s = np.log10 if self.col in log_columns else lambda x: x
            return x / (self.max - self.min)

    x_coors = np.linspace(0, 1, len(df.columns))
    axes = [lax.twinx() for _ in x_coors]
    scalers = [Scaler(col) for col in df.columns]

    for ax, x, s in zip(axes, x_coors, scalers):
        if s.col in log_columns:
            ax.set_yscale("log")

        ax.set_ylim(s.min * (1-enlarge), s.max * (1+enlarge))
        ax.spines["right"].set_position(("data", x))
        ax.spines["left"].set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_zorder(100)
        ax.yaxis.set_zorder(100)

        if s.col in ticklabels:
            ax.set_yticks(np.linspace(0, 1, len(df[s.col])))
            ax.set_yticklabels(ticklabels[s.col])

    lax.xaxis.tick_top()
    lax.tick_params(axis="x", which="both", length=0)
    lax.set_xticks(x_coors)
    lax.set_xticklabels(df.columns)
    lax.set_ylim(0, 1)
    lax.spines["top"].set_visible(False)
    lax.spines["bottom"].set_visible(False)
    lax.spines["right"].set_visible(False)
    lax.spines["left"].set_visible(False)
    lax.get_yaxis().set_visible(False)

    ##########################
    # Function for smoothing #
    ##########################
    def smooth(x, y):
        d = 0.33 * (1 / (len(x) - 1))
        x_help = (
            [x[0], x[0] + d]
            + sum([[xx - d, xx, xx + d] for xx in x[1:-1]], [])
            + [x[-1] - d, x[-1]]
        )
        y_help = [y[0], y[0]] + sum([3 * [yy] for yy in y[1:-1]], []) + [y[-1], y[-1]]

        spline = make_interp_spline(x_help, y_help, k=1)  # type: BSpline
        x_spline = np.linspace(min(x), max(x), 300)
        return x_spline, spline(x_spline)

    ###########################
    # Loop over the dataframe #
    ###########################
    x_offsets = np.linspace(-0.01, 0.01, len(df)) if jitter_x else np.zeros(len(df))
    colormap_vals = np.linspace(0, 1, len(df))

    if isinstance(cmap, str):
        cmap = plt.colormaps[cmap]

    for i, x_offset, cmap_idx in zip(df.index, x_offsets, colormap_vals):
        x = x_coors + x_offset
        y = [s(df.loc[i, col]) for s, col in zip(scalers, df.columns)]

        yerr = None
        if error_df is not None:
            yerr = [
                s.only_scale(error_df.loc[i, col + error_suffix])
                for s, col in zip(scalers, df.columns)
            ]
            yerr = [yerr.copy(), yerr.copy()]

            # TODO refactor this into the Scaler class
            for j, s in enumerate(scalers):
                if s.col in log_columns:
                    y_val, err_val = (
                        df.loc[i, s.col],
                        error_df.loc[i, s.col + error_suffix],
                    )

                    dlow = np.log10(y_val) - np.log10(y_val - err_val)
                    dhigh = np.log10(y_val + err_val) - np.log10(y_val)

                    dlow /= np.log10(s.max) - np.log10(s.min)
                    dhigh /= np.log10(s.max) - np.log10(s.min)

                    yerr[0][j] = dlow
                    yerr[1][j] = dhigh

        color = cmap(cmap_idx) if cmap is not None else None
        line = lax.plot(*smooth(x, y), lw=lw, zorder=i, c=color)[0]
        lax.errorbar(
            x=x,
            y=y,
            yerr=yerr,
            capsize=capsize,
            lw=lw,
            capthick=max(1, lw - 1),
            fmt=",",
            color=line._color,
            zorder=i,
        )

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame(
        {
            "timing": [100, 1000, 10000, 1, 1, 1, 1],
            "cmps": [32, 28, 24, 20, 16, 12, 8],
            "cmps_errs": [1, 1, 0.5, 0, 0, 0, 0],
            "wc": [-10, -12, -24, 0, 0, 0, 0],
            "wc_errs": [1, 1, 0.5, 0, 0, 0, 0],
            "timing_errs": [90, 900, 9000, 0, 0, 0, 0],
            "test": [0, 0.5, 1, 0, 0, 0, 0],
        }
    )

    fig, ax = parallel_coordinates(
        df[["cmps", "wc", "timing"]],
        error_df=df[["cmps_errs", "wc_errs", "timing_errs"]],
        error_suffix="_errs",
        jitter_x=True,
        log_columns=["timing"],
    )
    fig.suptitle("With log errors")
    fig.tight_layout()
    plt.show()
    exit()

    fig, ax = parallel_coordinates(
        df[df["cmps"] > 1],
        log_columns=["timing"],
    )
    fig.suptitle("Select rows with log")
    fig.tight_layout()
    plt.show()
    # exit()

    fig, ax = parallel_coordinates(df)
    fig.suptitle("Simple")
    fig.tight_layout()
    plt.show()
    # exit()

    fig, ax = parallel_coordinates(df, cmap="plasma")
    fig.suptitle("Simple with cmap")
    fig.tight_layout()
    plt.show()
    # exit()

    fig, ax = parallel_coordinates(
        df[["cmps", "wc", "timing", "test"]], log_columns=["timing"]
    )
    fig.suptitle("With log")
    fig.tight_layout()
    plt.show()

    fig, ax = parallel_coordinates(
        df[["cmps", "wc", "timing"]],
        error_df=df[["cmps_errs", "wc_errs", "timing_errs"]],
        error_suffix="_errs",
        jitter_x=True,
    )
    fig.suptitle("With errors")
    fig.tight_layout()
    plt.show()
