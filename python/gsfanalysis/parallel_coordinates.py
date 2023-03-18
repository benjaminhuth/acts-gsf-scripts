import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def parallel_coordinates(
    df, error_df=None, error_suffix="", lw=3, capsize=10, jitter_x=False, log_columns=[]
):
    fig, lax = plt.subplots(figsize=(18, 7))

    if error_df is not None:
        for col in df.columns:
            if not col + error_suffix in error_df:
                error_df = error_df.copy()
                error_df[col + error_suffix] = np.zeros(len(df))

    lmax = max(
        df.iloc[:, 0]
        + (error_df.iloc[:, 0] if error_df is not None else np.zeros(len(df)))
    )
    lmin = min(
        df.iloc[:, 0]
        - (error_df.iloc[:, 0] if error_df is not None else np.zeros(len(df)))
    )
    lmax += 0.1 * (lmax - lmin)
    lmin -= 0.1 * (lmax - lmin)

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

            if col in log_columns:
                assert self.min > 0
                assert self.max > 0
                self.min /= 10
                self.max *= 10
            else:
                self.min -= 0.1 * (self.max - self.min)
                self.max += 0.1 * (self.max - self.min)

        def __call__(self, x):
            s = np.log10 if self.col in log_columns else lambda x: x
            x_std = (s(x) - s(self.min)) / (s(self.max) - s(self.min))
            return x_std * (lmax - lmin) + lmin

        def only_scale(self, x):
            s = np.log10 if self.col in log_columns else lambda x: x
            x_std = s(x) / (s(self.max) - s(self.min))
            return x_std * (lmax - lmin)

    x_coors = np.linspace(0, 1, len(df.columns))
    axes = [lax] + [lax.twinx() for _ in x_coors[1:]]
    scalers = [Scaler(col) for col in df.columns]

    for ax, x, s in zip(axes[1:], x_coors[1:], scalers[1:]):
        if s.col in log_columns:
            ax.set_yscale("log")

        ax.set_ylim(s.min, s.max)
        ax.spines["right"].set_position(("data", x))

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_zorder(100)
        ax.yaxis.set_zorder(100)

    lax.spines["left"].set_position(("data", 0))
    lax.spines["right"].set_visible(False)
    lax.xaxis.tick_top()
    lax.tick_params(axis="x", which="both", length=0)
    lax.set_xticks(x_coors)
    lax.set_xticklabels(df.columns)
    lax.set_ylim(lmin, lmax)

    # for i in range(len(df)):
    #     print(i, [ s(df.loc[i, col]) for s, col in zip(scalers, df.columns) ])

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

    jitter = np.linspace(-0.01, 0.01, len(df)) if jitter_x else np.zeros(len(df))

    for i in range(len(df)):
        x = x_coors + jitter[i]
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

                    yerr[0][j] = dlow * (lmax - lmin)
                    yerr[1][j] = dhigh * (lmax - lmin)

        line = lax.plot(*smooth(x, y), lw=lw, zorder=i)[0]
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
            "cmps": [0, 2, 3],
            "cmps_errs": [1, 1, 0.5],
            "wc": [-10, -12, -24],
            "wc_errs": [1, 1, 0.5],
            "timing": [100, 1000, 10000],
            "timing_errs": [10, 100, 9000],
            "test": [0, 0.5, 1],
        }
    )

    parallel_coordinates(
        df[["test", "timing"]],
        log_columns=["timing"],
    )
    parallel_coordinates(df[["cmps", "wc", "timing", "test"]], log_columns=["timing"])
    parallel_coordinates(
        df[["cmps", "wc", "timing"]],
        error_df=df[["cmps_errs", "wc_errs", "timing_errs"]],
        error_suffix="_errs",
        jitter_x=True,
    )
    parallel_coordinates(
        df[["cmps", "wc", "timing"]],
        error_df=df[["cmps_errs", "wc_errs", "timing_errs"]],
        error_suffix="_errs",
        jitter_x=True,
        log_columns=["timing"],
    )

    plt.show()
