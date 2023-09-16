import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gsfanalysis.statistics import mode


def make_gsf_detailed_comparison_plots(
    sets, logy=False, mean_height=50, assymetric_interval=False
):
    """
    A set is a tuple (df, label, color-string)
    """

    # TODO make assymetric
    def quantile_interval(data, q, midpoint=0):
        sorted_data = np.sort(data - midpoint)
        q = np.quantile(abs(sorted_data), q)
        interval = (
            midpoint + sorted_data[sorted_data > -q][0],
            midpoint + sorted_data[sorted_data < q][-1],
        )
        return 0.5 * (interval[1] - interval[0]), interval

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    keys = ["res_eQOP_fit", "res_ePNORM_fit", "pull_eQOP_fit"]
    base_ranges = [(-0.05, 0.05), (-0.3, 0.3), (-2, 2)]

    # First evaluate ranges, so that we have sane binning afterwards for the hist
    for i, key in enumerate(keys):
        for df, _, _ in sets:
            m = mode(df[key])

            if assymetric_interval:
                _, interval = quantile_interval(df[key], 0.95, m)
            else:
                q = np.quantile(abs(df[key] - m), 0.95)
                interval = (m - q, m + q)

            base_ranges[i] = (
                min([base_ranges[i][0], interval[0], np.mean(df[key])]),
                max([base_ranges[i][1], interval[1], np.mean(df[key])]),
            )

    for ax, key, clip_range in zip(axes, keys, base_ranges):
        for df, label, color in sets:
            histopts = dict(bins=100, alpha=0.3, color=color, zorder=-10)

            sample_mode = mode(df[key])

            if assymetric_interval:
                q95, q95_interval = quantile_interval(df[key], 0.95, sample_mode)
            else:
                q95 = np.quantile(abs(df[key] - sample_mode), 0.95)
                q95_interval = (sample_mode - q95, sample_mode + q95)

            if assymetric_interval:
                q68, q68_interval = quantile_interval(df[key], 0.68, sample_mode)
            else:
                q68 = np.quantile(abs(df[key] - sample_mode), 0.68)
                q68_interval = (sample_mode - q68, sample_mode + q68)

            hist, _, _ = ax.hist(
                np.clip(df[key], *clip_range),
                # label="{} - rms: {:.3f}".format(label, rms(df[key])),
                label=label,
                range=clip_range,
                **histopts
            )

            ax.vlines(
                [sample_mode],
                ymin=0,
                ymax=0.66 * max(hist),
                color=color,
                label="mode: {:.3f}".format(sample_mode),
                ls="--",
                zorder=-3,
                lw=1,
            )

            ax.plot(
                q95_interval,
                2 * [0.33 * max(hist)],
                "-o",
                color=color,
                # label="Q95: {:.3f}\n   rms: {:.3f}".format(q95, rms(df[key].between(*q95_interval))))
                label="Q95: {:.3f}".format(q95),
            )
            ax.plot(
                q68_interval,
                2 * [0.66 * max(hist)],
                "-s",
                color=color,
                # label="Q68: {:.3f}\n   rms: {:.3f}".format(q68, rms(df[key].between(*q68_interval))))
                label="Q68: {:.3f}".format(q68),
            )

            sample_mean = np.mean(df[key])
            ax.vlines(
                [sample_mean],
                ymin=0,
                ymax=mean_height,
                color=color,
                label="mean: {:.3f}".format(sample_mean),
                zorder=-3,
                lw=3,
            )

    if logy:
        for ax in axes:
            ax.set_yscale("log")

    legend_opts = dict(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    axes[0].set_title("res QOP")
    axes[0].legend(**legend_opts)
    axes[0].set_xlabel("${qop}_{fit} - {qop}_{true} \quad [GeV^{-1}]$")

    axes[1].set_title("res PNORM")
    axes[1].legend(**legend_opts)
    axes[1].set_xlabel("$({p}_{fit} - {p}_{true}) \;/\; p_{true}$")

    axes[2].set_title("pull QOP")
    axes[2].legend(**legend_opts)
    axes[2].set_xlabel("$({qop}_{fit} - {qop}_{true}) \;/\; \sigma_{qop,fit}$")

    return fig, ax
