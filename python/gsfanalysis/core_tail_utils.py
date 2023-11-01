import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps

from .statistics import rms


def add_core_to_df_quantile(df, key, quantile=0.95, core_label="is_core"):
    d = np.quantile(np.sort(abs(df[key])), quantile)
    core_range = (-d, d)

    df[core_label] = df[key].between(*core_range)

    return df


def plot_core_tail_residuals(
    df, key, parts=["core", "tail", "all"], coor_unit=None, fig_ax=None, bins="rice"
):
    fig, ax = plt.subplots() if fig_ax is None else fig_ax

    part_dfs, labels, colors = [], [], []
    if "core" in parts:
        core_part = df[df["is_core"]]
        part_dfs.append(core_part[key])
        labels.append(
            "core: $\mu$={:.2f}, rms={:.2f}".format(
                np.mean(core_part[key]), rms(core_part[key])
            )
        )
        colors.append("tab:orange")

    if "tail" in parts:
        tail_part = df[~df["is_core"]]
        part_dfs.append(tail_part[key])
        labels.append(
            "tail: $\mu$={:.2f}, rms={:.2f}".format(
                np.mean(tail_part[key]), rms(tail_part[key])
            )
        )
        colors.append("tab:blue")

    assert len(part_dfs) > 0

    hist_options = dict(edgecolor="black", linewidth=1.2, stacked=True)

    _, bins, _ = ax.hist(
        part_dfs,
        bins=bins,
        label=labels,
        histtype="stepfilled",
        color=colors,
        **hist_options
    )

    if "all" in parts:
        all_label = "all: $\mu$={:.2f}, rms={:.2f}".format(
            np.mean(df[key]), rms(df[key])
        )
        ax.hist(df[key], bins=bins, histtype="step", label=all_label, **hist_options)

    ax.set_yscale("log")
    ax.legend()

    if not coor_unit is None:
        name, unit = coor_unit
        ax.set_xlabel(
            "${{{}}}_{{fit}} - {{{}}}_{{true}} \quad [{}]$".format(name, name, unit)
        )

    return fig, ax


def plot_core_tail_pulls(
    df, key, part="core", coor_unit=None, fig_ax=None, bins="rice"
):
    fig, ax = plt.subplots() if fig_ax is None else fig_ax

    if part == "core":
        part_df = df[df["is_core"]]
        color = "tab:orange"
    elif part == "tail":
        part_df = df[~df["is_core"]]
        color = "tab:blue"
    elif part == "all":
        part_df = df
        color = "grey"

    pull = part_df[key]
    pull = pull[np.isfinite(pull)]
    mu, sigma = sps.norm.fit(pull)
    skew = sps.skew(pull)

    ax.hist(np.clip(pull, -5 * sigma, 5 * sigma), bins=bins, density=True, color=color)

    x = np.linspace(*ax.get_xlim(), 200)
    ax.plot(
        x,
        sps.norm.pdf(x, mu, sigma),
        label="$\mu$={:.2f}, $\sigma$={:.2f}, $\\tilde{{\mu}}_3={:.2f}$".format(
            mu, sigma, skew
        ),
        color="red",
        lw=2.0,
    )

    ax.legend()

    if not coor_unit is None:
        name, unit = coor_unit
        ax.set_xlabel(
            "$({{{}}}_{{fit}} - {{{}}}_{{true}}) / \sigma_{{{},fit}} \quad [{}]$".format(
                name, name, name, unit
            )
        )

    return fig, ax
