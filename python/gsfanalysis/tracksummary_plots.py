import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .core_tail_utils import rms
from .statistics import mode


def correlation_plot(df, fig_ax=None, absolute=True):
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()

    corrcoefs = np.corrcoef(df.astype(float).to_numpy().T)
    if absolute:
        corrcoefs = abs(corrcoefs)

    mask = np.logical_not(np.tri(corrcoefs.shape[0], k=0))
    corrcoefs = np.ma.array(corrcoefs, mask=mask)

    im = ax.imshow(
        corrcoefs,
        origin="lower",
        aspect=0.3,
        vmin=0 if absolute else -1,
        vmax=1,
    )
    fig.colorbar(im, ax=ax, label="Correlation coefficient")

    keys = df.columns.tolist()
    ticks = np.arange(len(keys))

    ax.xaxis.tick_top()

    ax.set_xticks(ticks, keys, rotation=-45, ha="right")
    ax.set_yticks(ticks, keys)

    ax.set_xticks(ticks - 0.5, minor=True)
    ax.set_yticks(ticks - 0.5, minor=True)

    ax.set_aspect(0.7)

    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    return fig, ax


def ratio_hist(ax, df, bins, label, clip=(0, 2)):
    clipped_ratio = np.clip(df["p_fit"] / df["t_p"], *clip)

    # find mode
    np_n, np_bins = np.histogram(clipped_ratio, bins=bins)
    max_idx = np.argmax(np_n)
    mode = 0.5 * (np_bins[max_idx] + np_bins[max_idx + 1])

    # draw hist
    n, bins, _ = ax.hist(
        clipped_ratio,
        bins=bins,
        alpha=0.5,
        label="{} (mean={:.3f}, mode={:.3f})".format(
            label, np.mean(clipped_ratio), mode
        ),
    )

    mids = 0.5 * (bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    std = np.average((mids - mean) ** 2, weights=n)

    # print("\tHist {}: {:.3f} +- {:.3f}".format(label, mean, std))

    return bins


########################
# p_fit / p_true ratio #
########################


def ratio_residual_plot(summary_gsf, summary_kf, log_scale=False, bins=200):
    fig, ax = plt.subplots(1, 2)

    # Ratio hist
    b = ratio_hist(ax[0], summary_gsf, bins, "GSF")
    ratio_hist(ax[0], summary_kf, b, "KF")

    ax[0].set_title("Ratio")
    ax[0].set_xlabel("$p_{fit} / p_{true}$")
    ax[0].legend()

    # Residual hist
    clip = (-3, 3)
    _, b, _ = ax[1].hist(
        np.clip(summary_gsf["res_p_fit"], *clip),
        bins=bins,
        alpha=0.5,
        label="GSF",
    )
    ax[1].hist(np.clip(summary_kf["res_p_fit"], *clip), bins=b, alpha=0.5, label="KF")

    ax[1].set_title("Residual")
    ax[1].set_xlabel("Residual  $p_{fit}$ - $p_{true}$")
    ax[1].legend()

    if log_scale:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

    return fig, ax


######################
# Correlation matrix #
######################


def correlation_scatter_plot(summary, clip_res, do_printout=False):

    keys = [
        "chi2Sum",
        "t_theta",
        "t_phi",
        "t_p",
        "res_p_fit",
        "t_delta_p",
        "t_delta_p_first_surface",
    ]

    event = summary["event_nr"].to_numpy()
    traj = summary["multiTraj_nr"].to_numpy()

    data = summary[keys].to_numpy().T

    def plot_mat_and_scatter(fig, ax, k_delta_p, k_res_p):
        correlation_plot(summary[keys], (fig, ax[0]))

        ax[1].scatter(
            np.clip(data[k_res_p], clip_res[0], clip_res[1]),
            data[k_delta_p],
            alpha=0.5,
        )
        ax[1].set_xlabel(keys[k_res_p])
        ax[1].set_ylabel(keys[k_delta_p])
        ax[0].set_title(keys[k_delta_p])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    k_res_p_fit = 4
    k_true_p_delta = 5
    k_true_p_delta_first_surface = 6

    plot_mat_and_scatter(fig, axes[:, 0], k_true_p_delta, k_res_p_fit)
    plot_mat_and_scatter(fig, axes[:, 1], k_true_p_delta_first_surface, k_res_p_fit)

    def print_idx_tuples(label, idxs):
        strs = [
            "(e={}, t={}, r={:.2f}, l={:.2f})".format(
                event[i], traj[i], data[k_res_p_fit, i], data[k_true_p_delta, i]
            )
            for i in idxs
        ]
        if len(strs) > 0:
            print(
                "{} ({:.1f}%)\n\t".format(label, 100.0 * len(idxs) / len(event)),
                ", ".join(strs),
            )
        else:
            print("no samples")

    if do_printout:
        # Energy increase
        idxs_increase = np.nonzero(data[k_res_p_fit] > 0.5)[0]
        idxs_decrease = np.nonzero(
            np.logical_and(data[k_res_p_fit] < -0.5, data[k_true_p_delta] >= -0.5)
        )[0]
        idxs_loss = np.nonzero(
            np.logical_and(data[k_res_p_fit] < -0.5, data[k_true_p_delta] < -0.5)
        )[0]

        print_idx_tuples("Energy decrease without loss", idxs_decrease)
        print_idx_tuples("Energy increase", idxs_increase)
        print_idx_tuples("Wrong energy loss", idxs_loss)

    return fig, axes


def make_full_residual_plot(dfs, labels, log=True, clip_std=None, clip_quantile=None):
    assert len(dfs) == len(labels)
    assert clip_std is None or clip_quantile is None

    fig, axes = plt.subplots(2, 4, figsize=(18, 5))

    res_keys = [
        "res_eLOC0_fit",
        "res_eLOC1_fit",
        "res_ePHI_fit",
        "res_eTHETA_fit",
        "res_eQOP_fit",
        "res_eT_fit",
        "res_eP_fit",
        "res_ePNORM_fit",
    ]
    coor_names = ["d_0", "z", "\phi", "\\theta", "q/p", "t", "p", "p norm"]
    units = ["mm", "mm", "rad", "rad", "GeV^{-1}", "ns", "GeV", ""]

    for ax, key, name, unit in zip(axes.flatten(), res_keys, coor_names, units):
        values = np.concatenate([df[key] for df in dfs])

        if clip_std is not None:
            mu, std = np.mean(values), np.std(values)
            values = np.clip(values, mu - clip_std * std, mu + clip_std * std)

        if clip_quantile is not None:
            m = mode(values)
            q = np.quantile(abs(values - m), clip_quantile)
            lo = min(values[(values - m) > -q])
            hi = max(values[(values - m) < q])
            values = np.clip(values, lo, hi)

        _, bins = np.histogram(values, bins="rice")

        hist_opts = dict(
            alpha=0.5, histtype="stepfilled", linewidth=1.2, edgecolor="black"
        )

        for df, fitter in zip(dfs, labels):
            ax.hist(
                np.clip(df[key], min(bins), max(bins)),
                bins=bins,
                **hist_opts,
                label=fitter
            )

        if log:
            ax.set_yscale("log")

        ax.set_title("${}$".format(name))
        ax.set_xlabel(
            "${{{}}}_{{fit}} - {{{}}}_{{true}} \quad [{}]$".format(name, name, unit)
        )

    axes.flatten()[-1].set_xlabel("$(p_{fit} - p_{true}) / p_{true}$")

    axes[0, 0].legend()
    return fig, axes


def print_basic_statistics(dfs, names):
    assert len(dfs) == len(names)
    data = {}

    def append_to_data(key, value):
        if not key in data:
            data[key] = []
        data[key].append(value)

    def make_summary_frame(df, index):
        append_to_data("index", index)

        # Bad tracks
        bad = df[
            ~df["res_eLOC0_fit"].between(-100, 100)
            | ~df["res_eLOC1_fit"].between(-100, 100)
        ]
        append_to_data("bad loc", len(bad) / len(df) if len(df) > 0 else 0.0)

        outliers = df[df["nOutliers"] > 0]
        append_to_data("outlier ratio", len(outliers) / len(df) if len(df) > 0 else 0.0)

        append_to_data("avg states", np.mean(df["nStates"]))
        append_to_data("avg outliers", np.mean(df["nOutliers"]))
        append_to_data("avg measuremnts", np.mean(df["nMeasurements"]))

        res_qop = df["res_eQOP_fit"]
        pull_qop = df["pull_eQOP_fit"]
        append_to_data("res QOP mean", np.mean(res_qop))
        append_to_data("res QOP mode", mode(res_qop))
        append_to_data("res QOP rms", rms(res_qop))
        append_to_data("pull QOP mean", np.mean(pull_qop))
        append_to_data("pull QOP mode", mode(pull_qop))
        append_to_data("pull QOP std", np.std(pull_qop))

    for df, name in zip(dfs, names):
        make_summary_frame(df, name)

    with pd.option_context("display.float_format", "{:0.3f}".format):
        print(pd.DataFrame(data).set_index("index").transpose())


def print_worst(tracksummary):
    keys = ["res_eLOC0_fit", "res_eLOC1_fit", "res_eQOP_fit"]

    for key in keys:
        idxs = tracksummary[key].argsort()
        print(tracksummary.loc[idxs[:5], ["event_nr", "multiTraj_nr", key]])
        print()

    print(
        tracksummary[tracksummary["event_nr"] == 0][
            ["event_nr", "multiTraj_nr", "res_eQOP_fit"]
        ]
    )
