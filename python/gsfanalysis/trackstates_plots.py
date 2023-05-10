import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def make_x_value(tx, ty, tz, main_direction):
    if main_direction == "x":
        return tx
    elif main_direction == "y":
        return ty
    elif main_direction == "z":
        return tz
    elif main_direction == "r":
        return np.hypot(tx, ty)
    else:
        raise "error"


###############################
# nMeasurements, nHoles, etc. #
###############################


def plot_measurements_holes(trackstates):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].bar(
        *np.unique(trackstates["nMeasurements"].array(library="np"), return_counts=True)
    )
    ax[0].set_title("nMeasurements")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax[0].set_xticks(np.linspace(0,10,11))

    ax[1].bar(
        *np.unique(
            trackstates["nStates"].array(library="np")
            - trackstates["nMeasurements"].array(library="np"),
            return_counts=True,
        )
    )
    ax[1].set_title("nHoles")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax[1].set_xticks(np.linspace(0,10,11))

    ax[2].bar(
        *np.unique(
            [float(min(track)) for track in get(["t_x"])[0]], return_counts=True
        ),
        width=10
    )
    # ax[2].set_xticks(np.linspace(50,500,10))
    ax[2].set_title("x of first hit")
    ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig, ax


def single_particle_momentumplot(
    tracksummary, trackstates, label_predicted, label_filtered
):
    fig, ax = plt.subplots()

    # ????????
    x = trackstates["pathLength"].to_numpy()
    assert len(x) == 1
    x = x[0]

    p_true_states = 1.0 / abs(trackstates["t_eQOP"][0])
    p_prt_states = 1.0 / abs(trackstates["eQOP_prt"][0])
    p_flt_states = 1.0 / abs(trackstates["eQOP_flt"][0])

    ax.plot(x, p_true_states, label="true p")
    ax.plot(x, p_prt_states, label=label_predicted)
    ax.plot(x, p_flt_states, label=label_filtered, ls="--")

    p_true = float(tracksummary["t_p"][0])
    p_fit = 1.0 / abs(float(tracksummary["eQOP_fit"][0]))

    ax.scatter([0.0], [p_true], label="true p (origin)")
    ax.scatter([0.0], [p_fit], label="fitted p")

    ax.legend()

    return fig, ax


####################################################
# Plots at track positions (e.g. beginning or end) #
####################################################


def plot_at_track_position(
    trk_idx,
    trackstates,
    fitter_name,
    main_direction,
    clip_ratio=(0, 2),
    clip_abs=(0, 20),
    bins=100,
    log=True,
):
    aggregation = trackstates.groupby(["event_nr", "multiTraj_nr"]).agg(
        {
            "t_x": lambda s: s.iloc[trk_idx],
            "t_y": lambda s: s.iloc[trk_idx],
            "t_z": lambda s: s.iloc[trk_idx],
            "t_eQOP": lambda s: s.iloc[trk_idx],
            "eQOP_flt": lambda s: s.iloc[trk_idx],
            "eQOP_smt": lambda s: s.iloc[trk_idx],
        }
    )

    aggregation["t_p"] = abs(1.0 / aggregation["t_eQOP"])
    aggregation["p_smt"] = abs(1.0 / aggregation["eQOP_smt"])
    aggregation["p_flt"] = abs(1.0 / aggregation["eQOP_flt"])

    if clip_abs:
        aggregation["p_smt"] = np.clip(aggregation["p_smt"], clip_abs[0], clip_abs[1])
        aggregation["p_flt"] = np.clip(aggregation["p_flt"], clip_abs[0], clip_abs[1])

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("At track position '{}'".format(trk_idx))

    _, b, _ = ax[0].hist(
        aggregation["p_flt"],
        bins=bins,
        alpha=0.5,
        label="{} flt".format(fitter_name),
    )
    _ = ax[0].hist(aggregation["t_p"], bins=b, alpha=0.5, label="true")
    ax[0].set_title("filtered (forward) vs. truth")
    if log:
        ax[0].set_yscale("log")
    ax[0].legend()

    _, b, _ = ax[1].hist(
        aggregation["p_smt"],
        bins=bins,
        alpha=0.5,
        label="{} flt".format(fitter_name),
    )
    _ = ax[1].hist(aggregation["t_p"], bins=b, alpha=0.5, label="true")
    ax[1].set_title("smoothed (smoothed) vs. truth")
    if log:
        ax[1].set_yscale("log")
    ax[1].legend()

    # ratio = np.array(aggregation["p_flt"]) / np.array(aggregation["t_p"])
    # if clip_ratio:
    #     ratio = np.clip(ratio, clip_ratio[0], clip_ratio[1])
    # _ = ax[0, 1].hist(ratio, bins=bins)
    # ax[0, 1].set_title("ratio flt / true")
    #
    # ratio = np.array(aggregation["p_smt"]) / np.array(aggregation["t_p"])
    # if clip_ratio:
    #     ratio = np.clip(ratio, clip_ratio[0], clip_ratio[1])
    # _ = ax[1, 1].hist(ratio, bins=bins)
    # ax[1, 1].set_title("ration smt / true")

    get_pos = {
        "x": lambda df: df["t_x"],
        "y": lambda df: df["t_y"],
        "z": lambda df: df["t_z"],
        "r": lambda df: np.hypot(df["t_x"], df["t_y"]),
    }

    all_pos = get_pos[main_direction](trackstates)

    ax[2].hist(get_pos[main_direction](aggregation), range=(min(all_pos), max(all_pos)))
    ax[2].set_title("{}-position at track index {}".format(main_direction, trk_idx))
    if log:
        ax[2].set_yscale("log")

    return fig, ax


##########################################################
# ratio true/predicted and true/filtered at track states #
##########################################################


def performance_at_trackstates(
    trackstates, main_direction, clip=(0, 8), log_scale=False
):
    a = SimpleTrackstatesGetter(trackstates)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    xs = []
    ratios_flt = []
    ratios_prt = []

    for tx, ty, tz, track_true_qop, track_prt_qop, track_flt_qop in zip(
        *a.get(["t_x", "t_y", "t_z", "t_eQOP", "eQOP_prt", "eQOP_flt"])
    ):
        xvalues = make_x_value(tx, ty, tz, main_direction)

        for x, t_qop, prt_qop, flt_qop in zip(
            xvalues, track_true_qop, track_prt_qop, track_flt_qop
        ):
            xs.append(x)
            ratios_flt.append(np.clip(float(flt_qop / t_qop), clip[0], clip[1]))
            ratios_prt.append(np.clip(float(prt_qop / t_qop), clip[0], clip[1]))

    ax[0].axhline(y=1, color="grey", linestyle="-")
    ax[0].scatter(xs, ratios_prt, alpha=0.05)
    ax[0].set_title("Forward (predicted) p/p_true")
    ax[1].axhline(y=1, color="grey", linestyle="-")
    ax[1].scatter(-1 * np.array(xs), ratios_flt, alpha=0.05)
    ax[1].set_title("Backward (filtered) p/p_true")

    if log_scale:
        ax[1].set_yscale("log")
        ax[0].set_yscale("log")

    return fig, ax


def single_particle_momentumplot(
    tracksummary, trackstates, label_predicted, label_filtered
):
    fig, ax = plt.subplots()

    # ????????
    x = trackstates["pathLength"].to_numpy()
    assert len(x) == 1
    x = x[0]

    p_true_states = 1.0 / abs(trackstates["t_eQOP"][0])
    p_prt_states = 1.0 / abs(trackstates["eQOP_prt"][0])
    p_flt_states = 1.0 / abs(trackstates["eQOP_flt"][0])

    ax.plot(x, p_true_states, label="true p")
    ax.plot(x, p_prt_states, label=label_predicted)
    ax.plot(x, p_flt_states, label=label_filtered, ls="--")

    p_true = float(tracksummary["t_p"][0])
    p_fit = 1.0 / abs(float(tracksummary["eQOP_fit"][0]))

    ax.scatter([0.0], [p_true], label="true p (origin)")
    ax.scatter([0.0], [p_fit], label="fitted p")

    ax.legend()

    return fig, ax
