import argparse
from pathlib import Path

import numpy as np
import uproot
import matplotlib.pyplot as plt
import awkward as ak
from matplotlib.ticker import MaxNLocator

def make_x_value(tx, ty, tz, main_direction):
    if main_direction == 'x':
        return tx
    elif main_direction == 'y':
        return ty
    elif main_direction == 'z':
        return tz
    elif main_direction == 'r':
        return np.hypot(tx, ty)
    else:
        raise "error"

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
    
    im = ax.imshow(corrcoefs, origin="lower", aspect=0.3, vmin=0 if absolute else -1, vmax=1)
    fig.colorbar(im, ax=ax, label='Correlation coefficient')

    keys = df.columns.tolist()
    ticks = np.arange(len(keys))
    
    ax.xaxis.tick_top()
    
    ax.set_xticks(ticks, keys, rotation=-45, ha='right')
    ax.set_yticks(ticks, keys)    
    
    ax.set_xticks(ticks-0.5, minor=True)
    ax.set_yticks(ticks-0.5, minor=True)
    
    ax.set_aspect(0.7)
        
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    return fig, ax

def ratio_hist(ax, df, bins, label, clip = (0,2)):    
    clipped_ratio = np.clip(df["p_fit"] / df["t_p"], *clip)

    # find mode
    np_n, np_bins = np.histogram(clipped_ratio, bins=bins)
    max_idx = np.argmax(np_n)
    mode = 0.5*(np_bins[max_idx] + np_bins[max_idx+1])

    # draw hist
    n, bins, _ = ax.hist(clipped_ratio, bins=bins, alpha=0.5,
                         label="{} (mean={:.3f}, mode={:.3f})".format(label, np.mean(clipped_ratio), mode))

    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    std = np.average((mids - mean)**2, weights=n)

    print("\tHist {}: {:.3f} +- {:.3f}".format(label, mean, std))

    return bins

########################
# p_fit / p_true ratio #
########################

def ratio_residual_plot(summary_gsf, summary_kf, log_scale=False, bins=200):
    fig, ax = plt.subplots(1,2)

    # Ratio hist
    b = ratio_hist(ax[0], summary_gsf, bins, "GSF")
    ratio_hist(ax[0], summary_kf, b, "KF")

    ax[0].set_title("Ratio")
    ax[0].set_xlabel("$p_{fit} / p_{true}$")
    ax[0].legend()

    # Residual hist
    clip = (-3,3)
    _, b, _ = ax[1].hist(np.clip(summary_gsf["res_p_fit"], *clip), bins=bins, alpha=0.5, label="GSF")
    ax[1].hist(np.clip(summary_kf["res_p_fit"], *clip), bins=b, alpha=0.5, label="KF")

    ax[1].set_title("Residual")
    ax[1].set_xlabel("Residual  $p_{fit}$ - $p_{true}$")
    ax[1].legend()

    if log_scale:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')

    return fig, ax


###############################
# nMeasurements, nHoles, etc. #
###############################

def plot_measurements_holes(trackstates):
    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].bar(*np.unique(trackstates["nMeasurements"].array(library="np"), return_counts=True))
    ax[0].set_title("nMeasurements")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax[0].set_xticks(np.linspace(0,10,11))

    ax[1].bar(*np.unique(trackstates["nStates"].array(library="np") - trackstates["nMeasurements"].array(library="np"), return_counts=True))
    ax[1].set_title("nHoles")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax[1].set_xticks(np.linspace(0,10,11))

    ax[2].bar(*np.unique([ float(min(track)) for track in get(["t_x"])[0] ],return_counts=True), width=10)
    # ax[2].set_xticks(np.linspace(50,500,10))
    ax[2].set_title("x of first hit")
    ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig, ax


####################################################
# Plots at track positions (e.g. beginning or end) #
####################################################

def plot_at_track_position(trk_idx, trackstates, fitter_name, main_direction, clip_ratio=(0,2), clip_abs=(0,20), bins = 200):
    aggregation = trackstates.groupby(["event_nr", "multiTraj_nr"]).agg({
        "t_x": lambda s: s.iloc[trk_idx],
        "t_y": lambda s: s.iloc[trk_idx],
        "t_z": lambda s: s.iloc[trk_idx],
        "t_eQOP": lambda s: s.iloc[trk_idx],
        "eQOP_flt": lambda s: s.iloc[trk_idx],
        "eQOP_smt": lambda s: s.iloc[trk_idx],
    })
    
    aggregation["t_p"] = abs(1./aggregation["t_eQOP"])
    aggregation["p_smt"] = abs(1./aggregation["eQOP_smt"])
    aggregation["p_flt"] = abs(1./aggregation["eQOP_flt"])
    
    if clip_abs:
        aggregation["p_smt"] = np.clip(aggregation["p_smt"], clip_abs[0], clip_abs[1])
        aggregation["p_flt"] = np.clip(aggregation["p_flt"], clip_abs[0], clip_abs[1])
    
    fig, ax = plt.subplots(2,3)
    fig.suptitle("At track position '{}'".format(trk_idx))

    _, b, _ = ax[0,0].hist(aggregation["p_flt"], bins=bins, alpha=0.5, label="{} prt".format(fitter_name))
    _ = ax[0,0].hist(aggregation["t_p"], bins=b, alpha=0.5,label="true")
    ax[0,0].set_title("filtered & true")
    ax[0,0].set_yscale('log')
    ax[0,0].legend()
    
    _, b, _ = ax[1,0].hist(aggregation["p_smt"], bins=bins, alpha=0.5, label="{} flt".format(fitter_name))
    _ = ax[1,0].hist(aggregation["t_p"], bins=b, alpha=0.5,label="true")
    ax[1,0].set_title("smoothed & true")
    ax[1,0].set_yscale('log')
    ax[1,0].legend()
    
    ratio = np.array(aggregation["p_flt"]) / np.array(aggregation["t_p"])
    if clip_ratio:
        ratio = np.clip(ratio, clip_ratio[0], clip_ratio[1])
    _ = ax[0,1].hist(ratio, bins=bins)
    ax[0,1].set_title("ratio flt / true")
    
    ratio = np.array(aggregation["p_smt"]) / np.array(aggregation["t_p"])
    if clip_ratio:
        ratio = np.clip(ratio, clip_ratio[0], clip_ratio[1])
    _ = ax[1,1].hist(ratio, bins=bins)
    ax[1,1].set_title("ration smt / true")
    
    get_pos = {
        "x": lambda df: df["t_x"],
        "y": lambda df: df["t_y"],
        "z": lambda df: df["t_z"],
        "r": lambda df: np.hypot(df["t_x"], df["t_y"]),
    }
    
    all_pos = get_pos[main_direction](trackstates)
    
    ax[0,2].hist(get_pos[main_direction](aggregation), range=(min(all_pos), max(all_pos)))
    ax[0,2].set_title("{}-position at track index {}".format(main_direction, trk_idx))

    ax.flat[-1].set_visible(False)
    return fig, ax
    


##########################################################
# ratio true/predicted and true/filtered at track states #
##########################################################

def performance_at_trackstates(trackstates, main_direction, clip=(0,8), log_scale=False):
    a = SimpleTrackstatesGetter(trackstates)

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    xs = []
    ratios_flt = []
    ratios_prt = []

    for tx, ty, tz, track_true_qop, track_prt_qop, track_flt_qop in zip(*a.get(["t_x", "t_y", "t_z", "t_eQOP", "eQOP_prt", "eQOP_flt"])):
        xvalues = make_x_value(tx, ty, tz, main_direction)

        for x, t_qop, prt_qop, flt_qop in zip(xvalues, track_true_qop, track_prt_qop, track_flt_qop):
            xs.append(x)
            ratios_flt.append(np.clip(float(flt_qop/t_qop), clip[0], clip[1]))
            ratios_prt.append(np.clip(float(prt_qop/t_qop), clip[0], clip[1]))

    ax[0].axhline(y = 1, color = 'grey', linestyle = '-')
    ax[0].scatter(xs, ratios_prt, alpha=0.05)
    ax[0].set_title("Forward (predicted) p/p_true")
    ax[1].axhline(y = 1, color = 'grey', linestyle = '-')
    ax[1].scatter(-1 * np.array(xs), ratios_flt, alpha=0.05)
    ax[1].set_title("Backward (filtered) p/p_true")

    if log_scale:
        ax[1].set_yscale('log')
        ax[0].set_yscale('log')

    return fig, ax


######################
# Correlation matrix #
######################

def correlation_scatter_plot(summary, states, clip_res, do_printout=False):
    fig, ax = plt.subplots(2,1)
    
    keys = ["chi2Sum", 't_theta', 't_phi', 't_p', 'res_p_fit', 't_delta_p']
    correlation_plot(summary[keys], (fig, ax[0]))

    event = states["event_nr"].to_numpy()
    traj = states["multiTraj_nr"].to_numpy()

    data = summary[keys].to_numpy().T
    k_res_p_fit = 4
    k_true_p_delta = 5

    ax[1].scatter(np.clip(data[k_res_p_fit], clip_res[0], clip_res[1]), data[k_true_p_delta], alpha=0.5)
    ax[1].set_xlabel(keys[k_res_p_fit])
    ax[1].set_ylabel(keys[k_true_p_delta])

    def print_idx_tuples(label, idxs):
        strs = ["(e={}, t={}, r={:.2f}, l={:.2f})".format(event[i], traj[i], data[k_res_p_fit,i], data[k_true_p_delta,i]) for i in idxs ]
        if len(strs) > 0:
            print("{} ({:.1f}%)\n\t".format(label, 100.*len(idxs)/len(event)),", ".join(strs))
        else:
            print("no samples")

    # Energy increase
    idxs_increase = np.nonzero(data[k_res_p_fit] > 0.5)[0]
    idxs_decrease = np.nonzero(np.logical_and(data[k_res_p_fit] < -0.5, data[k_true_p_delta] >= -0.5))[0]
    idxs_loss = np.nonzero(np.logical_and(data[k_res_p_fit] < -0.5, data[k_true_p_delta] < -0.5))[0]

    if do_printout:
        print_idx_tuples("Energy decrease without loss",idxs_decrease)
        print_idx_tuples("Energy increase",idxs_increase)
        print_idx_tuples("Wrong energy loss", idxs_loss)

    return fig, ax

def single_particle_momentumplot(tracksummary, trackstates, label_predicted, label_filtered):
    fig, ax = plt.subplots()

    # ????????
    x = trackstates["pathLength"].to_numpy()
    assert len(x) == 1
    x = x[0]

    p_true_states = 1./abs(trackstates["t_eQOP"][0])
    p_prt_states = 1./abs(trackstates["eQOP_prt"][0])
    p_flt_states = 1./abs(trackstates["eQOP_flt"][0])

    ax.plot(x, p_true_states, label="true p")
    ax.plot(x, p_prt_states,  label=label_predicted)
    ax.plot(x, p_flt_states,  label=label_filtered, ls="--")

    p_true = float(tracksummary["t_p"][0])
    p_fit = 1./abs(float(tracksummary["eQOP_fit"][0]))

    ax.scatter([0.], [p_true], label="true p (origin)")
    ax.scatter([0.], [p_fit], label="fitted p")

    ax.legend()

    return fig, ax




def uproot_to_pandas(summary, states):
    exclude_keys = ['measurementChi2', 'outlierChi2', 'measurementVolume', 'measurementLayer', 'outlierVolume', 'outlierLayer']
    summary_keys = [ k for k in summary.keys() if not k in exclude_keys ]
    
    summary_df = ak.to_dataframe(summary.arrays(summary_keys), how='outer') \
        .reset_index() \
        .drop(["entry", "subTraj_nr", "subentry"], axis=1) \
        #.set_index(["event_nr", "multiTraj_nr"])
    
    states_df = ak.to_dataframe(states.arrays(), how='outer') \
        .reset_index() \
        .drop(["entry", "subTraj_nr"], axis=1) \
        .rename({"subentry": "trackState_nr"}, axis=1) \
        #.set_index(["event_nr","multiTraj_nr","trackState_nr"])
            
    summary_df["p_fit"] = abs(1./summary_df["eQOP_fit"])
    summary_df["res_p_fit"] = summary_df["p_fit"] - summary_df["t_p"]
    
    def delta_p(df):
        p = 1./abs(df["t_eQOP"].to_numpy())
        return p[0] - p[-1]

    summary_df["t_delta_p"] = states_df.groupby(["event_nr", "multiTraj_nr"]).apply(delta_p).to_numpy()
    
    return summary_df, states_df


def make_full_residual_plot(df):
    fig, axes = plt.subplots(2,3)

    for ax, key in zip(axes.flatten(), ['res_eLOC0_fit', 'res_eLOC1_fit', 'res_ePHI_fit', 'res_eTHETA_fit', 'res_eQOP_fit', 'res_eT_fit']):
        ax.hist(df[key], bins='rice')
        ax.set_yscale('log')
        ax.set_title(key[5:][:-4])
        
    return fig, axes


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSF/KF Analysis script')
    parser.add_argument('input_dir', help="where the root/ and csv/ dirs are")
    parser.add_argument('--main_direction', 
                        help="e.g. x for telescope and r for cylindrical", 
                        type=str, choices=['r', 'x', 'y', 'z'], default="truth")
    parser.add_argument('--disable_meas_holes', help="do not do the measurements holes plot", 
                        default=False, action="store_true")
    
    args = vars(parser.parse_args())
    
    path = Path(args["input_dir"])
    assert path.exists() and (path / "root").exists()

    summary_gsf, trackstates_gsf = uproot_to_pandas(
        uproot.open(str(path / "root/tracksummary_gsf.root:tracksummary")),
        uproot.open(str(path / "root/trackstates_gsf.root:trackstates"))
    )
    
    summary_kf, trackstates_kf = uproot_to_pandas(
        uproot.open(str(path / "root/tracksummary_kf.root:tracksummary")),
        uproot.open(str(path / "root/trackstates_kf.root:trackstates"))
    )

    fig, ax = ratio_residual_plot(summary_gsf, summary_kf)
    fig.suptitle("Ratio/Res")
    fig.tight_layout()
    
    fig, ax = correlation_scatter_plot(summary_gsf, trackstates_gsf, clip_res=(-4,4))
    fig.suptitle("GSF correlation")
    fig.tight_layout()

    fig, ax = correlation_scatter_plot(summary_kf, trackstates_kf, clip_res=(-4,4))
    fig.suptitle("KF correlation")
    fig.tight_layout()

    if not args["disable_meas_holes"]:
        print("Make plots at last surface...")
        fig, ax = plot_at_track_position(0, trackstates_gsf, "GSF", args["main_direction"], bins=50)
        fig.suptitle("GSF - last measurement surface")
        fig.tight_layout()
    
        print("Make plots at first surface...")
        fig, ax = plot_at_track_position(-1, trackstates_gsf, "GSF", args["main_direction"], bins=50)
        fig.suptitle("GSF - first measurement surface")
        fig.tight_layout()

    plt.show()
