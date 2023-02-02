import argparse

import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class SimpleTrackstatesGetter:
    def __init__(self, trackstates):
        self.trackstates = trackstates

    def get(self, keys):
        return [ self.trackstates[key].array(library="np") for key in keys ]

    def loop(self, keys):
        return zip(*self.get(keys))

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


def ratio_hist(ax, tree, bins, label, mask):
    p_true = np.concatenate(tree['t_p'].array(library="np"))
    p_pred = abs(1. / np.concatenate(tree['eQOP_fit'].array(library="np")))
    
    if mask is not None:
        p_true = p_true[mask]
        p_pred = p_pred[mask]
    
    clipped_ratio = np.clip(p_pred / p_true, 0, 2)

    # find mode
    np_n, np_bins = np.histogram(clipped_ratio, bins=bins)
    max_idx = np.argmax(np_n)
    mode = 0.5*(np_bins[max_idx] + np_bins[max_idx+1])

    # draw hist
    n, bins, _ = ax.hist(clipped_ratio, bins=bins, label="{} (mean={:.3f}, mode={:.3f})".format(label, np.mean(clipped_ratio), mode), alpha=0.5)

    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    std = np.average((mids - mean)**2, weights=n)

    print("\tHist {}: {:.3f} +- {:.9f}".format(label, mean, std))

    return bins

def residual_hist(ax, tree, bins, label, mask):
    qop_true = np.concatenate(tree['t_charge'].array(library="np")) / np.concatenate(tree['t_p'].array(library="np"))
    qop_fit = np.concatenate(tree['eQOP_fit'].array(library="np"))

    if mask is not None:
        qop_true = qop_true[mask]
        qop_fit = qop_fit[mask]

    clipped_residual = np.clip(qop_fit - qop_true, -0.2, 0.2)
    _, bins, _ = ax.hist(clipped_residual, bins=bins, alpha=0.5, label=label)

    return bins

########################
# p_fit / p_true ratio #
########################

def make_ratio_plot(summary_gsf, summary_kf, log_scale=False, bins=200):
    # this only passes for electrons where the energy loss is lower than 1 GeV
    def make_energy_loss_mask(trackstates):
        return np.array([ True if abs(abs(1/e[0])-abs(1/e[-1])) < 1 else False for e in trackstates["t_eQOP"].array(library="np")])

    fig, ax = plt.subplots(1,2)

    gsf_energy_loss_mask = None #make_energy_loss_mask(trackstates_gsf)
    kf_energy_loss_mask = None #make_energy_loss_mask(trackstates_kf)

    print("Make ratio plots...")
    b = ratio_hist(ax[0], summary_gsf, bins, "GSF", mask=gsf_energy_loss_mask)
    ratio_hist(ax[0], summary_kf, b, "KF", mask=kf_energy_loss_mask)

    ax[0].set_title("Ratio")
    ax[0].set_xlabel("p_fit / p_true")
    ax[0].legend()

    print("Make residual plots...")
    b = residual_hist(ax[1], summary_gsf, bins, "GSF", mask=gsf_energy_loss_mask)
    residual_hist(ax[1], summary_kf, b, "KF", mask=kf_energy_loss_mask)

    ax[1].set_title("Residual")
    ax[1].set_xlabel("Residual  q/p_fit - q/p_true")
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

def plot_at_track_position(pos, trackstates, fitter_name, main_direction, clip_ratio=(0,2), clip_abs=(0,20)):
    a = SimpleTrackstatesGetter(trackstates)

    end_energies_true = []
    end_energies_flt = []
    end_energies_prt = []
    x_values = []

    for tx, ty, tz, true_qop, flt_qop, prt_qop in a.loop(["t_x", "t_y", "t_z", "t_eQOP", "eQOP_flt", "eQOP_prt"]):
        x_values.append(make_x_value(tx, ty, tz, main_direction)[pos])

        end_energies_true.append(abs(1/true_qop[pos]))
        end_energies_flt.append(abs(1/flt_qop[pos]))
        end_energies_prt.append(abs(1/prt_qop[pos]))

    if clip_abs:
        end_energies_prt = np.clip(end_energies_prt, clip_abs[0], clip_abs[1])
        end_energies_flt = np.clip(end_energies_flt, clip_abs[0], clip_abs[1])
        
    # print("\tprt (fwd): ",np.mean(end_energies_prt), np.std(end_energies_prt))
    # print("\tflt (bwd): ",np.mean(end_energies_flt), np.std(end_energies_flt))
    # print("\ttrue:",np.mean(end_energies_true), np.std(end_energies_true))
    
    fig, ax = plt.subplots(2,3, figsize=(14,5))
    fig.suptitle("At track position '{}'".format(pos))
    
    bins = 200

    _, bins, _ = ax[0,0].hist(end_energies_prt, bins=bins, alpha=0.5, label="{} prt".format(fitter_name))
    _ = ax[0,0].hist(end_energies_true, bins=bins, alpha=0.5,label="true")
    ax[0,0].set_title("prt energy")
    ax[0,0].set_yscale('log')
    ax[0,0].legend()
    
    _, bins, _ = ax[1,0].hist(end_energies_flt, bins=bins, alpha=0.5, label="{} flt".format(fitter_name))
    _ = ax[1,0].hist(end_energies_true, bins=bins, alpha=0.5,label="true")
    ax[1,0].set_title("flt energy")
    ax[1,0].set_yscale('log')
    ax[1,0].legend()
    
    bins = 200
    ratio = np.array(end_energies_prt) / np.array(end_energies_true)
    if clip_ratio:
        ratio = np.clip(ratio, clip_ratio[0], clip_ratio[1])
    _ = ax[0,1].hist(ratio, bins=bins)
    ax[0,1].set_title("prt / true")
    
    ratio = np.array(end_energies_flt) / np.array(end_energies_true)
    if clip_ratio:
        ratio = np.clip(ratio, clip_ratio[0], clip_ratio[1])
    _ = ax[1,1].hist(ratio, bins=bins)
    ax[1,1].set_title("flt / true")

    ax[0,2].hist(x_values)
    ax[0,2].set_title("{} position at pos {}".format(main_direction, pos))

    ax.flat[-1].set_visible(False)
    
    fig.tight_layout()
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

def correlation_plots(tracksummary, trackstates, clip_res):
    def get(key):
        if key == 'p_fit':
            return abs(1./get('eQOP_fit'))
        if key == "res_p_fit":
            return abs(1./get('eQOP_fit')) - get('t_p')
        if key == "true_p_delta":
            teqop = trackstates['t_eQOP'].array(library="np")
            return np.array([ 1./abs(a[0]) - 1./abs(a[-1]) for a in teqop ])
        else:
            return np.concatenate(tracksummary[key].array(library="np"))

    fig, ax = plt.subplots(2,1)
    #keys = ['p_fit', 't_pT', 't_eta', 't_phi', "chi2Sum", "nMeasurements", "nHoles", "nStates"]
    keys = ["chi2Sum", 't_theta', 't_phi', 't_p', 'res_p_fit', 'true_p_delta']

    data = np.vstack([ get(k) for k in keys ])

    event = trackstates["event_nr"].array(library="np")
    traj = trackstates["multiTraj_nr"].array(library="np")

    im = ax[0].imshow(np.corrcoef(data), origin="lower", aspect=0.3)
    fig.colorbar(im, ax=ax[0], label='Interactive colorbar')

    ax[0].set_xticks(np.arange(len(keys)))
    ax[0].set_yticks(np.arange(len(keys)))

    ax[0].set_xticklabels(keys)
    ax[0].set_yticklabels(keys)

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

    print_idx_tuples("Energy decrease without loss",idxs_decrease)
    print_idx_tuples("Energy increase",idxs_increase)
    print_idx_tuples("Wrong energy loss", idxs_loss)

    return fig, ax

def single_particle_momentumplot(tracksummary, trackstates, label_predicted, label_filtered):
    fig, ax = plt.subplots()

    x = trackstates["pathLength"].array(library="np")
    assert len(x) == 1
    x = x[0]

    p_true_states = 1./abs(trackstates["t_eQOP"].array(library="np")[0])
    p_prt_states = 1./abs(trackstates["eQOP_prt"].array(library="np")[0])
    p_flt_states = 1./abs(trackstates["eQOP_flt"].array(library="np")[0])

    ax.plot(x, p_true_states, label="true p")
    ax.plot(x, p_prt_states,  label=label_predicted)
    ax.plot(x, p_flt_states,  label=label_filtered, ls="--")

    p_true = float(tracksummary["t_p"].array(library="np")[0])
    p_fit = 1./abs(float(tracksummary["eQOP_fit"].array(library="np")[0]))

    ax.scatter([0.], [p_true], label="true p (origin)")
    ax.scatter([0.], [p_fit], label="fitted p")

    ax.legend()

    return fig, ax

# def single_particle_path(trackstates)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSF/KF Analysis script')
    parser.add_argument('--main_direction', help="e.g. x for telescope and r for cylindrical", type=str, choices=['r', 'x', 'y', 'z'], default="truth")
    parser.add_argument('--disable_meas_holes', help="do not do the measurements holes plot", default=False, action="store_true")
    args = vars(parser.parse_args())

    summary_gsf = uproot.open("root/tracksummary_gsf.root:tracksummary")
    summary_kf = uproot.open("root/tracksummary_kf.root:tracksummary")
    trackstates_gsf = uproot.open("root/trackstates_gsf.root:trackstates")
    trackstates_kf = uproot.open("root/trackstates_kf.root:trackstates")

    make_ratio_plot(summary_gsf, summary_kf)
    fig, ax = correlation_plots(summary_gsf, trackstates_gsf, clip_res=(-4,4))
    fig.suptitle("GSF correlation")

    fig, ax = correlation_plots(summary_kf, trackstates_kf, clip_res=(-4,4))
    fig.suptitle("KF correlation")

    if not args["disable_meas_holes"]:
        print("Make plots at last surface...")
        fig, ax = plot_at_track_position(0, trackstates_gsf, "GSF", args["main_direction"])
        fig.suptitle("At the last measurement surface")

        print("Make plots at first surface...")
        fig, ax = plot_at_track_position(-1, trackstates_gsf, "KF", args["main_direction"])
        fig.suptitle("At the first measurement surface")

    plt.show()
