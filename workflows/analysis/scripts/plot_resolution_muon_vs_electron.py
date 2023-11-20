from math import floor, ceil, log, pow
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gsfanalysis.pandas_import import *
import gsfanalysis.statistics as stats
from gsfanalysis.core_tail_utils import add_core_to_df_quantile

import tqdm

from utils_plotting import plot_binned_errorbar

def add_columns(df):
    p_fit = df["t_charge"]*(1/df["eQOP_fit"])
    df["res_eP_fit"] = df["t_p"] - p_fit
    df["res_ePNORM_fit"] = (df["t_p"] - p_fit) / df["t_p"]

    pT_fit = p_fit*np.sin(df["eTHETA_fit"])
    test = df["t_p"]*np.sin(df["t_theta"])
    assert ((df["t_pT"] - test) < 1.e-3).all()

    df["res_ePT_fit"] = (df["t_pT"] - pT_fit) / df["t_pT"]

    t_pL = np.sqrt(df["t_p"]**2 - df["t_pT"]**2)

    test = np.abs(df["t_p"]*np.cos(df["t_theta"]))
    assert ((t_pL - test) < 1.e-3).all()

    pL_fit = np.abs(p_fit*np.cos(df["eTHETA_fit"]))
    df["res_ePL_fit"] = (t_pL - pL_fit) / t_pL
    
    return df



def load(f):
    s = uproot_to_pandas(
        uproot.open(f"{f}:tracksummary"),
    )
    
    s = s[ (s.event_nr < 10) ].copy()
    
    prev_len = len(s)
    s = select_particles_and_unify_index(s)
    print(Path(f).name,"outlier & hole removal: ", prev_len,"->",len(s))
    
    s["res_eQOP_fit"] *= 1000
    s = add_columns(s)
    
    return s

summary_gsf = load(snakemake.input[0])
summary_kf_e = load(snakemake.input[1])
summary_kf_mu = load(snakemake.input[2])

# _, bins, _ = plt.hist(summary_gsf[KEY], bins=100, alpha=0.5, label="GSF")
# plt.title(f"{KEY} - {STAT.__name__}")
# plt.hist(summary_kf_e[KEY], bins=bins, alpha=0.5, label="KF e")
# plt.hist(summary_kf_mu[KEY], bins=bins, alpha=0.5, label="KF mu")
# plt.yscale('log')
# plt.legend()
# plt.show()

print(summary_gsf.keys())


eta_bins = np.linspace(-3,3,20)
pt_bins = np.linspace(0,100,20)
q=0.999 #0.95

def MAE(x, axis=None):
    return np.mean(np.abs(x), axis=axis)

#STAT = np.std
#STAT = stats.rms
STAT = MAE

ylabels = [
    f"$\\mathrm{{{STAT.__name__.upper()}}}(q/p_{{true}} - q/p_{{fit}})\quad[MeV^{{-1}}]$",
    f"$\\mathrm{{{STAT.__name__.upper()}}}((p_{{true}} - p_{{fit}})/p_{{true}})$",
    f"$\\mathrm{{{STAT.__name__.upper()}}}((p_{{true}} - p_{{fit}}))$",
]
keys = ["res_eQOP_fit"] #, "res_ePNORM_fit", "res_eP_fit"]

bin_keys = ["t_eta", "t_pT"]
xlabels = ["$\eta$", "$p_T$ [GeV]"]
xbins = [eta_bins, pt_bins]

fig, axes = plt.subplots(len(keys),len(bin_keys), figsize=(5*len(bin_keys),4*len(keys)))

if len(axes.shape) == 1:
    axes = axes[np.newaxis, :]

for axrow, ylabel, key in zip(axes, ylabels, keys):

    core_label = ylabel + "_core"
    summary_gsf = add_core_to_df_quantile(summary_gsf, key, q, core_label)
    summary_kf_e = add_core_to_df_quantile(summary_kf_e, key, q, core_label)
    summary_kf_mu = add_core_to_df_quantile(summary_kf_mu, key, q, core_label)

    gsf_mask = summary_gsf[core_label]
    kfe_mask = summary_kf_e[core_label]
    kfmu_mask = summary_kf_mu[core_label]

    for ax, bin_key, bins, xlabel in zip(axrow, bin_keys, xbins, xlabels):
        # fmt: off
        plot_binned_errorbar(ax, summary_gsf.loc[gsf_mask, bin_key], summary_gsf.loc[gsf_mask, key], bins, STAT,
                             color="tab:orange", label="GSF with electrons", fmt="none")
        plot_binned_errorbar(ax, summary_kf_e.loc[kfe_mask, bin_key], summary_kf_e.loc[kfe_mask, key], bins, STAT,
                             color="tab:blue", label="KF with electrons", fmt="none")
        plot_binned_errorbar(ax, summary_kf_mu.loc[kfmu_mask, bin_key], summary_kf_mu.loc[kfmu_mask, key], bins, STAT,
                             color="tab:cyan", label="KF with muons", fmt="none")
        # fmt: on
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        title = key.replace("res_e","").replace("_fit", "")
        ax.set_title(f"{title} resolution vs {xlabel[:6]}")

if snakemake.params["log"]:
    for a in axes.flatten():
        a.set_yscale('log')
        a.set_ylim(
            pow(10, floor(log(a.get_ylim()[0])/log(10))),
            pow(10, ceil(log(a.get_ylim()[1])/log(10)))
        )

axes[0,1].legend()
fig.tight_layout()

# fig2, ax = plt.subplots()
# plot_binned_errorbar(ax, summary_gsf.loc[gsf_mask, bin_key], summary_gsf.loc[gsf_mask, key], bins, STAT,
#                         color="tab:orange", label="GSF with electrons", fmt="none")
# plot_binned_errorbar(ax, summary_kf_e.loc[kfe_mask, bin_key], summary_kf_e.loc[kfe_mask, key], bins, STAT,
#                         color="tab:blue", label="KF with electrons", fmt="none")
# plot_binned_errorbar(ax, summary_kf_mu.loc[kfmu_mask, bin_key], summary_kf_mu.loc[kfmu_mask, key], bins, STAT,
#                         color="tab:cyan", label="KF with muons", fmt="none")


if snakemake.config["plt_show"]:
    plt.show()

fig.savefig(snakemake.output[0])
