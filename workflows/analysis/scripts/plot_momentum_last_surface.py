import matplotlib.pyplot as plt
import numpy as np
import pickle

from gsfanalysis.pandas_import import *


with open(snakemake.input[0], "rb") as f:
    summary_gsf = pickle.load(f)
    

with open(snakemake.input[1], "rb") as f:
    summary_kf = pickle.load(f)

# summary_gsf, summary_kf = select_particles_and_unify_index(
#     summary_gsf.copy(), summary_kf.copy(), #max_eloss_first_surface=np.inf,
# )

assert len(summary_gsf) > 0
assert len(summary_kf) > 0

try:
    plt.figure()
    _, bins, _ = plt.hist(summary_gsf.max_material_fwd, bins=100, range=(0,1.0), label="max", histtype="step")
    _, bins, _ = plt.hist(summary_gsf.sum_material_fwd, bins=bins, range=(0,1.0), label="sum", histtype="step")
    plt.legend()
    plt.yscale('log')
    plt.title("material whole eta")
except:
    print("missing material column")

p_bins=np.linspace(0,13,130)

plt.figure()
plt.title("Momentum distribution at last surface")
plt.hist(summary_gsf.t_final_p, bins=p_bins, color="black", histtype="step", label="True")
plt.hist(summary_kf.final_eP_flt, bins=p_bins, color="tab:blue", histtype="step", label="KF (filtered)")
plt.hist(summary_gsf.final_eP_flt, bins=p_bins, color="tab:orange", histtype="step", label="GSF (filtered)")
plt.legend(loc="upper left")
plt.xlabel("$p_{flt}$ [GeV]")
plt.yscale('log')
plt.savefig(snakemake.output[0])

centerize = lambda df: df[ abs(df.t_eta) < 1 ].copy()
summary_gsf_center = centerize(summary_gsf)
summary_kf_center = centerize(summary_kf)


plt.figure()
plt.title("|eta| < 1")
plt.hist(summary_gsf_center.t_final_p, bins=p_bins, color="black", histtype="step", label="True")
plt.hist(summary_kf_center.final_eP_flt, bins=p_bins, color="tab:blue", histtype="step", label="KF (flt)")
plt.hist(summary_gsf_center.final_eP_flt, bins=p_bins, color="tab:orange", histtype="step", label="GSF (flt)")
plt.yscale('log')
plt.savefig(snakemake.output[1])

try:
    plt.figure()
    _, bins, _ = plt.hist(summary_gsf_center.max_material_fwd, bins=100, range=(0,1.0), label="max", histtype="step")
    _, bins, _ = plt.hist(summary_gsf_center.sum_material_fwd, bins=bins, range=(0,1.0), label="sum", histtype="step")
    plt.legend()
    plt.yscale('log')
    plt.title("material |eta| < 1")
except:
    print("missing material column")

res_pnorm = abs(summary_gsf_center.t_final_p - summary_gsf_center.final_eP_flt) / summary_gsf_center.t_final_p
remove_good = summary_gsf_center[ res_pnorm > 0.05 ].copy()
keep_good = summary_gsf_center[ res_pnorm <= 0.05 ].copy()

bins=np.linspace(0,8.5,85)
plt.figure()
plt.title("|eta| < 1, remove good")
plt.hist(remove_good.t_final_p, bins=bins, color="black", histtype="step", label="True")
plt.hist(remove_good.final_eP_flt, bins=bins, color="tab:orange", histtype="step", label="GSF (flt)")
plt.savefig(snakemake.output[2])

try:
    plt.figure()
    H, x_edges, y_edges = np.histogram2d(remove_good.final_eP_flt, remove_good.max_material_fwd, bins=(p_bins, np.linspace(0,0.5,20)))

    H = (y_edges[:-1] + np.diff(y_edges))*H
    print(H)

    plt.plot(x_edges[:-1]+np.diff(x_edges), np.mean(H, axis=1))
    plt.xlim(0,8.5)
    plt.ylim(0,2)
    plt.xlabel("final p flt")
    plt.ylabel("< x/x0 max >")


    keys = ["nStates", "nMeasurements", "nOutliers", "nHoles", "chi2Sum", "max_material_fwd", "sum_material_fwd"]

    print("keep good mean")
    print(keep_good[keys].mean())
    print("")

    print("remove good mean")
    print(remove_good[keys].mean())
    print("")

    print("remove good, between 0.75 and 0.95")
    print(remove_good[ remove_good.final_eP_flt.between(0.75,0.95) ][keys].mean())
    print("")

    print("remove good, between 3.8 and 4.1")
    print(remove_good[ remove_good.final_eP_flt.between(3.8,4.1) ][keys].mean())
    print("")

    print("remove good, between 8 and 9")
    print(remove_good[ remove_good.final_eP_flt.between(8,9) ][keys].mean())
    print("")
except:
    print("missing material column")

# # r=[0.75,0.95]
# r=[3.8,4.1]
# r=[3.3,4.7]
# 
# 
# 
# plt.figure()
# plt.hist2d(issue1.t_final_p, issue1.final_eP_flt, bins=(10,10), range=[r, r])
# plt.xlabel("true final p")
# plt.ylabel("p final flt")
# plt.tight_layout()

# plt.show()
