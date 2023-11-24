import ROOT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad

from utils import *

z0_masses_gsf = pd.read_csv(snakemake.input[0])
z0_masses_kf = pd.read_csv(snakemake.input[1])

valrange=(0,300)

x = ROOT.RooRealVar("x", "x", *valrange)
mean = ROOT.RooRealVar("mean", "mean", *valrange)
width = ROOT.RooRealVar("width", "width", 0, 50)

pdf = ROOT.RooBreitWigner("pdf", "pdf", x, mean, width)

arglist = ROOT.RooLinkedList()
arglist.Add(ROOT.RooFit.Save(True))
arglist.Add(ROOT.RooFit.PrintLevel(-1))


def fit_with_root(masses, bins=20):
    bin_weights, bin_edges = np.histogramdd(masses, bins=bins, range=[valrange])

    # Note that this is NON-RELATIVISTIC!!!
    datahist = ROOT.RooDataHist.from_numpy(bin_weights, [x], bins=bin_edges)
    res = pdf.chi2FitTo(datahist, arglist)

    tf1 = pdf.asTF(x, res.floatParsFinal()).Clone()

    def make_val_err(r):
        return (r.getValV(), r.getError())

    return tf1, {
        "mass": make_val_err(res.floatParsFinal().at(0)),
        "width": make_val_err(res.floatParsFinal().at(1)),
    }


tf1_gsf, res_gsf = fit_with_root(z0_masses_gsf.mass_fit.to_numpy(), bins=30)
tf1_kf, res_kf = fit_with_root(z0_masses_kf.mass_fit.to_numpy(), bins=30)

print(res_gsf)
print(res_kf)

class TF1Wrapper:
    def __init__(self, tf1):
        self.tf1 = tf1
        self.c = quad(lambda x: tf1.Eval(x), *valrange)[0]
    def __call__(self, x):
        return np.array([ self.tf1.Eval(xx) for xx in x]) / self.c

plotter = MassSpectrumPlotter()
plotter.plot_zmass_spectrum(TF1Wrapper(tf1_gsf), z0_masses_gsf.mass_fit, res_gsf["mass"][0], "GSF", "tab:orange")
plotter.plot_zmass_spectrum(TF1Wrapper(tf1_kf), z0_masses_kf.mass_fit, res_kf["mass"][0], "KF", "tab:blue")
plotter.add_pdg_fit()

plotter.ax.legend()
plotter.fig.tight_layout()

plotter.fig.savefig(snakemake.output[0])

plt.show()
