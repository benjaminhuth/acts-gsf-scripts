import numpy as np
import ROOT


def fit_with_root(masses, bins=20, valrange=(75, 125)):
    bin_weights, bin_edges = np.histogramdd(masses, bins=bins, range=[valrange])

    #print(bin_weights.reshape(-1,1))
    x = ROOT.RooRealVar("x", "x", *valrange)
    mean = ROOT.RooRealVar("mean", "mean", 0, 150)
    width = ROOT.RooRealVar("width", "width", 0, 50)
    sigma = ROOT.RooRealVar("sigma", "sigma", 0, 10)

    datahist = ROOT.RooDataHist.from_numpy(bin_weights, [x], bins=bin_edges)
    
    # Note that this is NON-RELATIVISTIC!!!
    pdf = ROOT.RooVoigtian("pdf", "pdf", x, mean, width, sigma)

    arglist = ROOT.RooLinkedList()
    arglist.Add(ROOT.RooFit.Save(True))
    arglist.Add(ROOT.RooFit.PrintLevel(-1))

    res = pdf.chi2FitTo(datahist, arglist)

    def make_val_err(r):
        return (r.getValV(), r.getError())

    return {
        "Z0 mass": make_val_err(res.floatParsFinal().at(0)),
        "Γ (width)": make_val_err(res.floatParsFinal().at(1)),
        "sigma": make_val_err(res.floatParsFinal().at(2)),
    }
