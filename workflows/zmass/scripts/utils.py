import matplotlib.pyplot as plt
import numpy as np

def preprocess_tracksummary(tracksummary_gsf, tracksummary_kf):
    f = lambda df: None if len(df) != 2 else df
    tracksummary_gsf = tracksummary_gsf.groupby("event_nr").apply(f).reset_index(drop=True)
    tracksummary_kf = tracksummary_kf.groupby("event_nr").apply(f).reset_index(drop=True)

    # Remove events where any of the electrons has a energy less then 10 GeV
    f = lambda df: None if any(df.t_pT < 10) else df
    tracksummary_gsf = tracksummary_gsf.groupby("event_nr").apply(f).reset_index(drop=True)
    tracksummary_kf = tracksummary_kf.groupby("event_nr").apply(f).reset_index(drop=True)

    # Unify index
    tracksummary_gsf = tracksummary_gsf.set_index(["event_nr", "track_nr"]).copy()
    tracksummary_kf = tracksummary_kf.set_index(["event_nr", "track_nr"]).copy()

    unified_index = tracksummary_gsf.index.intersection(tracksummary_kf.index)

    tracksummary_gsf = tracksummary_gsf.loc[unified_index, :].reset_index(drop=False).copy()
    tracksummary_kf = tracksummary_kf.loc[unified_index, :].reset_index(drop=False).copy()

    return tracksummary_gsf, tracksummary_kf



class MassSpectrumPlotter:
    def __init__(self):
        self.x = np.arange(0,140,0.1)
        self.bins=np.arange(0,140,1.5)

        self.fig, self.ax = plt.subplots()

        self.ax.set_title("$Z_0$ mass estimate with KF and GSF")
        self.ax.set_xlabel("$Z_0$ invariant mass [GeV]")
        self.ax.set_ylabel("normalized density")

        self.ax.legend()

    def plot_zmass_spectrum(self, pdf, masses, mass_estimate, fitter_name, color):
        self.ax.plot(self.x, pdf(self.x),
                     label=f"fit from {fitter_name}: {mass_estimate:.2f} GeV", color=color)

        self.ax.hist(masses, bins=self.bins, alpha=0.3, color=color, density=True)
        self.ax.vlines([mass_estimate], ymin=0, ymax=0.04, color=[color], ls=":")

    def add_pdg_fit(self):
        Z0_width_lit = 2.4955
        Z0_mass_lit = 91.1876

        self.ax.vlines([91.1876], ymin=0, ymax=0.04, color=['black'], ls=":", label="PDG fit: 91.1876(21) GeV")

