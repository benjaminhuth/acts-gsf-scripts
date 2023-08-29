

def preprocess_tracksummary(tracksummary_gsf, tracksummary_kf):
    f = lambda df: None if len(df) != 2 else df
    tracksummary_gsf = tracksummary_gsf.groupby("event_nr").apply(f).reset_index(drop=True)
    tracksummary_kf = tracksummary_kf.groupby("event_nr").apply(f).reset_index(drop=True)

    # Remove events where any of the electrons has a energy less then 10 GeV
    f = lambda df: None if any(df.t_pT < 10) else df
    tracksummary_gsf = tracksummary_gsf.groupby("event_nr").apply(f).reset_index(drop=True)
    tracksummary_kf = tracksummary_kf.groupby("event_nr").apply(f).reset_index(drop=True)

    # Unify index
    tracksummary_gsf = tracksummary_gsf.set_index(["event_nr", "multiTraj_nr"]).copy()
    tracksummary_kf = tracksummary_kf.set_index(["event_nr", "multiTraj_nr"]).copy()

    unified_index = tracksummary_gsf.index.intersection(tracksummary_kf.index)

    tracksummary_gsf = tracksummary_gsf.loc[unified_index, :].reset_index(drop=False).copy()
    tracksummary_kf = tracksummary_kf.loc[unified_index, :].reset_index(drop=False).copy()

    return tracksummary_gsf, tracksummary_kf
