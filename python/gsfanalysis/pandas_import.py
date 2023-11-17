import awkward as ak
import pandas as pd
import numpy as np
import uproot

import datetime

def log(msg):
    print(datetime.datetime.now().strftime("%H:%M:%S"),"   INFO   ",msg,flush=True)


def uproot_to_pandas(summary, states=None):
    log("Start importing summary df")
    # print(summary.keys())
    exclude_from_summary_keys = [
        "measurementChi2",
        "outlierChi2",
        "measurementVolume",
        "measurementLayer",
        "outlierVolume",
        "outlierLayer",
    ] + [ k for k in summary.keys() if "gsf" in k ]
    
    summary_keys = [k for k in summary.keys() if not k in exclude_from_summary_keys]

    summary_df = (
        ak.to_dataframe(summary.arrays(summary_keys), how="outer")
        .reset_index()
        .drop(["entry", "subentry"], axis=1)
    )  # .set_index(["event_nr", "multiTraj_nr"])

    summary_df = (
        summary_df.sort_values("event_nr", kind="stable").reset_index(drop=True).copy()
    )

    summary_df["p_fit"] = abs(1.0 / summary_df.eQOP_fit)
    summary_df["res_eP_fit"] = summary_df.t_p - summary_df.p_fit
    summary_df["res_ePNORM_fit"] = summary_df.res_eP_fit / summary_df.t_p

    if states is None:
        states_df = None
    else:
        states_keys = [k for k in states.keys() if not "gsf" in k]

        log("Start importing states, states.arrays...")
        arrays = states.arrays(states_keys)
        
        log("Do ak.to_dataframe...")
        states_df = ak.to_dataframe(arrays, how="outer").reset_index()
        
        log("Done ak.to_dataframe, now sort...")
        states_df = (
            states_df.sort_values("event_nr", kind="stable").reset_index(drop=True).copy()
        )
        
        # NaNs are non measurment states etc...
        log("Done sorting, now drop nan...")
        states_df = states_df[ ~pd.isna(states_df.t_x) ].copy()

        qop_loc = states_df.columns.get_loc("t_eQOP")
        qop_flt_loc = states_df.columns.get_loc("eQOP_flt")

        log("Now group...")
        groupby = states_df.groupby(["event_nr", "track_nr"])

        log("Done grouping")
        summary_df["t_final_p"] = (
            groupby.apply(lambda df: abs(1.0 / df.iloc[0, qop_loc]))
            .to_numpy()
        )
        
        summary_df["final_eP_flt"] = (
            groupby.apply(lambda df: abs(1.0 / df.iloc[0, qop_flt_loc]))
            .to_numpy()
        )
        
        summary_df["t_p_first_surface"] = (
            groupby.apply(lambda df: abs(1.0 / df.iloc[-1, qop_loc]))
            .to_numpy()
        )
        
        assert (summary_df.t_final_p <= summary_df.t_p_first_surface).all()
        assert (summary_df.t_p_first_surface <= summary_df.t_p).all()

        summary_df["t_delta_p"] = summary_df.t_final_p - summary_df.t_p
        summary_df["t_delta_p_first_surface"] = (
            summary_df.t_p_first_surface - summary_df.t_p
        )
        
    if "gsf_weights" in summary.keys():
        components_df = ak.to_dataframe(
            summary.arrays(filter_name="/gsf|event|multi/i"),
            how="outer",
        ).reset_index().rename(columns={"subsubentry":"component_nr"}).drop(columns=["entry","subentry"])

        components_df["component_nr"] = components_df["component_nr"].fillna(0).astype(int)
    else:
        components_df = None
        
    r = (summary_df, states_df, components_df)
    r = tuple(df for df in r if df is not None)
    
    if len(r) == 1:
        return r[0]
    else:
        return r


def select_particles_and_unify_index(*args, max_outliers=0, max_holes=0, min_measurements=9, max_eloss_first_surface=0.1):
    def select(df):
        # print(df.keys())
        sdf = df[ (df.nOutliers <= max_outliers) & (df.nHoles <= max_holes) & (df.nMeasurements >= min_measurements) ].copy()
        
        if "t_delta_p_first_surface" in df.keys():
            sdf = sdf[ sdf["t_delta_p_first_surface"] <= max_eloss_first_surface ].copy()
        else:
            print(f"WARNING: could not apply selection 't_delta_p_first_surface <= {max_eloss_first_surface}'. Key not present.")
            
        return sdf
    

    dfs = tuple(select(df) for df in args)
    dfs = tuple(df.set_index(["event_nr", "track_nr"]) for df in dfs)

    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    dfs = tuple(df.loc[common_idx, :].reset_index() for df in dfs)

    assert all([len(dfs[0]) == len(df) for df in dfs])
    assert len(dfs) > 0

    if len(dfs) == 1:
        return dfs[0]
    else:
        return dfs
