import awkward as ak
import pandas as pd
import numpy as np
import uproot


def uproot_to_pandas(summary, states=None):
    exclude_from_summary_keys = [
        "measurementChi2",
        "outlierChi2",
        "measurementVolume",
        "measurementLayer",
        "outlierVolume",
        "outlierLayer",
    ]
    summary_keys = [k for k in summary.keys() if not k in exclude_from_summary_keys]

    summary_df = (
        ak.to_dataframe(summary.arrays(summary_keys), how="outer")
        .reset_index()
        .drop(["entry", "subTraj_nr", "subentry"], axis=1)
    )  # .set_index(["event_nr", "multiTraj_nr"])

    summary_df = (
        summary_df.sort_values("event_nr", kind="stable").reset_index(drop=True).copy()
    )

    summary_df["res_eP_fit"] = summary_df["t_p"] - abs(1.0 / summary_df["eQOP_fit"])
    summary_df["res_ePNORM_fit"] = summary_df["res_eP_fit"] / summary_df["t_p"]

    if states is None:
        return summary_df

    states_keys = [k for k in states.keys() if not "gsf" in k]

    states_df = (
        ak.to_dataframe(states.arrays(states_keys), how="outer")
        .reset_index()
        .drop(["entry", "subTraj_nr"], axis=1)
        .rename({"subentry": "trackState_nr"}, axis=1)
    )  # .set_index(["event_nr","multiTraj_nr","trackState_nr"])

    states_df = (
        states_df.sort_values("event_nr", kind="stable").reset_index(drop=True).copy()
    )

    summary_df["p_fit"] = abs(1.0 / summary_df["eQOP_fit"])
    summary_df["res_p_fit"] = summary_df["p_fit"] - summary_df["t_p"]

    def delta_p(df):
        p = 1.0 / abs(df["t_eQOP"].to_numpy())
        return p[0] - p[-1]

    qop_loc = states_df.columns.get_loc("t_eQOP")

    summary_df["t_final_p"] = (
        states_df.groupby(["event_nr", "multiTraj_nr"])
        .apply(lambda df: abs(1.0 / df.iloc[0, qop_loc]))
        .to_numpy()
    )

    summary_df["t_p_first_surface"] = (
        states_df.groupby(["event_nr", "multiTraj_nr"])
        .apply(lambda df: abs(1.0 / df.iloc[-1, qop_loc]))
        .to_numpy()
    )

    summary_df["t_delta_p"] = summary_df.t_final_p - summary_df.t_p
    summary_df["t_delta_p_first_surface"] = (
        summary_df.t_p_first_surface - summary_df.t_p
    )

    # summary_df["t_delta_p_first_surface"]

    if "gsf_cmps_weights_flt" in states.keys():
        levelNames = lambda x: {
            0: "track_nr",
            1: "state_nr",
            2: "component_nr",
        }[x]
        components_df = (
            ak.to_dataframe(
                states.arrays(filter_name="/gsf|event|multi/i"),
                how="outer",
                levelname=levelNames,
            )
            .reset_index()
            .drop(columns=["track_nr"])
            .set_index(["event_nr", "multiTraj_nr", "state_nr", "component_nr"])
        )

        return summary_df, states_df, components_df
    else:
        return summary_df, states_df


def remove_outliers_and_unify_index(*args):
    dfs = tuple(df[df["nOutliers"] == 0] for df in args)
    dfs = tuple(df.set_index(["event_nr", "multiTraj_nr"]) for df in dfs)

    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    dfs = tuple(df.loc[common_idx, :].reset_index() for df in dfs)

    assert all([len(dfs[0]) == len(df) for df in dfs])

    return dfs
