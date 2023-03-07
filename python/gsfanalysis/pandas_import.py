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

    if states is None:
        return summary_df
    else:
        states_keys = [k for k in states.keys() if not "gsf" in k]

        states_df = (
            ak.to_dataframe(states.arrays(states_keys), how="outer")
            .reset_index()
            .drop(["entry", "subTraj_nr"], axis=1)
            .rename({"subentry": "trackState_nr"}, axis=1)
        )  # .set_index(["event_nr","multiTraj_nr","trackState_nr"])

        summary_df["p_fit"] = abs(1.0 / summary_df["eQOP_fit"])
        summary_df["res_p_fit"] = summary_df["p_fit"] - summary_df["t_p"]

        def delta_p(df):
            p = 1.0 / abs(df["t_eQOP"].to_numpy())
            return p[0] - p[-1]

        summary_df["t_delta_p"] = (
            states_df.groupby(["event_nr", "multiTraj_nr"]).apply(delta_p).to_numpy()
        )

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


def remove_outliers_and_unify_index(summary1, summary2):
    summary1no = summary1[summary1["nOutliers"] == 0]
    summary2no = summary2[summary2["nOutliers"] == 0]

    summary1no = summary1no.set_index(["event_nr", "multiTraj_nr"])
    summary2no = summary2no.set_index(["event_nr", "multiTraj_nr"])
    common_idx = summary1no.index.intersection(summary2no.index)
    summary1no = summary1no.loc[common_idx, :].reset_index()
    summary2no = summary2no.loc[common_idx, :].reset_index()

    assert len(summary1no) == len(summary2no)

    return summary1no, summary2no
