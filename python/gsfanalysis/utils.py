from collections.abc import Iterable
import matplotlib.pyplot as plt
import atlasify


def _add_to_axes(axes, add_fn):
    if isinstance(axes, Iterable):
        for ax in axes.flatten():
            add_fn(ax)
        return axes
    else:
        add_fn(axes)
        return axes


def add_commit_hash(axes, run_config):
    def add(ax):
        # Commit hash
        ax.text(
            0.01,
            1.01,
            run_config["acts-commit-hash"],
            color="grey",
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize="x-small",
            transform=ax.transAxes,
        )

    return _add_to_axes(axes, add)


def add_run_infos(axes, run_config, gsf=True):
    def add(ax):
        # Base info
        sim = "FATRAS" if run_config["fatras"] else "Geant4"
        detector = "ODD" if run_config["detector"] == "odd" else run_config["detector"]
        if run_config["pmin"] == run_config["pmax"]:
            momentum = str(run_config["pmin"])
        else:
            momentum = "{}-{}".format(run_config["pmin"], run_config["pmax"])

        info_str = f"{sim}, {detector}, {momentum}GeV"

        # Additional GSF infos
        if gsf:
            gsf_infos = "{} cmps, {:.0e} weight cutoff".format(
                run_config["gsf"]["maxComponents"], run_config["gsf"]["weightCutoff"]
            )
            info_str = info_str + "\n" + gsf_infos

        atlasify.atlasify(
            atlas=False,
            subtext=info_str,
            enlarge=(20 if ax.get_yscale() == "log" else 1.3),
            axes=ax,
        )

    axes = add_commit_hash(axes, run_config)
    return _add_to_axes(axes, add)
