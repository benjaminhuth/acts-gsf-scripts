from collections.abc import Iterable

import matplotlib.pyplot as plt

def _add_to_axes(axes, add_fn):
    if isinstance(axes, Iterable):
        for ax in axes.flatten():
            add_fn(ax)
        return axes
    else:
        add_fn(axes)
        return axes

def add_gsf_run_infos(axes, run_config):
    def add(ax):
        info_str = "{}, c={},\nwc={}".format(
            "FATRAS" if run_config["fatras"] else "Geant4",
            run_config["gsf"]["maxComponents"],
            run_config["gsf"]["weightCutoff"]
        )
        if run_config["pmin"] == run_config["pmax"]:
            info_str += ", p={}GeV".format(run_config["pmin"])
        else:
            info_str += ", p={}-{} GeV".format(run_config["pmin"], run_config["pmax"])
            
        print(info_str)
        ax.legend(title=info_str, alignment='left', title_fontproperties=dict(weight='bold'))
        
    return _add_to_axes(axes, add)

def add_commit_hash(axes, run_config):
    def add(ax):
        ax.text(0.01,1.01,run_config["acts-commit-hash"],
                color="grey",
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize='x-small',
                transform = ax.transAxes)
        
    return _add_to_axes(axes, add)
