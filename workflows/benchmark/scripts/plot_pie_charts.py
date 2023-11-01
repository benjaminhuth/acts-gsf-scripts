
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import get_count_dict, group_dict, color_dict


def add_other_and_percentize(data_dict, total_samples, format_percent=True):
    sum_values = sum(data_dict.values())
    data_dict["Other"] = total_samples - sum_values
    
    if not format_percent:
        return data_dict
    else:
        data_formatted = {}
        
        for k in data_dict.keys():
            r = data_dict[k]/total_samples
            data_formatted[f"{k} ({r:.1%})"] = data_dict[k]
            
        return data_formatted


count_dict, _ = get_count_dict(snakemake.input[0])

fig2, ax2 = plt.subplots()
# ax2.set_title("GSF sample distribution")


data = { k: sum([ count_dict[sk] for sk in group_dict[k] ]) for k in group_dict.keys() }

import pprint
pprint.pprint(data)

data = add_other_and_percentize(data, count_dict["propagate_impl"], format_percent=False)

pprint.pprint(data)

colors2 = list(color_dict.values()) + ["darkgray"]
ax2.pie(
    data.values(), 
    labels=data.keys(), 
    colors=colors2, 
    autopct='%1.0f%%', 
    wedgeprops={'linewidth': 1, 'ec': 'black'},
    # explode=[0.05 for _ in data],
)

plt.show()
fig2.savefig(snakemake.output[0])
