
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

################
# Count samples#
################

search_dict = {
    "propagate_impl": "Propagator<…>::propagate_impl<…>(…)",
    "actor": "GsfActor<…>::operator()<…>(…)",
    "cmp_reduction": "GsfActor<…>::reduceComponents<…>(…)",
    "kalman": "GsfActor<…>::kalmanUpdate<…>(…)",
    "no_measurement": "GsfActor<…>::noMeasurementUpdate<…>(…)",
    "convolution": "GsfActor<…>::convoluteComponents<…>(…)",
    "update_stepper": "GsfActor<…>::updateStepper<…>(…)",
    "transport_cov": "EigenStepper<…>::transportCovarianceToBound(…)",
    "step": "MultiEigenStepperLoop<…>::step<…>(…)",
    "nav1": "Navigator::preStep<…>(…)",
    "nav2": "Navigator::postStep<…>(…)",
}

count_dict = { key: 0 for key in search_dict.keys() }
totalSamples=0

for line in open(snakemake.input[0], "r"):
    samples = int(line.strip().split(' ')[-1])
    
    for k in search_dict.keys():
        if search_dict[k] in line:
            count_dict[k] += samples
            
    totalSamples+=samples


###################
# Helper function #
###################

def add_other_and_percentize(data_dict, total_samples):
    sum_values = sum(data_dict.values())
    data_dict["Other"] = total_samples - sum_values
    
    data_formatted = {}
    
    for k in data_dict.keys():
        r = data_dict[k]/total_samples
        data_formatted[f"{k} ({r:.1%})"] = data_dict[k]
        
    return data_formatted


colors = list(mcolors.TABLEAU_COLORS.keys())
colors.remove("tab:gray")
colors.remove("tab:brown")

##################################
# Pie chart for propagation loop #
##################################

fig, ax = plt.subplots()
ax.set_title("Propagator sample distribution")

data = {
    "GSF Actor": count_dict["actor"],
    "Navigation": count_dict["nav1"] + count_dict["nav2"],
    "Stepping": count_dict["step"],
}

data = add_other_and_percentize(data, count_dict["propagate_impl"])
colors1 = colors[:len(data)-1] + ["darkgray"]
ax.pie(data.values(), labels=data.keys(), colors=colors1)

fig.savefig(snakemake.output[0])

###########################
# Pie chart for GSF Actor #
###########################

fig2, ax2 = plt.subplots()
ax2.set_title("GSF Actor sample distribution")

data2 = {
    "Kalman update": count_dict["kalman"],
    "No-measurement update": count_dict["no_measurement"],
    "Covriance transport": count_dict["transport_cov"],
    "Component\nconvolution" : count_dict["convolution"],
    "Component reduction": count_dict["cmp_reduction"],
}

data2 = add_other_and_percentize(data2, count_dict["actor"])
colors2 = colors[:len(data2)-1] + ["darkgray"]
ax2.pie(data2.values(), labels=data2.keys(), colors=colors2)

fig2.savefig(snakemake.output[1])
