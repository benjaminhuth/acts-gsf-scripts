

color_dict = {
    "stepping & navigation": "tab:blue",
    "mixture reduction": "tab:orange",
    "kalman": "tab:green",
    "mixture convolution": "tab:red",
    "covariance transport": "tab:purple",
}

group_dict = {
    "stepping & navigation": ["step", "nav1", "nav2"],
    "mixture reduction": ["cmp_reduction"],
    "kalman": ["kalman"],
    "mixture convolution": ["convolution"],
    "covariance transport": ["transport_cov"],
}

assert group_dict.keys() == color_dict.keys()

def get_count_dict(input_file):
    search_dict = {
        "propagate_impl": "Propagator<…>::propagate_impl<…>(…)",
        "actor": "GsfActor<…>::operator()<…>(…)",
        "cmp_reduction": "reduceMixtureWithKLDistance(…)",
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

    for line in open(input_file, "r"):
        samples = int(line.strip().split(' ')[-1])
        
        for k in search_dict.keys():
            if search_dict[k] in line:
                count_dict[k] += samples
                
        totalSamples+=samples
        
    return count_dict, totalSamples
