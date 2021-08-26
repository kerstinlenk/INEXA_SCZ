import time

from src.Functions import filterSpikeTrains, computeBursts, analyzeAstrocyteActivity
import pandas as pd
import os

import numpy as np

resultsDirectory = "/media/lea/Seagate Expansion Drive/ihateinexa/results_yastro"
# resultsDirectory = "/media/lea/Seagate Expansion Drive/ihateinexa/results_inhibitory_excitatory_strength_modified_new"

# param_configuration = ["results_050_090", "results_060_090"]
param_configuration = ["results_0.075", "results_0.1"]
# param_configuration2 = ["results_astrocytes_reduced"]

# param_configuration2 = ["results_010_070", "results_050_070", "results_060_070",
#                        "results_070_070", "results_070_080", "results_070_090","results_0998_0999923","results_0998_09999615007",
#                        "results_0998_09999807502",
#                         "results_09990004998_0999923", "results_09990004998_09999615007", "results_09990004998_09999807502",
#                           "results_0999500125_0999923", "results_0999500125_09999615007","results_0999500125_09999807502"
#                         ]

# "200", "201", "202", "210", "211", "212", "220", "221", "222"
connects = ["connect_0", "connect_1", "connect_2"]
networks = ["network0", "network1", "network2"]
# param_configuration = ["results_reduced"]

for folder in param_configuration:
    for network in networks:
        for connect in connects:
            activityMean = []
            activityStd = []
            activityLengthmean = []
            activitLengthStd = []

            nrActivationsWithZero = []
            nrZeroActivations = []
            for i in range(0, 10):
                iteration_name = "iteration" + str(i)
                path = resultsDirectory + "/" + folder + "/" + network + "/" + connect + "/" + iteration_name + "/"
                print(path)
                activity_animation = pd.read_csv(path + "AstroData_activityAnimation.csv", dtype=np.single).values
                activity_length, nr_activations_per_astrocyte = analyzeAstrocyteActivity(activity_animation)

                number_of_zero_activations = np.count_nonzero(nr_activations_per_astrocyte)
                nr_activations_per_astrocyte_woz = nr_activations_per_astrocyte[nr_activations_per_astrocyte != 0]
                results_dictionary = {}
                nrZeroActivations.append(number_of_zero_activations)
                activityMean.append(np.mean(activity_length))
                activityStd.append(np.std(activity_length))
                activityLengthmean.append(np.mean(nr_activations_per_astrocyte_woz))
                nrActivationsWithZero.append(np.mean(nr_activations_per_astrocyte))
                activitLengthStd.append(np.std(nr_activations_per_astrocyte_woz))

            dict = {"MeanAstrocyteActivityLength": activityMean, "StdAstrocyteActivityLength": activityStd,
                    "MeanNrActivationsPerAstrocyte": nrActivationsWithZero,
                    "StdNrActivationsPerAstrocyte": activitLengthStd,
                    "MeanNrActivationsPerAstrocyteWOZ": activityLengthmean,
                    "NrZeroActivations": nrZeroActivations}
            pd.DataFrame.from_dict(dict).to_csv(resultsDirectory + "/" + folder + "/" + network + "/" + connect + "/astroInfo.csv", index=None)
