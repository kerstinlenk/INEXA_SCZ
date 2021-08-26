import time
import os
import numpy as np
import pandas as pd
import sys

from src.model import init_model, run_model


def run_and_init_model(results_directory, network_path):
    params, neuron, astrocyte, network = init_model(results_directory, network_path)
    return run_model(neuron.c, neuron, astrocyte, network, params, results_directory)


nrOfIterations = 5
nrConnectivities = 1
tic = time.perf_counter()
# Just run the model for each combination of parameters.

j = 5
folder_j = "../results/network_" + str(j)
print(folder_j)
if not os.path.isdir(folder_j):
    os.makedirs(folder_j)

x = 1


burst_data = []

connectivity_folder = folder_j + "/connect_" + str(x)
if not os.path.isdir(connectivity_folder):
    os.mkdir(connectivity_folder)

for i in range(nrOfIterations):

    path_to_result = connectivity_folder + "/iteration" + str(i)
    if not os.path.isdir(path_to_result + "/"):
        os.mkdir(path_to_result + "/")
    resultsDirectory = path_to_result
    burstData = run_and_init_model(resultsDirectory, network_path="../networks/network_" + str(j) + "/connect_" + str(x) + "/")


    burst_data.append(burstData)

    toc = time.perf_counter()
    print("Time ", toc - tic)

df5 = pd.DataFrame(burst_data)
df5.to_csv(connectivity_folder + "/burst.csv")

toc = time.perf_counter()
print("Time ", toc - tic)
