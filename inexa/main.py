import time
import os
import pandas as pd

from src.model import init_model, run_model
from src.usability_functions import create_folder

NETWORK_NR = 5
NETWORK_PATH = "../results/network_" + str(NETWORK_NR)
CONNECTIVITY_NR = 1
CONNECTIVITY_PATH = NETWORK_PATH + "/connect_" + str(CONNECTIVITY_NR)


def run_and_init_model(results_directory, network_path):
    params, neuron, astrocyte, network = init_model(results_directory, network_path)
    return run_model(neuron.c, neuron, astrocyte, network, params, results_directory)


create_folder(NETWORK_PATH)
create_folder(CONNECTIVITY_PATH)

burst_data = []

for i in range(1):
    tic = time.perf_counter()

    path_to_result = CONNECTIVITY_PATH + "/iteration_" + str(i)
    create_folder(path_to_result)
    burstData = run_and_init_model(path_to_result, network_path=NETWORK_PATH)

    burst_data.append(burstData)
    toc = time.perf_counter()
    print("Time ", toc - tic)

df5 = pd.DataFrame(burst_data)
df5.to_csv(CONNECTIVITY_PATH + "/burst.csv")

toc = time.perf_counter()
print("Time ", toc - tic)
