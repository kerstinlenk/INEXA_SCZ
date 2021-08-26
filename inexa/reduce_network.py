import pandas as pd
import numpy as np
import random
import os
import os.path
from os import path
import shutil


number_of_astrocytes = 107
number_of_neurons = 250
reduction_rate = 0.25 # 25 %

def do_reduction(input_path, output_path, reduce_astrocytes, reduce_neurons):
    astrocyte_indices_tb_removed = random.sample(range(0, number_of_astrocytes), int(reduction_rate * number_of_astrocytes))
    neuron_indices_tb_removed_ex = np.array(random.sample(range(0, 200), 49))
    neuron_indices_tb_removed_in = np.array(random.sample(range(200, 250), 13))
    neuron_indices_tb_removed = np.concatenate((neuron_indices_tb_removed_in, neuron_indices_tb_removed_ex))
    np.count_nonzero(neuron_indices_tb_removed < 200)
    print("Number of exictatory removed: ", np.count_nonzero(neuron_indices_tb_removed < 200))

    if reduce_neurons:
        neuron_network_top = pd.read_csv(input_path + "NeuronNetworkTopology.csv", sep=";", header=None).dropna(how='all', axis=1).to_numpy()
        neuronnetwork_top_cleaned = np.delete(neuron_network_top, neuron_indices_tb_removed, axis=1)
        pd.DataFrame(neuronnetwork_top_cleaned).to_csv(output_path + "NeuronNetworkTopology.csv", header=None, index=None, sep=";")

        base_network = pd.read_csv(input_path + "BaseNetwork.csv", sep=";", header=None).dropna(how='all', axis=1).to_numpy()
        base_network = np.delete(base_network, neuron_indices_tb_removed, axis=0)
        base_network = np.delete(base_network, neuron_indices_tb_removed, axis=1)
        pd.DataFrame(base_network).to_csv(output_path + "BaseNetwork.csv", header=None, index=None, sep=";")
    else:
        shutil.copy2(input_path + "NeuronNetworkTopology.csv", output_path + "NeuronNetworkTopology.csv")
        shutil.copy2(input_path + "BaseNetwork.csv", output_path + "BaseNetwork.csv")

    if reduce_astrocytes:
        astrocyte_network_topology = pd.read_csv(input_path + "AstrocyteNetworkTopology.csv", sep=";", header=None).dropna(how='all',
                                                                                                                           axis=1).to_numpy()
        astrocyte_network_topology_cleaned = np.delete(astrocyte_network_topology, astrocyte_indices_tb_removed, axis=1)
        pd.DataFrame(astrocyte_network_topology_cleaned).to_csv(output_path + "AstrocyteNetworkTopology.csv", header=None, index=None, sep=";")

        astrocyte_connections = pd.read_csv(input_path + "AstrocyteConnections.csv", sep=";", header=None).dropna(how='all', axis=1).to_numpy()
        astrocyte_connections_cleaned = np.delete(astrocyte_connections, astrocyte_indices_tb_removed, axis=0)
        astrocyte_connections_cleaned = np.delete(astrocyte_connections_cleaned, astrocyte_indices_tb_removed, axis=1)
        pd.DataFrame(astrocyte_connections_cleaned).to_csv(output_path + "AstrocyteConnections.csv", header=None, index=None, sep=";")

    else:
        shutil.copy2(input_path + "AstrocyteConnections.csv", output_path + "AstrocyteConnections.csv")
        shutil.copy2(input_path + "AstrocyteNetworkTopology.csv", output_path + "AstrocyteNetworkTopology.csv")

    temp = pd.read_csv(input_path + "AstrocyteNeuronConnections.csv", sep=";", header=None).dropna(how='all', axis=1).to_numpy()
    astrocyteNeuronConnections = np.zeros((number_of_neurons, number_of_neurons, number_of_astrocytes), dtype=np.int8)
    line = 0
    for iii in range(number_of_astrocytes):
        for i in range(number_of_neurons):
            for ii in range(number_of_neurons):
                astrocyteNeuronConnections[i, ii, iii] = temp[line, ii]
            line += 1

    if reduce_neurons:
        astrocyteNeuronConnections = np.delete(astrocyteNeuronConnections, neuron_indices_tb_removed, axis=0)
        astrocyteNeuronConnections = np.delete(astrocyteNeuronConnections, neuron_indices_tb_removed, axis=1)

    if reduce_astrocytes:
        astrocyteNeuronConnections = np.delete(astrocyteNeuronConnections, astrocyte_indices_tb_removed, axis=2)

    if reduce_neurons:
        pd.DataFrame(astrocyteNeuronConnections.T.reshape(-1, number_of_neurons - int(reduction_rate * number_of_neurons))).to_csv(
            output_path + "AstrocyteNeuronConnections.csv", sep=";", header=None, index=None)
    else:
        pd.DataFrame(astrocyteNeuronConnections.T.reshape(-1, number_of_neurons)).to_csv(
            output_path + "AstrocyteNeuronConnections.csv", sep=";", header=None, index=None)




networks = ["network_0", "network_1", "network_2"]
connects = ["connect_0", "connect_1", "connect_2"]

for network in networks:
    for connect in connects:
        input_path = "/home/lea/ETH/INEXA/networks/" + network + "/" + connect + "/"
        for i in range(0,3):
            output_path_astrocytes = "/home/lea/ETH/INEXA/networks/" + network + "/connect_1" + str(connect[-1]) + str(i) + "/"
            output_path_neurons = "/home/lea/ETH/INEXA/networks/" + network + "/connect_2" + str(connect[-1]) + str(i) + "/"
            if not path.exists(output_path_astrocytes):
                os.mkdir(output_path_astrocytes)
            if not path.exists(output_path_neurons):
                os.mkdir(output_path_neurons)

            do_reduction(input_path, output_path_astrocytes, True, False)
            do_reduction(input_path, output_path_neurons, False, True)



