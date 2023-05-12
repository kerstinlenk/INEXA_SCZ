import os
from shutil import copyfile

import numpy as np
import pandas as pd

from src.Config_Astrocyte import config_astrocytes
from src.Config_NetworkTopology import config_networkTopology
from src.Config_Neuron import config_neuron
from src.Config_Params import config_params
from src.Functions import generateNewSpike, saveParameters, computeStatisticalBurstData
from src.Neuron_Info import NeuronInfo


def init_model(resultDirectory="results", premadeNNTPath="nn", param_config=None):
    Params = config_params()
    Network = config_networkTopology(premadeNNTPath, premadeNNTPath)
    Neuron = config_neuron()
    Astrocyte = config_astrocytes(Network)

    # Neuron basic activity (von matlab code main,  ca zeile 33 bis 57)
    Neuron.setSynStr_bruteForce()

    # %%########## ----Creating base network topology------  ##########
    if Network.create_new_neuron_topology:
        Network.create_network_topologies(Neuron, Astrocyte)
        Network.collect_useful_values()
        Network.save_network_topology(Neuron, Astrocyte, resultDirectory)
    else:
        # Loading from file and saving to results folder
        Network.load_network_topologies(Neuron, Astrocyte)
        Network.collect_useful_values()
        Network.save_network_topology(Neuron, Astrocyte, resultDirectory)
        Network.base_activity = np.random.triangular(0, 1 / 2, 1, size=Neuron.number)

    return Params, Neuron, Astrocyte, Network


def run_model(pc, neuron, astrocyte, network, params, directory="results", saveData=True):

    # Update config, reset values where necessary
    neuron.setNeuronConfig(pc, network.base_network_activity, network.base_activity)

    astrocyte.setAstrocyteConfig(neuron.oex, neuron.oin, neuron.number, neuron.synStr, network.active_synapses,
                                 int(params.lengthST / params.getTInMS()))

    # NeuronInfo is just a structure to encapsulate data needed to compute a new spike
    ni = NeuronInfo(neuron.number, params.t, neuron.synStr, neuron.c_matrix, neuron.sensitivityMultiplier)

    # Init first, empty spike as well as spikeTrain
    spike = np.zeros([neuron.number], dtype=np.single)
    spikeTrain = np.zeros([neuron.number, int(params.lengthST / params.getTInMS())], order='F', dtype=np.single)

    # Generate Spike Trains for all neurons
    counter = 0
    for k in range(0, params.lengthST, params.getTInMS()):
        spike = generateNewSpike(k, ni, spike, astrocyte.astrocyteInhibition)
        # save Newly generated spike in spike train
        spikeTrain[:, counter] = spike
        counter += 1
        if astrocyte.usePreSynapse:
            returnValue = astrocyte.computeModifiedPreSynapseModel(spike, numberOfNeurons=neuron.number, network=network, tInMS=params.getTInMS())
            astrocyte.calculateAstrocyteData(network, returnValue, ni.synStr, k)
            ni.synStr = returnValue

    # Save spike train, astrocyte Data and default information about simulation
    if saveData:
        saveParameters(neuron.synStr, neuron.c_matrix, directory, str(pc), str(neuron.oex), str(neuron.oin))
        params.saveSpikeTrain(spikeTrain, directory + "/timestamps")
        astrocyte.saveAstrocyteData(directory, network)

    # compute statistical data
    transformedSpikeTrain = params.getSpikeTrain(spikeTrain)
    return computeStatisticalBurstData(transformedSpikeTrain, params.lengthST)
