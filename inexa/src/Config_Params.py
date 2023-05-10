# -*- coding: utf-8 -*-
"""
This File keeps track of general configurations
For example, runtime and size of timestep
"""

import os
import pandas as pd
import numpy as np


# Returns an instance of class Params
# Class Params contains basic parameters for the simulation (e.g. simulation time)
def config_params():
    #Time in msec
    lengthST = 300000 #Total simulation time: default value 300000
    t = 0.005 #Timestep size in seconds

    # Create object from class params containing the parameters listed above
    return params(lengthST, t)

"""
"""
class params:

    def __init__(self, lengthST, t):
        self.lengthST = lengthST
        self.t = t

    def getTInMS(self):
        return int(self.t * 1000)

    # Saves converted spike train to result file.
    def saveSpikeTrain(self, spikeTrain, path_to_resultfile):
        pd.DataFrame(self.getSpikeTrain(spikeTrain)).to_csv(path_to_resultfile + ".csv", index=False)

    # Converts spikeTrain
    # TODO please insert useful description here
    def getSpikeTrain(self, spikeTrain):
        ind = (spikeTrain == 1).astype(int)
        num_spiketrains = np.shape(spikeTrain)[1]
        most_spikes = np.max(np.sum(ind, axis=1))
        tmp = np.multiply(ind, 1 / 1000 * np.arange(0, num_spiketrains))
        nzero_ind = (tmp != 0).any(axis=0)
        tmp = tmp[:, nzero_ind]
        B = np.zeros((np.shape(tmp)[0], most_spikes))
        # print("most_spikes",most_spikes)
        for i in np.arange(0, np.shape(tmp)[0]):
            B[i, :] = np.concatenate(
                (np.sort(tmp[i, :][tmp[i, :] != 0]), np.zeros(most_spikes - np.shape(tmp[i, :][tmp[i, :] != 0])[0], )),
                axis=0)
        # print(B)
        return np.transpose(B)

