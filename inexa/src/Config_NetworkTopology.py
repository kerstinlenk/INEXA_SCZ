import os

import numpy as np
import pandas as pd

# Set Neuron parameters
from src.Functions import uniform_3D, distance_coupling, complex_connect, createGaussianConnections
import scipy
def config_networkTopology(premadeNNTPath):
    CultureSpace = np.array([750, 750, 10])

    MakeNewTopology = False
    usePremadeNeuronNetwork = True
    return NetworkTopology(CultureSpace, MakeNewTopology, usePremadeNeuronNetwork, premadeNNTPath)

class NetworkTopology:
    def __init__(self, cultureSpace, MakeNewTopology, usePremadeNeuronNetwork,premadeNNTPath):
        self.cultureSpace = cultureSpace
        self.makeNewTopology = MakeNewTopology
        self.premadeNNTPath = premadeNNTPath # by default "../networks/ANN_30/"
        if self.makeNewTopology:
            self.usePremadeNeuronNetwork = usePremadeNeuronNetwork
        else:
            self.TopologyLoadPath = premadeNNTPath

    def CreateNetworkTopology(self, Neuron, Astrocyte):  # TODO: -check which parameters to store in the object
        # --- Basic activity of all neurons --- which is a random number between 0 and an upper boundary from a triangular distribution
        numNeur = Neuron.number
        self.BaseBasicActivity = np.random.triangular(0, 0.5, 1,
                                                      size=numNeur)  # -why is there only a self.NeuronConnections when the if statement is true?
        self.synapsesConnectedToAstrocytes = np.zeros([numNeur, numNeur])
        # Give each neuron spatial coordinates
        # Use premade defined in premadeNNTPath/NeuronNetworkTopology.csv or create new
        if self.usePremadeNeuronNetwork:
            pathPremadeNN = self.premadeNNTPath + "NeuronNetworkTopology.csv"
            importeddata = pd.read_csv(pathPremadeNN, sep=";", header=None).dropna(how='all', axis=1).values
            self.neuronLocation = np.squeeze(importeddata)

            pathPremadeBN = self.premadeNNTPath + "BaseNetwork.csv"
            self.baseNetwork = pd.read_csv(pathPremadeBN, sep=";", header=None).dropna(how='all', axis=1).as_matrix()

        else:
            self.neuronLocation = uniform_3D(numNeur, self.cultureSpace, Astrocyte.MinimumNeuronDistance)
            neuronConnections = createGaussianConnections(self.neuronLocation, Astrocyte.NeuroSTD, cut_off=1e5)
            #  Excitatory and Inhibitory synaptic strengths is a random number between 0 and an upper boundary from a triangular distribution
            self.baseNetwork = np.zeros((numNeur, numNeur))
            self.baseNetwork[:Neuron.ex0, :] = np.random.triangular(0, 0.5, 1, size=(Neuron.ex0, Neuron.number))
            self.baseNetwork[Neuron.ex0:, :] = np.random.triangular(-1, -0.5, 0, size=(Neuron.in0, Neuron.number))
            self.baseNetwork[neuronConnections == 0] = 0  # non-active neuron connections = 0
            np.fill_diagonal(self.baseNetwork, 0)  # why not all neurons at once?

        # create astrocyte network
        # Each column is connected to each astrocyte.
        if (Astrocyte.useAstrocyteNetwork):
            # create astrocytes
            self.astrocyteLocation = uniform_3D(Astrocyte.numberOfAstrocytes, self.cultureSpace,
                                                Astrocyte.MinimumAstrocyteDistance)
            # Connecting astrocytes that ended up close enough to each other.
            self.astrocyteConnections = distance_coupling(self.astrocyteLocation, Astrocyte.connectionDistance)
            amountOfConnections = np.sum(self.astrocyteConnections, axis=0)
            self.activationThreshold = Astrocyte.slope * amountOfConnections + Astrocyte.intercept
            # Connect the two networks
            self.astrocyteNeuronConnections = complex_connect(self.neuronLocation, self.baseNetwork,
                                                              self.astrocyteLocation,
                                                              Astrocyte.ANconnectivitySTD,
                                                              Astrocyte.MaxAstrocyteReachDistance)
            self.initSynapsesConnectedToAstrocytes(numNeur)

    def initSynapsesConnectedToAstrocytes(self, numNeur):
        for i in range(0, numNeur):
            for ii in range(0, numNeur):
                self.synapsesConnectedToAstrocytes[i, ii] = np.sum(self.astrocyteNeuronConnections[i, ii, :])


    def CollectUsefulValues(self, Neuron):
        self.activeSynapses = np.ascontiguousarray(self.baseNetwork != 0, dtype=np.int8)  #TODO SparseMatrix Candidate
        self.excitatorySynapses = np.ascontiguousarray(self.baseNetwork > 0, dtype=np.int8)
        self.inhibitorySynapses = np.ascontiguousarray(self.baseNetwork < 0, dtype=np.int8)
        # amount of excitatory and inhibtory synapses
        self.numberOfInhibitorySyn = np.sum(self.inhibitorySynapses)
        self.numberOfExcitatorySyn = np.sum(self.excitatorySynapses)
        # synapse effect: Has values 1 for exitatory, 0 for not  connected and -1 for negatives.
        self.synapseEffect = self.excitatorySynapses - self.inhibitorySynapses  # Counting how many excitatory connections ended up to each astrocyte
        self.numberExcitatoryConnectionsToAstrocyte = sum(sum(self.astrocyteNeuronConnections))
        # Counting which synapses are connected to an astrocyte to begin with
        self.NumberAstrocytesConnectedToSynapse = np.sum(self.astrocyteNeuronConnections, axis=2)
        self.numberOfAstrocyteConnections = np.sum(self.astrocyteConnections, axis=0)

    def SaveNetworkTopology(self, Neuron, Astrocyte, directory):
        folderName = directory
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        # Neuron Network
        pd.DataFrame(self.neuronLocation).to_csv(folderName + "/NeuronNetworkTopology.csv", header=None, index=None)
        # Astrocyte Network
        if Astrocyte.useAstrocyteNetwork:
            pd.DataFrame(self.astrocyteLocation).to_csv(folderName + "/AstrocyteNetworkTopology.csv", header=None, index=None)
            pd.DataFrame(self.astrocyteConnections).to_csv(folderName + "/AstrocyteConnections.csv", header=None, index=None)
            pd.DataFrame(self.astrocyteNeuronConnections.T.reshape(-1, Neuron.number)).to_csv(folderName + "/AstrocyteNeuronConnections.csv", header=None, index=None)
        pd.DataFrame(self.baseNetwork).to_csv(folderName + "/BaseNetwork.csv", header=None, index=None)

    def LoadNetworkTopology(self, Neuron, Astrocyte):  #
        #print("Load Network Topology")
        self.synapsesConnectedToAstrocytes = np.zeros([Neuron.number, Neuron.number])
        # impoprt Neuron location and baseNetwork
        pathNNT = self.premadeNNTPath + "NeuronNetworkTopology.csv"
        pathBN = self.premadeNNTPath + "BaseNetwork.csv"
        self.neuronLocation = pd.read_csv(pathNNT, sep=";", header=None).dropna(how='all', axis=1).to_numpy()  # drop empty last column to avoid issues further down the line
        self.baseNetwork = pd.read_csv(pathBN, sep=";", header=None).dropna(how='all', axis=1).to_numpy()
        # import Astrocyte data
        if Astrocyte.useAstrocyteNetwork:
            self.loadAstrocyteNetwork(Astrocyte, Neuron)

    def loadAstrocyteNetwork(self, Astrocyte, Neuron):
        pathANT = self.TopologyLoadPath + "AstrocyteNetworkTopology.csv"
        pathANC = self.TopologyLoadPath + "AstrocyteNeuronConnections.csv"
        pathAC = self.TopologyLoadPath + "AstrocyteConnections.csv"
        self.astrocyteLocation = pd.read_csv(pathANT, sep=";", header=None).dropna(how='all', axis=1).to_numpy(dtype=np.short)
        self.astrocyteConnections = pd.read_csv(pathAC, sep=";", header=None).dropna(how='all', axis=1).to_numpy(np.short)

        temp = pd.read_csv(pathANC, sep=";", header=None).dropna(how='all', axis=1).to_numpy()
        self.astrocyteNeuronConnections = np.zeros((Neuron.number, Neuron.number, Astrocyte.numberOfAstrocytes), dtype=np.int8)
        line = 0
        for iii in range(Astrocyte.numberOfAstrocytes):
            for i in range(Neuron.number):
                for ii in range(Neuron.number):
                    self.astrocyteNeuronConnections[i, ii, iii] = temp[line, ii]
                line += 1

        self.CollectUsefulValues(Neuron)
        self.initSynapsesConnectedToAstrocytes(Neuron.number)
        self.activationThreshold = Astrocyte.slope * self.numberOfAstrocyteConnections + Astrocyte.intercept

    def astrocyte_dropout(self, prob_conn):
        if prob_conn < 1:
            self.astrocyteConnections = self.astrocyteConnections * (np.random.uniform(low=0.0, high=1.0, size=np.shape(self.astrocyteConnections)) <= (prob_conn))
