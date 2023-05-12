# -*- coding: utf-8 -*-
"""
Class keeping track of everything surrounding astrocytes
"""

import numpy as np

import pandas as pd

from src.Config_NetworkTopology import SEPARATOR, NetworkTopology


def config_astrocytes(network_configuration: NetworkTopology):
    # Minimum distance between 2 neurons in micrometers
    MinimumNeuronDistance = 10

    # Minimum distance between 2 astrocytes in micrometers
    MinimumAstrocyteDistance = 30

    # Standard deviation of connection distance
    NeuroSTD = 200

    # Standard deviation of astrocyte connectivity to neuron
    ANconnectivitySTD = 150

    # Absolute maximum distance after which the astrocyte will not connect to synapses
    MaxAstrocyteReachDistance = 70

    # Calcium at terminal bound to sensors at the beginning between 0 and 1.
    Calcium = 0

    # Resources at each synapse at the beginning between 0 and 1.
    Resources = 1

    # Glutamate amount at the beginning between 0 and 1. With presynapse simulator only this is a constant. With Astrocyte it becomes variable.
    AstrocyteGlutamate = 0.0

    # Effect parameter of astrocyte regulation of synaptic release
    alpha = 0.7

    # Regenerate resources. Every ms proportion RegenRes of used resources is added to resources. 0.01 meaning that 1% of used resources is regenerated each cycle. Between 0 and 1.
    RegenRes = 0.02  # this is 1-np.exp(-Omega_g*delta_t)

    # Regenerate Calcium equilibrium in presynapse. Each ms calcium bound to sensors drops to proportion CaRegen of the previous value. 0.99 meaning that 99% of calcium is left compared to previous amount after 1 ms. Between 0 and 1.
    CaRegen = 0.998  # this np.exp(-Omega_f), Omega_f = 0.002 / ms.

    # Reblock rate of NMDAR magnesium blockade. Mg left after every cycle. Between 0 and 1.
    ReBlock = 0.0

    # Removal of glutamate from astrocyte. Percentage of glutamate left after each ms. Omega_g = -log(GlutamateRemoval^5)/0.005
    GlutamateRemoval = 0.999923

    # Release amount by astrocyte
    AstrocyteReleaseAmount = 0.3

    # Level of Ca initializing glutamate release CaGlutamateReleaseLevelVector = [0.1;0.02;0.1];
    CaGlutamateReleaseLevel = 0.1

    # Cumulation factor for conversion from levels IP3 to calcium. Between 0 and 1.
    IPAccumulation = 0.05

    # Removal of IP3 from single synapse. Percentage of IP3 left after each ms. Omega_IP3 = -log(IP3degrading^5)/0.005
    IP3degrading = 0.85873

    NumberofAstrocytes = 107

    # How strong an effect synapses Ca rise has to astrocyte Ca? Higher number is stonger.
    SynapseCaEffect = 5

    # Maximum distance between 2 (connected[?]) astrocyte in micrometers
    connectionDistance = 100

    # Spontanous activity - only the astrocytes in unactivated state can activate
    SpontaneousActivation = 0

    # Average activation time of an astrocyte in milliseconds
    ActivationProbability = 5 / 1500

    # Average Activation Time
    ActivationTime = 5 / 7000

    # Average Refractory Time
    RefractoryTime = 5 / 5000

    # Increase in required flux to activate an astrocyte for each connection it has
    slope = 0.02

    # Minimum IP3 flux needed to activate an astrocyte
    intercept = 0.205

    # strength of astrocytic inhibition(negative number)
    AstrocyteInhibitionStrength = -0.01

    # astrocyte connectivity dropout
    prob_conn = 1

    return astrocyte_params(MinimumNeuronDistance,
                            MinimumAstrocyteDistance,
                            NeuroSTD,
                            ANconnectivitySTD,
                            MaxAstrocyteReachDistance,
                            network_configuration.pre_synapse,
                            network_configuration.simulate_astrocyte,
                            network_configuration.simulate_astrocyte_network,
                            Calcium,
                            Resources,
                            AstrocyteGlutamate,
                            alpha,
                            RegenRes,
                            CaRegen,
                            ReBlock,
                            NumberofAstrocytes,
                            SynapseCaEffect,
                            GlutamateRemoval,
                            AstrocyteReleaseAmount,
                            CaGlutamateReleaseLevel,
                            IPAccumulation,
                            IP3degrading,
                            connectionDistance,
                            SpontaneousActivation,
                            ActivationProbability,
                            ActivationTime,
                            RefractoryTime,
                            slope,
                            intercept,
                            AstrocyteInhibitionStrength,
                            prob_conn)


""" Class containing astrocyte parameters as in the INXEA model
"""


class astrocyte_params:

    def __init__(self,
                 MinimumNeuronDistance,
                 MinimumAstrocyteDistance,
                 NeuroSTD,
                 ANconnectivitySTD,
                 MaxAstrocyteReachDistance,
                 preSynapse,
                 Astrocyte,
                 AstrocyteNetwork,
                 Calcium,
                 Resources,
                 AstrocyteGlutamate,
                 alpha,
                 RegenRes,
                 CaRegen,
                 ReBlock,
                 numberOfAstrocytes,
                 SynapseCaEffect,
                 GlutamateRemoval,
                 AstrocyteReleaseAmount,
                 CaGlutamateReleaseLevel,
                 IPAccumulation,
                 IPdegrading,
                 connectionDistance,
                 SpontaneousActivation,
                 ActivationProbability,
                 ActivationTime,
                 RefractoryTime,
                 slope,
                 intercept,
                 AstrocyteInhibitionStrength,
                 prob_conn
                 ):
        self.prob_conn = prob_conn  # this regulated dropout of astrocyte connections if the paramter is smaller than 1 (value in [0,1])
        self.MinimumNeuronDistance = MinimumNeuronDistance
        self.MinimumAstrocyteDistance = MinimumAstrocyteDistance
        self.NeuroSTD = NeuroSTD
        self.ANconnectivitySTD = ANconnectivitySTD
        self.MaxAstrocyteReachDistance = MaxAstrocyteReachDistance

        self.usePreSynapse = preSynapse
        self.use_astrocyte = Astrocyte
        self.use_astrocyte_network = AstrocyteNetwork


        if not self.use_astrocyte_network:
            self.number_of_astrocytes = 0
            self.astrocyteInhibition = 0

        if self.usePreSynapse:
            self.Calcium = Calcium
            self.Resources = Resources
            self.AstrocyteGlutamate = AstrocyteGlutamate
            self.alpha = alpha
            self.RegenRes = RegenRes
            self.CaRegen = CaRegen
            self.ReBlock = ReBlock
        if self.use_astrocyte:
            self.GlutamateRemoval = GlutamateRemoval
            self.AstrocyteReleaseAmount = AstrocyteReleaseAmount
            self.CaGlutamateReleaseLevel = CaGlutamateReleaseLevel
            self.IPAccumulation = IPAccumulation
            self.IPdegrading = IPdegrading

        if self.use_astrocyte_network:
            self.number_of_astrocytes = numberOfAstrocytes
            self.SynapseCaEffect = SynapseCaEffect
            self.connectionDistance = connectionDistance
            self.spontaneousActivation = SpontaneousActivation
            self.ActivationProbability = ActivationProbability
            self.ActivationTime = ActivationTime
            self.RefractoryTime = RefractoryTime
            self.slope = slope
            self.intercept = intercept
            self.astrocyteInhibitionStrength = AstrocyteInhibitionStrength

    """
    Initialize Astrocyte config and reset values where necessary
    """

    def setAstrocyteConfig(self, p_oex0, p_oin0, numberOfNeurons, syn_str, activeSynapses, numberOfTimesteps):
        if self.usePreSynapse:
            self.configPreSynapseModel(numberOfNeurons, p_oex0, p_oin0, syn_str, activeSynapses)
        if self.use_astrocyte:
            self.configAstrocyteModel(numberOfNeurons)

        if self.use_astrocyte_network:
            self.configAstrocyteNetworkModel(numberOfNeurons, numberOfTimesteps)

    def configAstrocyteNetworkModel(self, numberOfNeurons, numberOfTimesteps):
        self.astrocyteState = np.zeros([self.number_of_astrocytes], dtype=np.int8)
        self.activityAnimation = np.zeros([self.number_of_astrocytes, numberOfTimesteps], dtype=np.int8,
                                          order='F')  # col major since we always store data for one timestep
        self.animationFrame = 0
        self.amountOfAstrocytesActivatedInSimulation = 0
        self.tripartiteSynapseCalciums = np.zeros([numberOfNeurons, numberOfNeurons, self.number_of_astrocytes], dtype=np.single)
        self.calcium_accumulation_from_synapses = np.zeros([self.number_of_astrocytes], dtype=np.single)

    def configAstrocyteModel(self, numberOfNeurons):
        self.ip3 = np.zeros([numberOfNeurons, numberOfNeurons], dtype=np.single)
        self.astrocyteCa = np.zeros([numberOfNeurons, numberOfNeurons], dtype=np.single)
        self.releasingAstrocytes = np.zeros([numberOfNeurons, numberOfNeurons], dtype=np.int8)
        self.inverse_over_threshold_synapses = np.ones([numberOfNeurons, numberOfNeurons], dtype=np.int8)
        self.astrocyteInhibition = np.zeros(numberOfNeurons)
        self.chosenAstrocyte = [0, 0]

    def configPreSynapseModel(self, numberOfNeurons, p_oex0, p_oin0, syn_str, activeSynapses):
        self.w = syn_str
        self.x = self.Resources * activeSynapses  # resources is a number between 0, 1, activeSynapses a numberOfNeurons x numberOfNeurons Matrix
        self.u = self.Calcium * activeSynapses  # calcium is a number
        self.glu = self.AstrocyteGlutamate * activeSynapses  # AstrGlutamate is a number As well
        max_ex = p_oex0
        max_in = p_oin0
        self.scalingFactor = 1
        if max_ex <= max_in:
            self.scalingFactor *= max_in
        else:
            self.scalingFactor *= max_ex
        # Scaling W is between 0 and 1. Needs to be rescaled back in return value of presynapse model
        self.w = (1. / self.scalingFactor) * self.w
        # Auxiliary matrixes for model, reserving memory
        self.spikingMatrix = np.zeros([numberOfNeurons, numberOfNeurons], dtype=np.single)
        self.rr = np.zeros([numberOfNeurons, numberOfNeurons], dtype=np.single)
        self.aux = np.zeros([numberOfNeurons, numberOfNeurons], dtype=np.single)
        self.dataGatheringVariables = np.zeros(shape=[2, 13])

    """
    Save Astrocyte Data to csv
    Creates a separate file for dataGatheringVariables, activityAnimation, astrocyteConnections and astrocyteConnections
    """

    def saveAstrocyteData(self, directory, network):
        if self.usePreSynapse:
            pd.DataFrame(self.dataGatheringVariables).to_csv(
                directory + "/" + "AstroData_dataGatheringVariables" + ".csv", header=None, index=None)

        if self.use_astrocyte_network:
            pd.DataFrame(self.activityAnimation).to_csv(directory + "/" + "AstroData_activityAnimation" + ".csv",
                                                        header=None, index=None)
            pd.DataFrame(network.astrocyte_connections).to_csv(
                directory + "/" + "AstroData_astrocyteConnections" + ".csv", header=None, index=None)
            pd.DataFrame(network.astrocyte_locations).to_csv(directory + "/" + "AstroData_astrocyteLocation" + ".csv",
                                                           header=None, index=None)
        # print("Saved astrocyte data")

    def computeModifiedPreSynapseModel(self, spikeVector, numberOfNeurons, network, tInMS):
        rv = np.zeros([numberOfNeurons, numberOfNeurons])
        if np.sum(spikeVector) > 0:
            a = network.excitatory_synapses * self.alpha
            u_alpha = np.multiply(np.multiply(self.w, network.excitatory_synapses), 1 - self.glu) + np.multiply(a,
                                                                                                                self.glu)

            uMatrix = np.where(self.w < 0, self.w, 0) + u_alpha
            self.spikingMatrix = np.transpose(np.transpose(network.active_synapses) * spikeVector)

            self.aux = np.multiply(np.multiply(uMatrix, self.spikingMatrix), 1 - self.u)

            # swapping "negative u" values to positive. calcium builds up t
            self.aux = np.multiply(network.synapse_effect, self.aux)

            # Adding calcium sensor
            self.u = self.u + self.aux

            # Using information of amounts of calcium and which neurons are spiking together, with resources available for computing new weights for synapses.
            # Redefining matrix aux according to resources released in spikes.
            self.aux = np.multiply(self.u, self.x)

            # release from spiking synapses
            self.rr = np.multiply(self.aux, self.spikingMatrix)

            # matrix is resources released in synapse times the effect of the synapse and taking only those that are spiking this slot.
            rv = np.multiply(network.synapse_effect, self.rr) * self.scalingFactor

            # Removing resources from matrix x according to used resources.
            self.x = self.x - self.rr
        else:
            self.rr.fill(0)  # Fill probably faster than completley allocating memory a new

        if self.usePreSynapse:
            for i in range(0, tInMS):
                self.x = self.x + np.multiply(self.RegenRes * (1 - self.x),
                                              network.active_synapses)  # Potentiating effect
            self.u = self.CaRegen ** tInMS * np.multiply(self.u, network.active_synapses)

        if self.use_astrocyte:
            if self.use_astrocyte_network:
                rrToAstrocyte = np.multiply(self.rr, network.is_synapse_connect_to_astrocyte)
            else:
                rrToAstrocyte = np.multiply(self.rr, network.excitatory_synapses)

            self.astrocyteSimulation(rrToAstrocyte, network, numberOfNeurons, tInMS)
            self.glu *= self.GlutamateRemoval ** tInMS

        return rv

    # Do the astrocyte simulation
    def astrocyteSimulation(self, rrToAstrocyte, network, numberOfNeurons, tInMs):

        # There's IP3 decrease in time. Update accordingly.
        self.ip3 *= self.IPdegrading ** tInMs

        # There's also IP3 as well as calcium uptake.
        self.ip3 += np.multiply(1 - self.ip3, rrToAstrocyte)
        self.astrocyteCa += self.IPAccumulation * (self.ip3 - self.astrocyteCa)

        # Only astrocytes can release that have a Ca concentration larger than minimal required release level and that were below the release level before
        self.releasingAstrocytes = np.multiply(self.astrocyteCa > self.CaGlutamateReleaseLevel,
                                               self.inverse_over_threshold_synapses)
        self.inverse_over_threshold_synapses = (self.astrocyteCa <= self.CaGlutamateReleaseLevel)

        # Let the releasing begin...
        self.glu += self.AstrocyteReleaseAmount * np.multiply(1 - self.glu,
                                                              self.releasingAstrocytes)

        if self.use_astrocyte_network:
            self.simulateAstrocyteNetwork(network, numberOfNeurons)

    """
    Simulate astrocyte network 
    """

    def simulateAstrocyteNetwork(self, network, numberOfNeurons):
        # Broadcast (1xnumberOfAstrocytes) to (numberOfNeurons x numberOfNeurons x numberOfAstrocytes)
        # Update the amount of calcium in the tripartite synapses according to the calcium update computed in astrocyteSimulation()
        caEffect = np.broadcast_to(self.astrocyteCa[:, :, None], self.astrocyteCa.shape + (self.number_of_astrocytes,))
        self.tripartiteSynapseCalciums = np.multiply(caEffect, network.astrocyte_neuron_connections)

        self.runNetworkModel(network)

        # Broadcast to (numberOfNeurons x numberOfNeurons x numberOfAstrocytes)
        # Communication within astrocyte (TODO ? )
        tile = np.broadcast_to(self.getActivatedAstrocyteMatrix(),
                               (numberOfNeurons, numberOfNeurons, self.number_of_astrocytes))
        caWave = np.sum(np.multiply(tile, network.astrocyte_neuron_connections), axis=2).astype(bool)
        self.astrocyteInhibition = np.sum(caWave * self.astrocyteInhibitionStrength, axis=0)
        self.ip3 = np.multiply(self.ip3, ~(caWave)) + caWave

    """
    Basically executes the UAR Model
    Activates/Deactivates astrocytes where necessary
    """

    def runNetworkModel(self, network):

        # Check to how many not active astrocytes this astrocyte is connected to
        # PropagationEfficency is then inversly proportional to that value (if no inactive connections, assume "infinit number" of inactive connections)
        # TODO: Does that make sense?
        propagationEfficency = np.zeros([self.number_of_astrocytes])
        for i in range(0, self.number_of_astrocytes):
            if self.astrocyteState[i] == 1:
                connections = network.astrocyte_connections[i, :]
                numberOfInactiveConnections = np.sum(
                    np.multiply(connections, self.astrocyteState != 1))
                if numberOfInactiveConnections != 0:
                    propagationEfficency[i] = 1. / numberOfInactiveConnections

        # This is the total amount of astrocytic calcium in each astrocyte. (-> We are summing over all regions (aka connections) of one astrocyte)
        caAccumulationFromSynapses = np.sum(self.tripartiteSynapseCalciums, axis=(0, 1))
        np.seterr(divide='ignore', invalid='ignore')  # suppress...
        caAccumulationFromSynapses = np.divide(caAccumulationFromSynapses,
                                               network.nr_of_excitatory_conns_to_astrocyte) * self.SynapseCaEffect

        # In the paper, sumOfEfficencies is the Beta-Term
        ip3FluxIn = np.multiply(np.dot(np.squeeze(propagationEfficency), network.astrocyte_connections),
                                network.activation_threshold)
        sumOfEfficencies = ip3FluxIn + caAccumulationFromSynapses

        # Activate if sum of efficencies is > activationThreshold and if that's the case, with probability activationProbability
        activating = np.multiply(sumOfEfficencies, self.getInactivatedAstrocyteMatrix()) > network.activation_threshold
        activated = np.logical_and((self.getRandomArray() < self.ActivationProbability), activating)
        self.astrocyteState += activated

        self.amountOfAstrocytesActivatedInSimulation += np.sum(activated)

        # Shut down dependent on time...
        shuttingDown = np.logical_and(self.getActivatedAstrocyteMatrix(), self.getRandomArray() < self.ActivationTime)
        self.astrocyteState -= 2 * shuttingDown

        # Inactivate if necessary
        inactivating = np.logical_and(self.getRefractoryAstrocyteMatrix(), self.getRandomArray() < self.RefractoryTime)
        self.astrocyteState += inactivating

        # Save current activity state in activityAnimation
        self.activityAnimation[:, self.animationFrame] = np.squeeze(self.astrocyteState)
        self.animationFrame += 1

    """
    Compute AVG-Data that can later be used to compare different networks
    To save memory, only the data of one specific astrocyte is recorded. The astrocyte is chosen at the first timestep
    """

    def calculateAstrocyteData(self, network, rV, synStr, timeIndex):
        if self.use_astrocyte:
            self.avgAstroGlutamate = np.sum(self.glu) / network.nr_of_excitatory_synapses
            self.avgCaInAstrocytes = np.sum(self.astrocyteCa) / network.nr_of_excitatory_synapses
            if timeIndex == 0 and self.use_astrocyte_network:
                result = np.argwhere(network.is_synapse_connect_to_astrocyte == 1)
                ind = int(len(result[:, 0]) / 2)
                self.chosenAstrocyte = [result[ind, 0], result[ind, 1]]

            self.singSynstr = rV[self.chosenAstrocyte[0], self.chosenAstrocyte[1]]
            self.singCal = self.u[self.chosenAstrocyte[0], self.chosenAstrocyte[1]]
            self.singRR = self.rr[self.chosenAstrocyte[0], self.chosenAstrocyte[1]]
            self.singIp3 = self.ip3[self.chosenAstrocyte[0], self.chosenAstrocyte[1]]
            self.singCaInAstrocytes = self.astrocyteCa[self.chosenAstrocyte[0], self.chosenAstrocyte[1]]
            self.singRelease = self.releasingAstrocytes[self.chosenAstrocyte[0], self.chosenAstrocyte[1]]
            self.singAstroGlutamate = self.glu[self.chosenAstrocyte[0], self.chosenAstrocyte[1]]
        if self.usePreSynapse:
            self.avgExInWeight = np.sum(np.multiply(np.multiply(self.spikingMatrix, synStr),
                                                    network.excitatory_synapses)) / network.nr_of_excitatory_synapses
            self.avgInWeight = -1 * np.sum(np.multiply(np.multiply(self.spikingMatrix, synStr),
                                                       network.inhibitory_synapses)) / network.nr_of_inhibitory_synapses

        self.gatherAstrocyteData()

    # take computed avgs and save them in dataGatheringVariable. That way, they can later be saved easily
    def gatherAstrocyteData(self):
        dataArray = np.zeros(13)
        if self.use_astrocyte:
            dataArray = [self.avgAstroGlutamate, 0, 0, 0, 0, self.avgCaInAstrocytes, self.singSynstr, self.singCal,
                         self.singRR,
                         self.singIp3, self.singCaInAstrocytes, self.singRelease, self.singAstroGlutamate]
        if self.usePreSynapse:
            dataArray[3] = self.avgExInWeight
            dataArray[4] = self.avgInWeight

        self.dataGatheringVariables = np.append(self.dataGatheringVariables, [dataArray], axis=0)

    def getRandomArray(self):
        return np.random.rand(self.number_of_astrocytes)

    """
    Returns array that indicates where astrocytes are inactivated
    """

    def getInactivatedAstrocyteMatrix(self):
        return self.astrocyteState == 0

    """
    Returns array that indicates where astrocyte state is active
    """

    def getActivatedAstrocyteMatrix(self):
        return self.astrocyteState == 1

    """
    Returns array that indicates where astrocyte state is refractory
    Refractory means that the astrocyte does not communicate with the other astrocytes via calcium waves
    """

    def getRefractoryAstrocyteMatrix(self):
        return self.astrocyteState == -1
