# -*- coding: utf-8 -*-
"""
Keeps track of parameters that are specific to neurons
"""
import numpy as np


# Set Neuron parameters
def config_neuron():
    nr_excitatory_neurons = 200  # Number of excitatory neurons
    nr_inhibitory_neurons = 50  # Number of inhibitory neurons
    c_k = np.array([0.02, 0.01, 0.02])  # Lower and upper boundary of random number for basic activity
    # Lower and upper boundary of random number for synaptic strength (all positive numbers),
    oex = 0.7  # excitatory, to other neurons, group 0
    oin = 0.7  # inhibitory, to other neurons, group 0
    sensitivityMultiplier = 0.1  # Factor for random variable

    #### Create object of class neuron_params containing all the parameters listed above
    return neuron_params(nr_excitatory_neurons, nr_inhibitory_neurons, c_k, oex, oin, sensitivityMultiplier)


"""
Class that keeps track of parameters specific to neurons
"""
class neuron_params:

    def __init__(self, nr_excitatory_neurons, nr_inhibitory_neurons, c_k, synStr_oex0, synStr_oin0, sensitivityMultiplier):
        self.nr_excitatory_neurons = nr_excitatory_neurons
        self.nr_inhibitory_neurons = nr_inhibitory_neurons
        self.c_k = c_k
        self.oex = synStr_oex0
        self.oin = synStr_oin0
        self.sensitivityMultiplier = sensitivityMultiplier
        self.number = nr_excitatory_neurons + nr_inhibitory_neurons

    def setSynStr_bruteForce(self):
        self.c = np.arange(self.c_k[0], self.c_k[2] + self.c_k[1], self.c_k[1])

    def setNeuronConfig(self, p_c, base_network, base_basic_activity):
        # Basic activity of all neurons which is a random number between 0 and an  upper boundary from a triangular distribution
        self.c_matrix = base_basic_activity * p_c

        # Excitatory Suand inhibitory synaptic strengths is a random number between 0 and an upper boundary from a triangular distribution
        synStr_ = np.zeros([self.nr_excitatory_neurons + self.nr_inhibitory_neurons, self.number], dtype=np.single)
        synStr_[:self.nr_excitatory_neurons, :] += base_network[:self.nr_excitatory_neurons, :] * self.oex
        synStr_[self.nr_excitatory_neurons:, :] += base_network[self.nr_excitatory_neurons:, :] * self.oin
        np.fill_diagonal(synStr_, 0)  # Autapses
        self.synStr = synStr_
