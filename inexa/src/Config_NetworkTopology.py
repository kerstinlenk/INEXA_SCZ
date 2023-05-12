import os

import numpy as np
import pandas as pd

# Set Neuron parameters
from src.Functions import uniform_3D, distance_coupling, complex_connect, create_gaussian_connections
from src.usability_functions import read_csv

SEPARATOR = ",|;"

BASE_NETWORK_FNAME = "/BaseNetwork.csv"

ASTROCYTE_NEURON_CONNECTIONS_FNAME = "/AstrocyteNeuronConnections.csv"

ASTROCYTE_CONNECTIONS_FNAME = "/AstrocyteConnections.csv"

ASTROCYTE_NETWORK_TOPOLOGY_FNAME = "/AstrocyteNetworkTopology.csv"

NEURON_NETWORK_TOPOLOGY_FNAME = "/NeuronNetworkTopology.csv"


def config_networkTopology(path_to_neuron_network, path_to_astro_network):
    return NetworkTopology(path_to_neuron_network, path_to_astro_network)


class NetworkTopology:
    culture_space_dim = np.array([750, 750, 10])  # [\mu m]

    # [\mu m]
    min_neuron_distance = 10
    neuron_connectivity_std = 200

    # [\mu m], min distance between 2 astrocytes
    min_astrocyte_distance = 30
    # Maximum distance between 2 (connected[?]) astrocyte in micrometers
    max_astrocyte_distance = 100
    # Absolute maximum distance after which the astrocyte will not connect to synapses
    max_astrocyte_reach_distance = 70

    # Standard deviation of astrocyte connectivity to neuron
    astrocyte_neuron_connectivity_std = 150

    create_new_neuron_topology = False
    create_new_astrocyte_topology = False

    path_to_neuron_networks = ""
    path_to_astrocyte_networks = ""

    # Is De Pitta presynapse simulator on? boolean
    pre_synapse = True  # Not taking any responsibility for False

    # Is astrocyte simulator on? boolean
    simulate_astrocyte = True  # Not taking any responsibility for False

    # Is astrocyte network simulator on? boolean
    simulate_astrocyte_network = True  # Not taking any responsibility for False

    def __init__(self, path_to_neuron_networks, path_to_astrocyte_networks):
        self.path_to_neuron_networks = path_to_neuron_networks
        self.path_to_astrocyte_network = path_to_astrocyte_networks

        self.base_activity = np.array([])
        self.is_synapse_connect_to_astrocyte = np.array([])
        self.neuron_locations = np.array([])
        self.base_network_activity = np.array([])

        self.astrocyte_locations = np.array([])
        self.astrocyte_connections = np.array([])

        self.activation_threshold = 0

        self.astrocyte_neuron_connections = np.array([])

    def create_network_topologies(self, neuron, astrocyte):
        """
        Create network topologies if respective flags (create_new_neuron_topology, create_new_astrocyte_topology) are set to True
        Connect Neuron/Neuron, Neuron/Astrocyte and Astrocyte/Astrocyte

        Neuron-Neuron connections are gaussian based on distance
        Astrocyte-Neuron are gaussian based on distance until it reaches a limiter
        Astrocyte-Astrocyte definitly connect within set radius
        """

        # --- Basic activity of all neurons --- which is a random number between 0 and an upper boundary from a triangular distribution
        nr_of_neurons = neuron.number

        self.base_activity = np.random.triangular(0, 0.5, 1, size=nr_of_neurons)
        self.is_synapse_connect_to_astrocyte = np.zeros([nr_of_neurons, nr_of_neurons], dtype=np.bool)

        if self.create_new_neuron_topology:
            # create randomly uniform 3D Locations within the culture space. Locations must be at least self.min_neuron_distance apart.
            self.neuron_locations = uniform_3D(nr_of_neurons, self.culture_space_dim, self.min_neuron_distance)

            neuronConnections = create_gaussian_connections(self.neuron_locations, self.neuron_connectivity_std, cut_off=1e5)

            #  Excitatory and Inhibitory synaptic strengths is a random number between 0 and an upper boundary from a triangular distribution
            self.base_network_activity = np.zeros((nr_of_neurons, nr_of_neurons))
            self.base_network_activity[:neuron.nr_excitatory_neurons, :] = np.random.triangular(0, 0.5, 1,
                                                                                                size=(neuron.nr_excitatory_neurons, neuron.number))
            self.base_network_activity[neuron.nr_excitatory_neurons:, :] = np.random.triangular(-1, -0.5, 0,
                                                                                                size=(neuron.nr_inhibitory_neurons, neuron.number))
            self.base_network_activity[neuronConnections == 0] = 0  # non-active neuron connections = 0
            np.fill_diagonal(self.base_network_activity, 0)
        else:
            self.load_neuron_network()

        # create astrocyte network
        # Each column is connected to each astrocyte.
        if astrocyte.use_astrocyte_network and self.create_new_astrocyte_topology:
            self.create_astrocyte_topology(astrocyte, nr_of_neurons)

    def create_astrocyte_topology(self, astrocyte, nr_of_neurons):
        # create astrocytes
        self.astrocyte_locations = uniform_3D(astrocyte.number_of_astrocytes, self.culture_space_dim, self.min_astrocyte_distance)
        # Connecting astrocytes that ended up close enough to each other.
        self.astrocyte_connections = distance_coupling(self.astrocyte_locations, self.max_astrocyte_distance)
        number_of_connections = np.sum(self.astrocyte_connections, axis=0)
        self.activation_threshold = astrocyte.slope * number_of_connections + astrocyte.intercept
        # Connect the two networks
        self.astrocyte_neuron_connections = complex_connect(self.neuron_locations, self.base_network_activity,
                                                            self.astrocyte_locations,
                                                            self.astrocyte_neuron_connectivity_std,
                                                            self.max_astrocyte_reach_distance)
        self.init_synapses_connected_to_astrocyte(nr_of_neurons)

    def init_synapses_connected_to_astrocyte(self, nr_of_neurons):
        for i in range(0, nr_of_neurons):
            for ii in range(0, nr_of_neurons):
                self.is_synapse_connect_to_astrocyte[i, ii] = np.sum(self.astrocyte_neuron_connections[i, ii, :])

    def collect_useful_values(self):
        self.active_synapses = np.ascontiguousarray(self.base_network_activity != 0, dtype=np.int8)
        self.excitatory_synapses = np.ascontiguousarray(self.base_network_activity > 0, dtype=np.int8)
        self.inhibitory_synapses = np.ascontiguousarray(self.base_network_activity < 0, dtype=np.int8)

        # amount of excitatory and inhibtory synapses
        self.nr_of_inhibitory_synapses = np.sum(self.inhibitory_synapses)
        self.nr_of_excitatory_synapses = np.sum(self.excitatory_synapses)

        # synapse effect: Has values 1 for exitatory, 0 for not  connected and -1 for negatives.
        self.synapse_effect = self.excitatory_synapses - self.inhibitory_synapses  # Counting how many excitatory connections ended up to each astrocyte

        if self.simulate_astrocyte_network:
            # self.nr_of_excitatory_conns_to_astrocyte = np.sum(self.astrocyte_neuron_connections, axis=(0, 1))
            self.nr_of_excitatory_conns_to_astrocyte = sum(sum(self.astrocyte_neuron_connections))
            # Counting which synapses are connected to an astrocyte to begin with
            self.nr_of_astrocytes_connected_to_synapse = np.sum(self.astrocyte_neuron_connections, axis=2)
        self.nr_of_astrocyte_connections = np.sum(self.astrocyte_connections, axis=0)

    def save_network_topology(self, neuron, astrocyte, directory):
        folderName = directory
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        # Neuron Network
        pd.DataFrame(self.neuron_locations).to_csv(folderName + NEURON_NETWORK_TOPOLOGY_FNAME, header=False, index=False)
        # Astrocyte Network
        if astrocyte.use_astrocyte_network:
            pd.DataFrame(self.astrocyte_locations).to_csv(folderName + ASTROCYTE_NETWORK_TOPOLOGY_FNAME, header=False, index=False)
            pd.DataFrame(self.astrocyte_connections).to_csv(folderName + ASTROCYTE_CONNECTIONS_FNAME, header=False, index=False)
            pd.DataFrame(self.astrocyte_neuron_connections.T.reshape(-1, neuron.number)).to_csv(folderName + ASTROCYTE_NEURON_CONNECTIONS_FNAME,
                                                                                                header=False, index=False)
        pd.DataFrame(self.base_network_activity).to_csv(folderName + BASE_NETWORK_FNAME, header=False, index=False)

    def load_network_topologies(self, neuron, astrocyte):  #
        """
        Load network topologies and their connectivity matrices. Initialize memory etc.

        """
        self.is_synapse_connect_to_astrocyte = np.zeros([neuron.number, neuron.number])
        self.load_neuron_network()
        # import Astrocyte data

        if self.create_new_astrocyte_topology and astrocyte.use_astrocyte_network:
            self.create_astrocyte_topology(astrocyte, neuron.number)
        elif astrocyte.use_astrocyte_network:
            self.load_astrocyte_network(astrocyte, neuron)

    def load_astrocyte_network(self, astrocyte, neuron):
        pathANT = self.path_to_astrocyte_network + ASTROCYTE_NETWORK_TOPOLOGY_FNAME
        pathANC = self.path_to_astrocyte_network + ASTROCYTE_NEURON_CONNECTIONS_FNAME
        pathAC = self.path_to_astrocyte_network + ASTROCYTE_CONNECTIONS_FNAME

        self.astrocyte_locations = read_csv(pathANT, sep=SEPARATOR, header=None).dropna(how='all', axis=1).to_numpy(dtype=np.short)
        self.astrocyte_connections = read_csv(pathAC, sep=SEPARATOR, header=None).dropna(how='all', axis=1).to_numpy(np.short)

        temp = read_csv(pathANC, sep=SEPARATOR, header=None).dropna(how='all', axis=1).to_numpy()
        self.astrocyte_neuron_connections = np.zeros((neuron.number, neuron.number, astrocyte.number_of_astrocytes), dtype=np.int8)

        # do some weird magic to reshape the temp array. honestly, there's probably an easier way to do this.
        line = 0
        for iii in range(astrocyte.number_of_astrocytes):
            for i in range(neuron.number):
                for ii in range(neuron.number):
                    self.astrocyte_neuron_connections[i, ii, iii] = temp[line, ii]
                line += 1

        self.collect_useful_values()
        self.init_synapses_connected_to_astrocyte(neuron.number)
        self.activation_threshold = astrocyte.slope * self.nr_of_astrocyte_connections + astrocyte.intercept

    def load_neuron_network(self):
        pathNNT = self.path_to_neuron_networks + NEURON_NETWORK_TOPOLOGY_FNAME
        pathBN = self.path_to_neuron_networks + BASE_NETWORK_FNAME
        self.neuron_locations = read_csv(pathNNT, sep=SEPARATOR, header=None).dropna(how='all', axis=1).to_numpy()
        self.base_network_activity = read_csv(pathBN, sep=SEPARATOR, header=None).dropna(how='all', axis=1).to_numpy()
