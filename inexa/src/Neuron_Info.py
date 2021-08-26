import numpy as np
import pandas as pd

class NeuronInfo:
    # c is numNeuronsx1, numberOfNeurons, t, sensitivityMultiplier are numbers, synStr is numberOfNeurons x numberOfNeurons,
    def __init__(self, numberOfNeurons, t, synStr, c, sensitivityMultiplier):
        self.numberOfNeurons = numberOfNeurons
        self.t = t
        self.synStr = synStr
        self.c = c
        self.sensitivityMultiplier = sensitivityMultiplier
