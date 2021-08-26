# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:02:12 2020

@author: hlind
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
import scipy.stats as stats
import os
import scipy


def complex_connect(N, BaseNetwork, A, ANconSTD, MaxAstrReachDist):
    """
    Connects incoming connections to a neuron to nearest
    astrocyte/astrocytes. Counting probability of each synapse to be
    connected to an astrocyte and dividing synapses between astrocytes by
    their probability.
    """
    numNeuron = np.size(N, axis=1)
    numAstr = np.size(A, axis=1)
    ANcon = np.zeros((numNeuron, numNeuron, numAstr))
    ANconProb = np.zeros((numNeuron, numAstr))
    ExcSynapse = np.array(BaseNetwork > 0, dtype=int)
    #    Calculating probability of each astrocyte to connect to synapses of a neuron.
    for n in np.arange(numNeuron):
        for a in np.arange(numAstr):
            d = np.linalg.norm(N[:, n] - A[:, a])
            if d < MaxAstrReachDist:
                ANconProb[n, a] = ANconSTD * np.sqrt(2 * np.pi) * stats.norm.pdf(d, loc=0, scale=ANconSTD)
    #   Finding out which astrocyte binds to which synapse
    for n1 in np.arange(numNeuron):
        candidateOrder = np.argsort(ANconProb[n1, :])[::-1]  # descending order
        candidateStrength = ANconProb[n1, candidateOrder]

        #        print("\n shape_ANconProb \n =",np.shape(ANconProb))
        #        print("\n ANconProb \n =",ANconProb)
        #        print("\n candidateOrder= \n",candidateOrder)
        #        print("\n shape_candidateOrder \n =",np.shape(candidateOrder))
        #        print("\n candidateStrength= \n",candidateStrength)
        #        print("\n shape_candidateStrength \n =",np.shape(candidateStrength))

        #    % When going through all the neurons if this incoming
        #    % connection is excitatory we try to connect an astrocyte to it
        for n2 in np.arange(numNeuron):
            if ExcSynapse[n2, n1] == 1:
                candidateNumber = 0  # start index=0
                NotAllocated = 1
                #                print(candidateStrength[candidateNumber])
                #                print(type(candidateStrength[candidateNumber]))
                #                print(type(0))
                #                print(type(0.0))
                #                print(candidateStrength[candidateNumber] > 0.0)
                while ((candidateNumber <= numAstr - 1) &
                       (candidateStrength[candidateNumber] > 0.0) &
                       (NotAllocated == 1)):
                    if candidateStrength[candidateNumber] > np.random.rand():
                        # connect to astrocyte
                        ANcon[n2, n1, candidateOrder[candidateNumber]] = 1
                        break
                    candidateNumber += 1

    return ANcon


def uniform_3D(N, culture_space, overlap_dist):  # copied to Functions
    """
    Function description: to be added

    """

    neurons = np.zeros((3, N))
    cult = culture_space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # might want to change the grid proportions to avoid misleading graphs

    for i in np.arange(0, N):
        tmp = np.random.uniform(0, 1, (3,)) * cult
        #            print(neurons[:,i].shape)
        for j in np.arange(0, i):
            its = 0;
            while (np.linalg.norm(neurons[:, j] - tmp) < overlap_dist):
                tmp = np.random.uniform(0, 1, (3,)) * cult
                its += 1
                assert (its < 10e4), "Small Space Error: failed to place object with sufficient distance after " + str(
                    its) + " iterations"
        neurons[:, i] = tmp
    #    print("\n neurons uniform \n",neurons)
    ax.scatter(neurons[0, :], neurons[1, :], neurons[2, :], color="black", marker="x")
    plt.show()

    return neurons


def distance_coupling(coord, threshold):
    """
    Function description: to be added:
    Briefly: Deterministic coupling by distance
    """
    M = np.size(coord, axis=1)
    nn_dense = np.zeros((M, M))

    for i in np.arange(0, M):
        for j in np.arange(0, M):
            # condition for connection:
            #                print(np.linalg.norm(neurons[:,i]-neurons[:,j]))
            if (np.linalg.norm(coord[:, i] - coord[:, j]) < threshold):
                nn_dense[i, j] = 1

            # do we want loop connections?
    #        nn_dense_unif = nn_dense_unif - np.identity(N**3)
    #    print("\n Uniform Connectivity Matrix\n",nn_dense)
    # Might be nice to plot the connections, but could make grid difficult to interpret ...

    return nn_dense


def createGaussianConnections(coord, std, cut_off):
    """
    Function description: to be added:
    Briefly: Deterministic coupling by distance

    Still want to add possibility of cut_off = None if we have no threshold
    """
    M = np.size(coord, axis=1)
    nn_dense = np.zeros((M, M))

    for i in np.arange(0, M):
        for j in np.arange(0, M):
            d = np.linalg.norm(coord[:, i] - coord[:, j])
            if (d > cut_off):
                nn_dense[i, j] = 0
            else:
                gauss_prob_connect = std * np.sqrt(2 * np.pi) * stats.norm.pdf(d, loc=0, scale=std)
                nn_dense[i, j] = int(gauss_prob_connect > np.random.uniform(0, 1))

    #    print("\n Uniform Connectivity Matrix\n",nn_dense)
    return nn_dense


# Output is lastSpike vector as used in main funciton of INEX. k is the time from the beginning in ms+ 1 and infoNeurons contains all the required information aout the neuronal network
def generateNewSpike(k, neuronInfo, lastSpike, astrocyteInhibition):
    if k == 0:
        lam = neuronInfo.c  # lam is supposed to be lambda
    else:
        spikingWeightsMatrix = np.zeros([neuronInfo.numberOfNeurons, neuronInfo.numberOfNeurons])

        for i in range(0, neuronInfo.numberOfNeurons):
            spikingWeightsMatrix[:, i] = np.multiply(neuronInfo.synStr[:, i], lastSpike)

        sumSyn = np.sum(spikingWeightsMatrix, axis=0) + astrocyteInhibition
        lam = sumSyn + neuronInfo.c
        lam = np.where(lam > 0, lam, 0)
        # print("k is larger than one")

    p = np.multiply(np.exp(-lam * neuronInfo.t), lam * neuronInfo.t)

    x = neuronInfo.sensitivityMultiplier * np.random.random(np.shape(p))

    lastSpike = x < p

    return lastSpike


def saveParameters(a, c, directory, zz1, zz2, zz3):
    # print("Save Parameters for Syn Strength and Basic Activity")
    file = open(directory + "/" + "Parameters_" + zz1 + "_" + zz2 + "_" + zz3, "w+")
    np.set_printoptions(threshold=np.inf)
    file.write("Synaptic Strength Parameters: ")
    for entry in a.flatten():
        file.write(str(entry) + "\n")

    file.write(" Basic Activity Parameters: ")
    for i in c.flatten():
        file.write(str(i) + "\n")

    file.close()


def getCMAinfo(spikeTrain, ids):
    isis = []
    for entry in spikeTrain:
        isis.append(np.diff(entry))

    v = np.hstack(isis[:])
    print(v)
    return 0, 0


def filterSpikeTrains(spikeTrainsUnfiltered):
    # Basically create arrays that do not include the 0 entries for whatever reason... TODO: Talk with Kerstin :)

    numberOfNeurons = np.shape(spikeTrainsUnfiltered)[1]  # -1 for ID row...
    numberOfTimesteps = np.shape(spikeTrainsUnfiltered)[0]

    spikeTrainsFiltered = [[] for i in range(0, numberOfNeurons)]  # For each neuron, create an empty arry

    numberOfNonZeroEntries = 0
    for timestep in range(numberOfTimesteps):  # Row 0 is the ID...
        for neuronIndex in range(0, numberOfNeurons):
            neuronSpikeEntry = spikeTrainsUnfiltered[timestep, neuronIndex]
            if neuronSpikeEntry != 0:
                spikeTrainsFiltered[neuronIndex].append(neuronSpikeEntry * 1000)
                numberOfNonZeroEntries += 1

    return spikeTrainsFiltered, numberOfNonZeroEntries


# TODO LFR: What exactly?!?
def skw2alpha(skw):
    if skw < 1:
        return 1, 0.5
    if skw < 4:
        return 0.7, 0.5
    if skw < 9:
        return 0.5, 0.3
    else:
        return 0.3, 0.1


def detectBurstNoRound(spikeSequence, burstThreshold, tailThreshold, minNumberOfSpikes=1):
    numberOfSpikes = len(spikeSequence)

    isSpikePartOfABurst = np.zeros([numberOfSpikes], dtype=np.int8)
    spikeType = np.zeros([numberOfSpikes],
                         dtype=np.int8)  # each spike gets a type. init all as type 0. type is an int \in (0,1,2)

    spikeDiff = np.diff(spikeSequence)
    burstSpikeIndices = np.argwhere(spikeDiff < burstThreshold)
    isSpikePartOfABurst[burstSpikeIndices] = 1
    isSpikePartOfABurst[burstSpikeIndices + 1] = 1

    # detect tail bursts
    tailSpikeIndices = np.argwhere((spikeDiff < tailThreshold))
    spikeType[tailSpikeIndices] = 2
    spikeType[tailSpikeIndices + 1] = 2
    spikeType = np.where(isSpikePartOfABurst == 1, 1, spikeType)  # This ensures that spikes aren't overwritten by tails

    # Filter out bursts with less than "minNumberOfSpikes" in a row
    breakPointStartsArray = np.array([])
    breakPointEndsArray = np.array([])
    numberOfSpikesInBursts = np.array([])

    burstSpikes = np.squeeze(np.argwhere(spikeType == 1))
    if len(burstSpikes.shape) != 0:
        breakPoints = np.squeeze(np.argwhere(np.diff(spikeSequence[burstSpikes]) > tailThreshold))

        if len(breakPoints.shape) != 0 and len(burstSpikes.shape) > 0 and len(burstSpikes) > 0:
            breakPointStartsArray = np.concatenate(([burstSpikes[0]], burstSpikes[breakPoints + 1]))
            breakPointEndsArray = np.append(burstSpikes[breakPoints], burstSpikes[-1])
            numberOfSpikesInBursts = breakPointEndsArray - breakPointStartsArray + 1


    burstsWithTooFewSpikes = np.argwhere(numberOfSpikesInBursts < minNumberOfSpikes)
    for i in range(0, len(burstsWithTooFewSpikes)):
        spikeType[breakPointStartsArray[burstsWithTooFewSpikes[i][0]]:breakPointEndsArray[burstsWithTooFewSpikes[i][0]]+1] = 0

    mergeHappened = True
    while mergeHappened:
        mergeHappened = False
        tailSpikeIndices = np.squeeze(np.argwhere(spikeType == 2))

        if len(tailSpikeIndices.shape) == 0:
            continue

        for tailSpikeIndex in tailSpikeIndices:
            isPartOfSpikeForward = False
            isPartOfSpikeBackward = False
            if tailSpikeIndex != 0:
                isPartOfSpikeBackward = spikeType[tailSpikeIndex - 1] == 1 and spikeSequence[tailSpikeIndex] - \
                                        spikeSequence[tailSpikeIndex - 1] < tailThreshold
            if tailSpikeIndex != numberOfSpikes - 1:
                isPartOfSpikeForward = spikeType[tailSpikeIndex + 1] == 1 and spikeSequence[tailSpikeIndex + 1] - \
                                       spikeSequence[tailSpikeIndex] < tailThreshold

            if isPartOfSpikeForward or isPartOfSpikeBackward:
                spikeType[tailSpikeIndex] = 1
                mergeHappened = True

    # As in matlab
    burstSpikes = np.squeeze(np.argwhere(spikeType == 1))
    if len(burstSpikes.shape) == 0 or len(burstSpikes) == 0:
        return spikeType, [], [], [], []
    breakPoints = np.squeeze(np.argwhere(np.diff(spikeSequence[burstSpikes]) > tailThreshold))
    if len(breakPoints.shape) == 0:
        return spikeType, [], [], [], []
    breakPointStartsArray = np.concatenate(([burstSpikes[0]], burstSpikes[breakPoints + 1]))
    breakPointEndsArray = np.append(burstSpikes[breakPoints], burstSpikes[-1])

    burstStarts = spikeSequence[breakPointStartsArray]
    burstEnds = spikeSequence[breakPointEndsArray]
    numberOfSpikesInBursts = breakPointEndsArray - breakPointStartsArray + 1
    burstDurations = burstEnds - burstStarts

    burstNumbers = np.zeros([len(spikeSequence)])
    for breakPointIndex in range(0, len(breakPointStartsArray)):
        burstNumbers[
        breakPointStartsArray[breakPointIndex]:breakPointEndsArray[breakPointIndex] + 1] = breakPointIndex + 1

    return burstNumbers, burstStarts, burstEnds, burstDurations, numberOfSpikesInBursts


def calculateThresholdForBurstDetection(cmaCurve, burstAlpha, tailAlpha):
    reversedCMA = cmaCurve[::-1]
    lastMaxIndex = np.shape(cmaCurve)[0] - np.argmax(reversedCMA) - 1
    maxCma = np.max(cmaCurve)

    burstDistance = np.absolute(np.subtract(cmaCurve[lastMaxIndex:], burstAlpha * maxCma))
    minIndexInBirst = np.shape(burstDistance)[0] - np.argmin(burstDistance[::-1]) - 1
    burstThreshold = lastMaxIndex + minIndexInBirst + 1  # + 1 to get the same result as in matlab! due to different indexing.

    tailDistance = np.absolute(np.subtract(cmaCurve[burstThreshold - 1:], tailAlpha * maxCma))
    minIndexInTail = np.shape(tailDistance)[0] - np.argmin(tailDistance[::-1]) - 1
    tailThreshold = burstThreshold + minIndexInTail + 1

    return burstThreshold, tailThreshold


def computeBursts(spikeTrains, ids, simulationTimeInMs):  # spike trains as they are saved to .csv!

    simulationTimeInMinutes = simulationTimeInMs / 1000 / 60
    spikeTrainFiltered, numberOfNonZeroEntries = filterSpikeTrains(spikeTrains)

    concatenatedChangesInSpike = np.zeros(shape=[numberOfNonZeroEntries - len(spikeTrainFiltered)])
    counter = 0
    for neuronSpikeTrain in spikeTrainFiltered:
        concatenatedChangesInSpike[counter:counter + len(neuronSpikeTrain) - 1] = np.diff(
            neuronSpikeTrain)  # Ignore the first value, we don't get that one in matlab either?
        counter += len(neuronSpikeTrain) - 1

    skewnessOfSpikeTrains = scipy.stats.skew(concatenatedChangesInSpike)
    burstAlpha, tailAlpha = skw2alpha(skewnessOfSpikeTrains)

    # binRanges.append(np.max(concatenatedChangesInSpike))
    concatenatedChangeInSpikeHistogram, ind = np.histogram(concatenatedChangesInSpike, [i for i in range(0,
                                                                                                         20001)])  # Attention: NOT EXACTLY THE SAME RESULT AS IN MATLB
    cumsum = np.cumsum(concatenatedChangeInSpikeHistogram)  # cumulative sum (i[0], i[0] + i[1], i[0]+i[1]+i[2], ...)
    cmaCurve = np.divide(cumsum, [i for i in range(1, 20001)])
    burstThreshold, tailThreshold = calculateThresholdForBurstDetection(cmaCurve, burstAlpha, tailAlpha)

    finalData = np.zeros([len(spikeTrainFiltered), 7])
    counter = 0
    for neuronSpikeTrain in spikeTrainFiltered:
        burstNumbers, burstStarts, burstEnds, burstDurations, numberOfSpikesInBursts = detectBurstNoRound(
            np.array(neuronSpikeTrain), burstThreshold, tailThreshold)

        # spikesIncludedBursts = np.squeeze(np.argwhere(burstNumbers > 0))

        spikeCount = len(neuronSpikeTrain)
        spikeRate = spikeCount / simulationTimeInMinutes
        burstRate = len(burstStarts) / simulationTimeInMinutes
        burstDuration = np.mean(burstDurations) if len(burstDurations) > 0 else 0
        spikesInBurst = np.mean(numberOfSpikesInBursts)
        burstSpikeRatio = np.sum(numberOfSpikesInBursts) / spikeCount
        # burstSpikes = spikesIncludedBursts
        finalData[counter] = [spikeCount, spikeRate, burstRate, burstDuration, spikesInBurst, burstSpikeRatio,
                              0]  # TODO sollen wir nur Komponente 1,2,3 hier berechnen, da wir nur diese später verwenden?
        counter += 1

    return finalData  # spike rate at index 1, burstRate at index 2, burst duration at index 3


def computeStatisticalBurstData(spikeTrains, simulationTimeInMs):  # spike trains as they are saved to .csv!
    results = computeBursts(spikeTrains, [], simulationTimeInMs)
    return np.mean(results[:, 1]), np.std(results[:, 1]), np.mean(results[:, 2]), np.std(results[:, 2]), np.mean(
        results[:, 3]), np.std(results[:, 3])  # 210117,flo: modified this to make handling data easier


def analyzeAstrocyteActivity(activityAnimation):
    nrAstrocytes, nrTimesteps = np.shape(activityAnimation)
    manipulatedActivityTrain = np.zeros(shape=(nrAstrocytes, nrTimesteps + 2))
    # add 0 at beginning and at end to avoid having to handle special cases
    manipulatedActivityTrain[:, 1:nrTimesteps + 1] = np.where(activityAnimation == -1, 0, activityAnimation)
    changeInState = np.diff(manipulatedActivityTrain)

    activityLength = []
    for i in range(nrAstrocytes):
        astrocyteDiff = changeInState[i, :]
        indices = astrocyteDiff.nonzero()[0]

        activityLengthForThisAstrocyte = [indices[j + 1] - indices[j] for j in range(0, len(indices)-1, 2)]
        activityLength = np.concatenate((activityLength, activityLengthForThisAstrocyte))
    nrActivationsPerAstrocyte = np.count_nonzero(changeInState > 0, axis=1)

    return activityLength, nrActivationsPerAstrocyte