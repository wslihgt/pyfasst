# copyright (C) 2010 Jean-Louis Durrieu

from numpy import arange, zeros, array, argmax, vstack, amax, ones, outer 

def viterbiTracking(logDensity, logPriorDensities, logTransitionMatrix,
                    verbose=False):
    """
    Naive implementation of the Viterbi algorithm:
    this is a bit slow, consider using viterbiTrackingArray instead.

    bestStatePath = viterbiTracking(logDensity, logPriorDensities,
                                    logTransitionMatrix, verbose=False)

    viterbiTracking returns the best path through matrix logDensity,
    assuming that logDensity contains the likelihood of the observation
    sequence, conditionally upon the hidden states. A hidden Markov
    model (HMM) is assumed, with prior probabilities for the states
    given by logPriorDensities, and transition probabilities given
    by the matrix logTransitionMatrix. More precisely:
    Inputs:
        logDensity is a S x N ndarray, where S is the number of hidden
                      states and N is the number of frames of the
                      observed signal. The element at row s and
                      column n contains the conditional likelihood
                      of the signal at frame n, conditionally upon
                      state s.
        logPrioroDensities is a ndarray of size S, containing the prior
                           probabilities of the hidden states of the HMM.
        logTransitionMatrix is a S x S ndarray containing the transition
                            probabilities: at row s and column t, it
                            contains the probability of having state t
                            after state s.
        verbose defines whether to display evolution information or not.
                Default is False.

    Outputs:
        bestStatePath is the sequence of best states, assuming the HMM
                      with the given parameters.
    """
    numberOfStates, numberOfFrames = logDensity.shape

    cumulativeProbability = zeros([numberOfStates, numberOfFrames])
    antecedents = zeros([numberOfStates, numberOfFrames])

    for state in arange(numberOfStates):
        antecedents[state, 0] = -1
        cumulativeProbability[state, 0] = logPriorDensities[state] \
                                          + logDensity[state, 0]
    
    for n in arange(1, numberOfFrames):
        if verbose:
            print "frame number ", n, "over ", numberOfFrames
        for state in arange(numberOfStates):
            if verbose:
                print "     state number ",state, " over ", numberOfStates
            cumulativeProbability[state, n] \
                                     = cumulativeProbability[0, n - 1] \
                                       + logTransitionMatrix[0, state]
            antecedents[state, n] = 0
            for state_ in arange(1, numberOfStates):
                if verbose:
                    print "          state number ",
                    print state_, " over ", numberOfStates
                tempCumProba = cumulativeProbability[state_, n - 1] \
                               + logTransitionMatrix[state_, state]
                if (tempCumProba > cumulativeProbability[state, n]):
                    cumulativeProbability[state, n] = tempCumProba
                    antecedents[state, n] = state_
            cumulativeProbability[state, n] \
                                      = cumulativeProbability[state, n] \
                                        + logDensity[state, n]

    # backtracking:
    bestStatePath = zeros(numberOfFrames)
    bestStatePath[-1] = argmax(cumulativeProbability[:, numberOfFrames - 1])
    for n in arange(numberOfFrames - 2, -1, -1):
        bestStatePath[n] = antecedents[bestStatePath[n + 1], n + 1]

    return bestStatePath

def viterbiTrackingArray(logDensity, logPriorDensities, logTransitionMatrix,
                         verbose=False):
    """
    bestStatePath = viterbiTrackingArray(logDensity, logPriorDensities,
                                    logTransitionMatrix, verbose=False)

    viterbiTrackingArray returns the best path through matrix logDensity,
    assuming that logDensity contains the likelihood of the observation
    sequence, conditionally upon the hidden states. A hidden Markov
    model (HMM) is assumed, with prior probabilities for the states
    given by logPriorDensities, and transition probabilities given
    by the matrix logTransitionMatrix. More precisely:
    Inputs:
        logDensity is a S x N ndarray, where S is the number of hidden
                      states and N is the number of frames of the
                      observed signal. The element at row s and
                      column n contains the conditional likelihood
                      of the signal at frame n, conditionally upon
                      state s. The given values should be given as the
                      logarithm of the probabilities.
        logPrioroDensities is a ndarray of size S, containing the prior
                           probabilities of the hidden states of the HMM,
                           logarithm of these values are expected.
        logTransitionMatrix is a S x S ndarray containing the transition
                            probabilities: at row s and column t, it
                            contains the probability of having state t
                            after state s, logarithm expected.
        verbose defines whether to display evolution information or not.
                Default is False.

    Outputs:
        bestStatePath is the sequence of best states, assuming the HMM
                      with the given parameters.
    """
    numberOfStates, numberOfFrames = logDensity.shape

    # logPriorDensities = vstack(logPriorDensities)
    onesStates = ones(numberOfStates)

    cumulativeProbability = zeros([numberOfStates, numberOfFrames])
    antecedents = zeros([numberOfStates, numberOfFrames], dtype=int)
    
    antecedents[:, 0] = -1
    cumulativeProbability[:, 0] = logPriorDensities[:] \
                                  + logDensity[:, 0]
    
    for n in arange(1, numberOfFrames):
        if verbose:
            print "frame number ", n, "over ", numberOfFrames
        # Find the state that minimizes the transition and the cumulative
        # probability. This operation can be done for all the target
        # states using numpy operations on ndarrays:
        # TODO: check that the transition
        antecedents[:, n] \
                   = argmax(outer(onesStates,
                                  cumulativeProbability[:, n - 1]) \
                            + logTransitionMatrix.T, axis=1)
        cumulativeProbability[:, n] \
                   = cumulativeProbability[antecedents[:, n], n - 1] \
                     + logTransitionMatrix[antecedents[:, n],
                                           arange(numberOfStates)] \
                     + logDensity[:, n]
    
    # backtracking:
    bestStatePath = zeros(numberOfFrames)
    bestStatePath[-1]= argmax(cumulativeProbability[:, numberOfFrames \
                                                                  - 1])
    for n in arange(numberOfFrames - 2, -1, -1):
        bestStatePath[n] = antecedents[bestStatePath[n + 1], n + 1]
        
    return bestStatePath
