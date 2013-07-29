#!/usr/bin/python

# copyright (C) 2012 Jean-Louis Durrieu

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t dtype_t

def viterbiTracking(int numberOfStates,
                    int numberOfFrames,
                    np.ndarray[dtype_t, ndim=2] logDensity,
                    np.ndarray[dtype_t, ndim=1] logPriorDensities,
                    np.ndarray[dtype_t, ndim=2] logTransitionMatrix,
                    bint verbose=False,):
    """
    Cython implementation of the Viterbi algorithm:
    
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
    cdef int n, state, state_
    cdef dtype_t tempCumProba
    cdef np.ndarray[dtype_t, ndim=2] cumulativeProbability
    cdef np.ndarray[np.int_t, ndim=2] antecedents
    cdef np.ndarray[np.int_t, ndim=1] bestStatePath
    
    cumulativeProbability = np.zeros([numberOfStates, numberOfFrames])
    antecedents = np.zeros([numberOfStates, numberOfFrames], dtype=np.int)
    
    for state in range(numberOfStates):
        antecedents[state, 0] = -1
        cumulativeProbability[state, 0] = logPriorDensities[state] \
                                          + logDensity[state, 0]
    
    for n in range(1, numberOfFrames):
        #if verbose:
        #    print "frame number ", n, "over ", numberOfFrames
        for state in range(numberOfStates):
            #if verbose:
            #    print "     state number ",state, " over ", numberOfStates
            cumulativeProbability[state, n] \
                                     = cumulativeProbability[0, n - 1] \
                                       + logTransitionMatrix[0, state]
            antecedents[state, n] = 0
            for state_ in range(1, numberOfStates):
                #if verbose:
                #    print "          state number ",
                #    print state_, " over ", numberOfStates
                tempCumProba = cumulativeProbability[state_, n - 1] \
                               + logTransitionMatrix[state_, state]
                if (tempCumProba > cumulativeProbability[state, n]):
                    cumulativeProbability[state, n] = tempCumProba
                    antecedents[state, n] = state_
            cumulativeProbability[state, n] \
                                      = cumulativeProbability[state, n] \
                                        + logDensity[state, n]
            
    # backtracking: 
    bestStatePath = np.zeros([numberOfFrames], dtype=np.int)
    bestStatePath[numberOfFrames-1] = \
        np.argmax(cumulativeProbability[:, numberOfFrames - 1])
    for n in range(numberOfFrames - 2, -1, -1):
        bestStatePath[n] = antecedents[bestStatePath[n + 1], n + 1]
    
    return bestStatePath

