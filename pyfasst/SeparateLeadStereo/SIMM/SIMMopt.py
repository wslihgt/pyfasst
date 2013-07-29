#!/usr/bin/python
#

# Script implementing the multiplicative rules from the following
# article:
# 
# J.-L. Durrieu, G. Richard, B. David and C. Fevotte
# Source/Filter Model for Unsupervised Main Melody
# Extraction From Polyphonic Audio Signals
# IEEE Transactions on Audio, Speech and Language Processing
# Vol. 18, No. 3, March 2010 
#
# with more details and new features explained in my PhD thesis:
#
# J.-L. Durrieu,
# Automatic Extraction of the Main Melody from Polyphonic Music Signals,
# EDITE
# Institut TELECOM, TELECOM ParisTech, CNRS LTCI

# changes to original version (SIMM.py):
#     Changing the memory representation of the arrays to fit the
#     best position and contiguity for np.dot to go faster
# 

import numpy as np
import time, os
import warnings

from numpy.random import randn
from string import join

def db(positiveValue):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(np.abs(positiveValue))

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)
    
    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    ratio = (X / Y)
    return np.sum((-np.log(ratio) + ratio - 1))

def SIMM(# the data to be fitted to:
         SX,
         # the basis matrices for the spectral combs
         WF0,
         # and for the elementary filters:
         WGAMMA,
         # number of desired filters, accompaniment spectra:
         numberOfFilters=4, numberOfAccompanimentSpectralShapes=10,
         # if any, initial amplitude matrices for 
         HGAMMA0=None, HPHI0=None,
         HF00=None,
         WM0=None, HM0=None,
         # Some more optional arguments, to control the "convergence"
         # of the algo
         numberOfIterations=1000, updateRulePower=1.0,
         stepNotes=4, 
         lambdaHF0=0.00,alphaHF0=0.99,
         displayEvolution=False, verbose=True,
         makeMovie=False,
         imageCanvas=None,
         progressBar=None,
         F0Table=None,
         computeError=False):
    """
    Changes: 20120820 the memory representation of the arrays, do not follow the
    following indications, for the moment... tests in progress
    
    HGAMMA, HPHI, HF0, HM, WM, recoError =
        SIMM(SX, WF0, WGAMMA, numberOfFilters=4,
             numberOfAccompanimentSpectralShapes=10, HGAMMA0=None, HPHI0=None,
             HF00=None, WM0=None, HM0=None, numberOfIterations=1000,
             updateRulePower=1.0, stepNotes=4, 
             lambdaHF0=0.00, alphaHF0=0.99, displayEvolution=False,
             verbose=True)

    Implementation of the Smooth-filters Instantaneous Mixture Model
    (SIMM). This model can be used to estimate the main melody of a
    song, and separate the lead voice from the accompaniment, provided
    that the basis WF0 is constituted of elements associated to
    particular pitches.

    Inputs:
        SX
            the F x N power spectrogram to be approximated.
            F is the number of frequency bins, while N is the number of
            analysis frames
        WF0
            the F x NF0 basis matrix containing the NF0 source elements
        WGAMMA
            the F x P basis matrix of P smooth elementary filters
        numberOfFilters
            the number of filters K to be considered
        numberOfAccompanimentSpectralShapes
            the number of spectral shapes R for the accompaniment
        HGAMMA0
            the P x K decomposition matrix of WPHI on WGAMMA
        HPHI0
            the K x N amplitude matrix of the filter part of the lead
            instrument
        HF00
            the NF0 x N amplitude matrix for the source part of the lead
            instrument
        WM0
            the F x R the matrix for spectral shapes of the
            accompaniment
        HM0
            the R x N amplitude matrix associated with each of the R
            accompaniment spectral shapes
        numberOfIterations
            the number of iterations for the estimatino algorithm
        updateRulePower
            the power to which the multiplicative gradient is elevated to
        stepNotes
            the number of elements in WF0 per semitone. stepNotes=4 means
            that there are 48 elements per octave in WF0.
        lambdaHF0
            Lagrangian multiplier for the octave control
        alphaHF0
            parameter that controls how much influence a lower octave
            can have on the upper octave's amplitude.

    Outputs:
        HGAMMA
            the estimated P x K decomposition matrix of WPHI on WGAMMA
        HPHI
            the estimated K x N amplitude matrix of the filter part 
        HF0
            the estimated NF0 x N amplitude matrix for the source part
        HM
            the estimated R x N amplitude matrix for the accompaniment
        WM
            the estimate F x R spectral shapes for the accompaniment
        recoError
            the successive values of the Itakura Saito divergence
            between the power spectrogram and the spectrogram
            computed thanks to the updated estimations of the matrices.

    Please also refer to the following article for more details about
    the algorithm within this function, as well as the meaning of the
    different matrices that are involved:
        J.-L. Durrieu, G. Richard, B. David and C. Fevotte
        Source/Filter Model for Unsupervised Main Melody
        Extraction From Polyphonic Audio Signals
        IEEE Transactions on Audio, Speech and Language Processing
        Vol. 18, No. 3, March 2010
    """
    
    
    eps = 10 ** (-20)

    if displayEvolution:
        import matplotlib.pyplot as plt
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()
    
    if not(progressBar is None):
        # progressBar is a QtGui.QProgressBar
        progressBar.show()
        progressBar.setMinimum(0)
        progressBar.setMaximum(numberOfIterations)
        progressBar.setValue(0)
        
    # renamed for convenience:
    K = numberOfFilters
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SX.shape
    if not SX.flags['C_CONTIGUOUS']:
        SX = np.ascontiguousarray(SX)
    NF0, Fwf0 = WF0.shape
    P, Fwgamma = WGAMMA.shape
    
    # Checking the sizes of the matrices
    if Fwf0 != F:
        raise ValueError("Dimension of arrays not right") # A REVOIR!!!
    if HGAMMA0 is None:
        HGAMMA0 = np.abs(randn(K, P))
    else:
        if not(isinstance(HGAMMA0,np.ndarray)): # default behaviour
            HGAMMA0 = np.ascontiguousarray(HGAMMA0)
        Khgamma0, Phgamma0 = HGAMMA0.shape
        if Phgamma0 != P or Khgamma0 != K:
            print "Wrong dimensions for given HGAMMA0, \n"
            print "random initialization used instead"
            HGAMMA0 = np.abs(randn(K, P))
            
    HGAMMA = HGAMMA0 # warning: this version does modify the original arrays!
    
    if HPHI0 is None: # default behaviour
        HPHI = np.abs(randn(N, K))
    else:
        Nhphi0, Khphi0  = np.array(HPHI0).shape
        if Khphi0 != K or Nhphi0 != N:
            print "Wrong dimensions for given HPHI0, \n"
            print "random initialization used instead"
            HPHI = np.abs(randn(N, K))
        else:
            HPHI = np.ascontiguousarray(HPHI0)
            
    if HF00 is None:
        HF00 = np.abs(randn(N, NF0))
    else:
        if np.array(HF00).shape[1] == NF0 and np.array(HF00).shape[0] == N:
            HF00 = np.ascontiguousarray(HF00)
        else:
            print "Wrong dimensions for given HF00, \n"
            print "random initialization used instead"
            HF00 = np.abs(randn(N, NF0))
    HF0 = HF00
    

    if HM0 is None:
        HM0 = np.abs(randn(N, R))
    else:
        if np.array(HM0).shape[1] == R and np.array(HM0).shape[0] == N:
            HM0 = np.ascontiguousarray(HM0)
        else:
            print "Wrong dimensions for given HM0, \n"
            print "random initialization used instead"
            HM0 = np.abs(randn(N, R))
    HM = HM0
    
    if WM0 is None:
        WM0 = np.abs(randn(R, F))
    else:
        if np.array(WM0).shape[1] == F and np.array(WM0).shape[0] == R:
            WM0 = np.ascontiguousarray(WM0)
        else:
            print "Wrong dimensions for given WM0, \n"
            print "random initialization used instead"
            WM0 = np.abs(randn(R, F))
    WM = WM0
    
    # making some yticks that make sense
    imgYticks = np.int32(np.linspace(NF0/5, NF0-1, num=5)).tolist()
    notesFreqs = {}
    notesFreqs['A4'] = 442
    notesFreqs['A2'] = notesFreqs['A4'] / 4
    notesFreqs['A3'] = notesFreqs['A4'] / 2
    notesFreqs['A5'] = notesFreqs['A4'] * 2
    notesFreqs['A6'] = notesFreqs['A4'] * 4
    if (F0Table is None):
        imgYticklabels = imgYticks
    else:
        imgYticklabels = np.int32(F0Table[imgYticks]).tolist()
        for k, v in notesFreqs.items():
            closestIndex = np.argmin(np.abs(F0Table-v))
            if np.abs(12*np.log2(F0Table[closestIndex])-12*np.log2(v)) < .25:
                imgYticks.append(closestIndex)
                imgYticklabels.append(k)
    
    # Iterations to estimate the SIMM parameters:
    WPHI = np.dot(HGAMMA, WGAMMA)
    SF0 = np.ascontiguousarray(np.dot(HF0, WF0).T)
    SPHI = np.ascontiguousarray(np.dot(HPHI, WPHI).T)
    SM = np.ascontiguousarray(np.dot(HM, WM).T)
    hatSX = SF0 * SPHI + SM
    
    ## SX = SX + np.abs(randn(F, N)) ** 2
                                       # should not need this line
                                       # which ensures that data is not
                                       # 0 everywhere. 
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])
    
    # Array containing the reconstruction error after the update of each 
    # of the parameter matrices:
    recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])

    if computeError:
        recoError[0] = ISDistortion(SX, hatSX)
    if verbose:
        print "Reconstruction error at beginning: ", recoError[0]
    counterError = 1
    
    if makeMovie:
        dirName = 'tmp%s/' %time.strftime("%Y%m%d%H%M%S")
        os.system('mkdir %s' %dirName)
    
    # Main loop for multiplicative updating rules:
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        if not(progressBar is None):
            progressBar.setValue(n+1)
        
        if verbose:
            print "iteration ", n, " over ", numberOfIterations
        
        if displayEvolution and not(imageCanvas is None):
            imageCanvas.ax.clear()
            imageCanvas.ax.imshow(db(HF0),
                                  origin='lower',
                                  cmap='jet',
                                  aspect='auto',
                                  interpolation='nearest')
            ## imageCanvas.ax.imshow(db(HF0) -
            ##                       np.outer(np.ones(NF0),
            ##                                np.sum(db(HF0),axis=0)),
            ##                       origin='lower',
            ##                       cmap='jet',
            ##                       aspect='auto')
            # plt.clim([np.amax(db(HF0))-100, np.amax(db(HF0))])
            ## imageCanvas.ax.get_images()[0].set_clim(-100, 0)
            imageCanvas.ax.get_images()[0].set_clim(np.amax(db(HF0))-100,\
                                                    np.amax(db(HF0)))
            imageCanvas.ax.set_yticks(imgYticks)
            imageCanvas.ax.set_yticklabels(imgYticklabels)
            imageCanvas.draw()
            imageCanvas.updateGeometry()
        
        if makeMovie:
            filename = dirName + '%04d' % n + '.png'
            plt.savefig(filename, dpi=100)
            
        # updating HF0:
        tempNumFbyN[:] = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN[:] = SPHI / np.maximum(hatSX, eps)
        
        ## # This to enable octave control
        ## HF0[np.arange(12 * stepNotes, NF0), :] \
        ##    = HF0[np.arange(12 * stepNotes, NF0), :] \
        ##      * (np.dot(WF0[:, np.arange(12 * stepNotes,
        ##                                 NF0)].T, tempNumFbyN) \
        ##         / np.maximum(
        ##     np.dot(WF0[:, np.arange(12 * stepNotes, NF0)].T,
        ##            tempDenFbyN) \
        ##     + lambdaHF0 * (- (alphaHF0 - 1.0) \
        ##                    / np.maximum(HF0[
        ##     np.arange(12 * stepNotes, NF0), :], eps) \
        ##                    + HF0[
        ##     np.arange(NF0 - 12 * stepNotes), :]),
        ##     eps)) ** omega
        ## 
        ## HF0[np.arange(12 * stepNotes), :] \
        ##    = HF0[np.arange(12 * stepNotes), :] \
        ##      * (np.dot(WF0[:, np.arange(12 * stepNotes)].T,
        ##               tempNumFbyN) /
        ##        np.maximum(
        ##         np.dot(WF0[:, np.arange(12 * stepNotes)].T,
        ##                tempDenFbyN), eps)) ** omega
        
        ## normal update rules:
        HF0 *= ((np.dot(WF0, tempNumFbyN) /
                 np.maximum(np.dot(WF0, tempDenFbyN), eps)) ** omega).T
        
        SF0[:] = np.maximum(np.dot(HF0, WF0).T, eps) # contiguity?
        if not SF0.flags['C_CONTIGUOUS']: #DEBUG
            raise AttributeError("SF0 not right."+str(SF0.flags))
        
        hatSX[:] = np.maximum(SF0 * SPHI + SM, eps)
        
        if computeError:
            recoError[counterError] = ISDistortion(SX, hatSX)
        
            if verbose:
                print "Reconstruction error difference after HF0   : ",
                print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        # updating HPHI
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)
        
        HPHI *= ((np.dot(WPHI, tempNumFbyN) / 
                  np.maximum(np.dot(WPHI, tempDenFbyN), eps)) 
                 ** omega).T
        sumHPHI = np.sum(HPHI, axis=1)
        HPHI[sumHPHI>0] /= np.vstack(sumHPHI[sumHPHI>0])
        HF0 *= np.vstack(sumHPHI)
        
        SF0[:] = np.maximum(np.dot(HF0, WF0).T, eps)
        SPHI[:] = np.maximum(np.dot(HPHI, WPHI).T, eps)
        hatSX[:] = np.maximum(SF0 * SPHI + SM, eps)
        
        if computeError:
            recoError[counterError] = ISDistortion(SX, hatSX)
        
            if verbose:
                print "Reconstruction error difference after HPHI  : ",
                print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        # updating HM
        tempNumFbyN[:] = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN[:] = 1 / np.maximum(hatSX, eps)
        
        HM *= ((np.dot(WM, tempNumFbyN) / 
                np.maximum(np.dot(WM, tempDenFbyN), eps))
               ** omega).T
        
        SM[:] = np.maximum(np.dot(HM, WM).T, eps)
        hatSX[:] = np.maximum(SF0 * SPHI + SM, eps)
        
        if computeError:
            recoError[counterError] = ISDistortion(SX, hatSX)
        
            if verbose:
                print "Reconstruction error difference after HM    : ", 
                print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        # updating HGAMMA
        tempNumFbyN[:] = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN[:] = SF0 / np.maximum(hatSX, eps)
        HGAMMA *= ((np.dot(WGAMMA, 
                           np.dot(tempNumFbyN, HPHI)) / 
                    np.maximum(np.dot(WGAMMA, 
                                      np.dot(tempDenFbyN, HPHI)), eps))
                   ** omega).T
        
        sumHGAMMA = np.sum(HGAMMA, axis=1)
        HGAMMA[sumHGAMMA>0] /= np.vstack(sumHGAMMA[sumHGAMMA>0])
        HPHI *= sumHGAMMA
        sumHPHI = np.sum(HPHI, axis=1)
        HPHI[sumHPHI>0] /= np.vstack(sumHPHI[sumHPHI>0])
        HF0 *= np.vstack(sumHPHI)
        
        WPHI[:] = np.maximum(np.dot(HGAMMA, WGAMMA), eps)
        SF0[:] = np.maximum(np.dot(HF0, WF0).T, eps)
        SPHI[:] = np.maximum(np.dot(HPHI, WPHI).T, eps)
        hatSX[:] = np.maximum(SF0 * SPHI + SM, eps)
        
        if computeError:
            recoError[counterError] = ISDistortion(SX, hatSX)
            
            if verbose:
                print "Reconstruction error difference after HGAMMA: ",
                print recoError[counterError] - recoError[counterError - 1]
            
        counterError += 1
        
        # updating WM, after a certain number of iterations
        #  TODO: add this as an option
        if n > -1: # this test can be used such that WM is updated only
                   # after a certain number of iterations
            tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
            tempDenFbyN = 1 / np.maximum(hatSX, eps)
            WM *= ((np.dot(tempNumFbyN, HM) /
                    np.maximum(np.dot(tempDenFbyN, HM),
                               eps)) ** omega).T
            
            sumWM = np.sum(WM, axis=1)
            WM[sumWM>0] /= np.vstack(sumWM[sumWM>0])
            HM *= sumWM
            
            SM[:] = np.maximum(np.dot(HM, WM).T, eps)
            hatSX[:] = np.maximum(SF0 * SPHI + SM, eps)
            
            if computeError:
                recoError[counterError] = ISDistortion(SX, hatSX)
            
                if verbose:
                    print "Reconstruction error difference after WM    : ",
                    print recoError[counterError] - recoError[counterError - 1]
            counterError += 1
            
    return HGAMMA, HPHI, HF0, HM, WM, recoError

def Stereo_SIMM(# the data to be fitted to:
         SXR, SXL,
         # the basis matrices for the spectral combs
         WF0,
         # and for the elementary filters:
         WGAMMA,
         # number of desired filters, accompaniment spectra:
         numberOfFilters=4, numberOfAccompanimentSpectralShapes=10,
         # if any, initial amplitude matrices for 
         HGAMMA0=None, HPHI0=None,
         HF00=None,
         WM0=None, HM0=None,
         # Some more optional arguments, to control the "convergence"
         # of the algo
         numberOfIterations=1000, updateRulePower=1.0,
         stepNotes=4, 
         lambdaHF0=0.00,alphaHF0=0.99,
         displayEvolution=False, verbose=True,
         updateHGAMMA=True):
    """
    HGAMMA, HPHI, HF0, HM, WM, recoError =
        SIMM(SXR, SXL, WF0, WGAMMA, numberOfFilters=4,
             numberOfAccompanimentSpectralShapes=10, HGAMMA0=None, HPHI0=None,
             HF00=None, WM0=None, HM0=None, numberOfIterations=1000,
             updateRulePower=1.0, stepNotes=4, 
             lambdaHF0=0.00, alphaHF0=0.99, displayEvolution=False,
             verbose=True)

    Implementation of the Smooth-filters Instantaneous Mixture Model
    (SIMM). This model can be used to estimate the main melody of a
    song, and separate the lead voice from the accompaniment, provided
    that the basis WF0 is constituted of elements associated to
    particular pitches.

    Inputs:
        SX
            the F x N power spectrogram to be approximated.
            F is the number of frequency bins, while N is the number of
            analysis frames
        WF0
            the F x NF0 basis matrix containing the NF0 source elements
        WGAMMA
            the F x P basis matrix of P smooth elementary filters
        numberOfFilters
            the number of filters K to be considered
        numberOfAccompanimentSpectralShapes
            the number of spectral shapes R for the accompaniment
        HGAMMA0
            the P x K decomposition matrix of WPHI on WGAMMA
        HPHI0
            the K x N amplitude matrix of the filter part of the lead
            instrument
        HF00
            the NF0 x N amplitude matrix for the source part of the lead
            instrument
        WM0
            the F x R the matrix for spectral shapes of the
            accompaniment
        HM0
            the R x N amplitude matrix associated with each of the R
            accompaniment spectral shapes
        numberOfIterations
            the number of iterations for the estimatino algorithm
        updateRulePower
            the power to which the multiplicative gradient is elevated to
        stepNotes
            the number of elements in WF0 per semitone. stepNotes=4 means
            that there are 48 elements per octave in WF0.
        lambdaHF0
            Lagrangian multiplier for the octave control
        alphaHF0
            parameter that controls how much influence a lower octave
            can have on the upper octave's amplitude.

    Outputs:
        HGAMMA
            the estimated P x K decomposition matrix of WPHI on WGAMMA
        HPHI
            the estimated K x N amplitude matrix of the filter part 
        HF0
            the estimated NF0 x N amplitude matrix for the source part
        HM
            the estimated R x N amplitude matrix for the accompaniment
        WM
            the estimate F x R spectral shapes for the accompaniment
        recoError
            the successive values of the Itakura Saito divergence
            between the power spectrogram and the spectrogram
            computed thanks to the updated estimations of the matrices.

    Please also refer to the following article for more details about
    the algorithm within this function, as well as the meaning of the
    different matrices that are involved:
        J.-L. Durrieu, G. Richard, B. David and C. Fevotte
        Source/Filter Model for Unsupervised Main Melody
        Extraction From Polyphonic Audio Signals
        IEEE Transactions on Audio, Speech and Language Processing
        Vol. 18, No. 3, March 2010
    """
    eps = 10 ** (-20)

    if displayEvolution:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()

    # renamed for convenience:
    K = numberOfFilters
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SXR.shape
    if (F, N) != SXL.shape:
        print "The input STFT matrices do not have the same dimension.\n"
        print "Please check what happened..."
        raise ValueError("Dimension of STFT matrices must be the same.")
        
    Fwf0, NF0 = WF0.shape
    Fwgamma, P = WGAMMA.shape
    
    # Checking the sizes of the matrices
    if Fwf0 != F:
        return False # A REVOIR!!!
    if HGAMMA0 is None:
        HGAMMA0 = np.abs(randn(P, K))
    else:
        if not(isinstance(HGAMMA0,np.ndarray)): # default behaviour
            HGAMMA0 = np.array(HGAMMA0)
        Phgamma0, Khgamma0 = HGAMMA0.shape
        if Phgamma0 != P or Khgamma0 != K:
            print "Wrong dimensions for given HGAMMA0, \n"
            print "random initialization used instead"
            HGAMMA0 = np.abs(randn(P, K))

    HGAMMA = np.copy(HGAMMA0)
    
    if HPHI0 is None: # default behaviour
        HPHI = np.abs(randn(K, N))
    else:
        Khphi0, Nhphi0 = np.array(HPHI0).shape
        if Khphi0 != K or Nhphi0 != N:
            print "Wrong dimensions for given HPHI0, \n"
            print "random initialization used instead"
            HPHI = np.abs(randn(K, N))
        else:
            HPHI = np.copy(np.array(HPHI0))

    if HF00 is None:
        HF00 = np.abs(randn(NF0, N))
    else:
        if np.array(HF00).shape[0] == NF0 and np.array(HF00).shape[1] == N:
            HF00 = np.array(HF00)
        else:
            print "Wrong dimensions for given HF00, \n"
            print "random initialization used instead"
            HF00 = np.abs(randn(NF0, N))
    HF0 = np.copy(HF00)

    if HM0 is None:
        HM0 = np.abs(randn(R, N))
    else:
        if np.array(HM0).shape[0] == R and np.array(HM0).shape[1] == N:
            HM0 = np.array(HM0)
        else:
            print "Wrong dimensions for given HM0, \n"
            print "random initialization used instead"
            HM0 = np.abs(randn(R, N))
    HM = np.copy(HM0)

    if WM0 is None:
        WM0 = np.abs(randn(F, R))
    else:
        if np.array(WM0).shape[0] == F and np.array(WM0).shape[1] == R:
            WM0 = np.array(WM0)
        else:
            print "Wrong dimensions for given WM0, \n"
            print "random initialization used instead"
            WM0 = np.abs(randn(F, R))
    WM = np.copy(WM0)

    alphaR = 0.5
    alphaL = 0.5
    betaR = np.diag(np.random.rand(R))
    betaL = np.eye(R) - betaR
    
    # Iterations to estimate the SIMM parameters:
    WPHI = np.dot(WGAMMA, HGAMMA)
    SF0 = np.dot(WF0, HF0)
    SPHI = np.dot(WPHI, HPHI)
    # SM = np.dot(WM, HM)
    hatSXR = (alphaR**2) * SF0 * SPHI + np.dot(np.dot(WM, betaR**2),HM)
    hatSXL = (alphaL**2) * SF0 * SPHI + np.dot(np.dot(WM, betaL**2),HM)

    # SX = SX + np.abs(randn(F, N)) ** 2
                                       # should not need this line
                                       # which ensures that data is not
                                       # 0 everywhere. 
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])

    # Array containing the reconstruction error after the update of each 
    # of the parameter matrices:
    recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
    recoError[0] = ISDistortion(SXR, hatSXR) + ISDistortion(SXL, hatSXL)
    if verbose:
        print "Reconstruction error at beginning: ", recoError[0]
    counterError = 1
    if displayEvolution:
        h1 = plt.figure(1)

    # Main loop for multiplicative updating rules:
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        if verbose:
            print "iteration ", n, " over ", numberOfIterations
        if displayEvolution:
            h1.clf();imageM(db(HF0));
            plt.clim([np.amax(db(HF0))-100, np.amax(db(HF0))]);plt.draw();
            # h1.clf();
            # imageM(HF0 * np.outer(np.ones([NF0, 1]),
            #                       1 / (HF0.max(axis=0))));

        # updating HF0:
        tempNumFbyN = ((alphaR**2) * SPHI * SXR) / np.maximum(hatSXR ** 2, eps)\
                      + ((alphaL**2) * SPHI * SXL) / np.maximum(hatSXL ** 2, eps)
        tempDenFbyN = (alphaR**2) * SPHI / np.maximum(hatSXR, eps)\
                      + (alphaL**2) * SPHI / np.maximum(hatSXL, eps)

        # This to enable octave control
        HF0[np.arange(12 * stepNotes, NF0), :] \
           = HF0[np.arange(12 * stepNotes, NF0), :] \
             * (np.dot(WF0[:, np.arange(12 * stepNotes,
                                        NF0)].T, tempNumFbyN) \
                / np.maximum(
            np.dot(WF0[:, np.arange(12 * stepNotes, NF0)].T,
                   tempDenFbyN) \
            + lambdaHF0 * (- (alphaHF0 - 1.0) \
                           / np.maximum(HF0[
            np.arange(12 * stepNotes, NF0), :], eps) \
                           + HF0[
            np.arange(NF0 - 12 * stepNotes), :]),
            eps)) ** omega
        
        HF0[np.arange(12 * stepNotes), :] \
           = HF0[np.arange(12 * stepNotes), :] \
             * (np.dot(WF0[:, np.arange(12 * stepNotes)].T,
                      tempNumFbyN) /
               np.maximum(
                np.dot(WF0[:, np.arange(12 * stepNotes)].T,
                       tempDenFbyN), eps)) ** omega

##        # normal update rules:
##        HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) /
##                     np.maximum(np.dot(WF0.T, tempDenFbyN), eps)) ** omega
        
        
        SF0 = np.maximum(np.dot(WF0, HF0), eps)
        hatSXR = np.maximum((alphaR**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaR**2),HM),
                            eps)
        hatSXL = np.maximum((alphaL**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaL**2),HM),
                            eps)
        
        ## recoError[counterError] = ISDistortion(SXR, hatSXR) \
        ##                           + ISDistortion(SXL, hatSXL)
        ## 
        ## if verbose:
        ##     print "Reconstruction error difference after HF0   : ",
        ##     print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
    
        # updating HPHI
        if updateHGAMMA or True:
            tempNumFbyN = ((alphaR**2) * SF0 * SXR) / np.maximum(hatSXR ** 2, eps)\
                          + ((alphaL**2) * SF0 * SXL) / np.maximum(hatSXL ** 2, eps)
            tempDenFbyN = (alphaR**2) * SF0 / np.maximum(hatSXR, eps)\
                          + (alphaL**2) * SF0 / np.maximum(hatSXL, eps)
            HPHI = HPHI * (np.dot(WPHI.T, tempNumFbyN) / np.maximum(np.dot(WPHI.T, tempDenFbyN), eps)) ** omega
            sumHPHI = np.sum(HPHI, axis=0)
            HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
            HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)
            
            SF0 = np.maximum(np.dot(WF0, HF0), eps)
            SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
            hatSXR = np.maximum((alphaR**2) * SF0 * SPHI + \
                                np.dot(np.dot(WM, betaR**2),HM),
                                eps)
            hatSXL = np.maximum((alphaL**2) * SF0 * SPHI + \
                                np.dot(np.dot(WM, betaL**2),HM),
                                eps)
            
            ## recoError[counterError] = ISDistortion(SXR, hatSXR) \
            ##                           + ISDistortion(SXL, hatSXL)
            ## 
            ## if verbose:
            ##     print "Reconstruction error difference after HPHI  : ", recoError[counterError] - recoError[counterError - 1]
            ##     
            counterError += 1
        
        
        # updating HM
        # tempNumFbyN = SXR / np.maximum(hatSXR ** 2, eps)\
        #               + SXL / np.maximum(hatSXL ** 2, eps)
        # tempDenFbyN = 1 / np.maximum(hatSXR, eps)\
        #               + 1 / np.maximum(hatSXL, eps)
        # HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) / np.maximum(np.dot(WM.T, tempDenFbyN), eps)) ** omega, eps)
        HM = HM * \
             ((np.dot(np.dot((betaR**2), WM.T), SXR /
                      np.maximum(hatSXR ** 2, eps)) +
               np.dot(np.dot((betaL**2), WM.T), SXL /
                      np.maximum(hatSXL ** 2, eps))
               ) /
              np.maximum(np.dot(np.dot((betaR**2), WM.T), 1 /
                                np.maximum(hatSXR, eps)) +
                         np.dot(np.dot((betaL**2), WM.T), 1 /
                                np.maximum(hatSXL, eps)),
                         eps)) ** omega
        
        hatSXR = np.maximum((alphaR**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaR**2),HM), eps)
        hatSXL = np.maximum((alphaL**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaL**2),HM), eps)
        
        ## recoError[counterError] = ISDistortion(SXR, hatSXR) \
        ##                           + ISDistortion(SXL, hatSXL)
        ## 
        ## if verbose:
        ##     print "Reconstruction error difference after HM    : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1  

        # updating HGAMMA
        if updateHGAMMA:
            tempNumFbyN = ((alphaR ** 2) * SF0 * SXR) / np.maximum(hatSXR ** 2, eps)\
                          + ((alphaL ** 2) * SF0 * SXL) / np.maximum(hatSXL ** 2, eps)
            tempDenFbyN = (alphaR ** 2) * SF0 / np.maximum(hatSXR, eps) \
                          + (alphaL ** 2) * SF0 / np.maximum(hatSXL, eps)
            
            HGAMMA = np.maximum(HGAMMA * (np.dot(WGAMMA.T, np.dot(tempNumFbyN, HPHI.T)) / np.maximum(np.dot(WGAMMA.T, np.dot(tempDenFbyN, HPHI.T)), eps)) ** omega, eps)
            
            sumHGAMMA = np.sum(HGAMMA, axis=0)
            HGAMMA[:, sumHGAMMA>0] = HGAMMA[:, sumHGAMMA>0] / np.outer(np.ones(P), sumHGAMMA[sumHGAMMA>0])
            HPHI = HPHI * np.outer(sumHGAMMA, np.ones(N))
            sumHPHI = np.sum(HPHI, axis=0)
            HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
            HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)
            
            WPHI = np.maximum(np.dot(WGAMMA, HGAMMA), eps)
            SF0 = np.maximum(np.dot(WF0, HF0), eps)
            SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
            
            hatSXR = np.maximum((alphaR**2) * SF0 * SPHI + \
                                np.dot(np.dot(WM, betaR**2),HM), eps)
            hatSXL = np.maximum((alphaL**2) * SF0 * SPHI + \
                                np.dot(np.dot(WM, betaL**2),HM), eps)
            
            ## recoError[counterError] = ISDistortion(SXR, hatSXR) \
            ##                           + ISDistortion(SXL, hatSXL)
            ## 
            ## if verbose:
            ##     print "Reconstruction error difference after HGAMMA: ",
            ##     print recoError[counterError] - recoError[counterError - 1]
            ## 
            counterError += 1
        
        # updating WM, after a certain number of iterations (here, after 1 iteration)
        if n > -1: # this test can be used such that WM is updated only
                  # after a certain number of iterations
##           tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
##            tempDenFbyN = 1 / np.maximum(hatSX, eps)
##            WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) /
##                                  np.maximum(np.dot(tempDenFbyN, HM.T),
##                                             eps)) ** omega, eps)
            WM = WM * \
                 ((np.dot(SXR / np.maximum(hatSXR ** 2, eps),
                          np.dot(HM.T, betaR ** 2)) +
                   np.dot(SXL / np.maximum(hatSXL ** 2, eps),
                          np.dot(HM.T, betaL ** 2))
                   ) /
                  (np.dot(1 / np.maximum(hatSXR, eps),
                          np.dot(HM.T, betaR ** 2)) +
                   np.dot(1 / np.maximum(hatSXL, eps),
                          np.dot(HM.T, betaL ** 2))
                   )) ** omega
            
            sumWM = np.sum(WM, axis=0)
            WM[:, sumWM>0] = (WM[:, sumWM>0] /
                              np.outer(np.ones(F),sumWM[sumWM>0]))
            HM = HM * np.outer(sumWM, np.ones(N))
            
            hatSXR = np.maximum((alphaR**2) * SF0 * SPHI + \
                                np.dot(np.dot(WM, betaR**2),HM), eps)
            hatSXL = np.maximum((alphaL**2) * SF0 * SPHI + \
                                np.dot(np.dot(WM, betaL**2),HM), eps)
            
            ## recoError[counterError] = ISDistortion(SXR, hatSXR) \
            ##                       + ISDistortion(SXL, hatSXL)
            ## 
            ## if verbose:
            ##     print "Reconstruction error difference after WM    : ",
            ##     print recoError[counterError] - recoError[counterError - 1]
            counterError += 1

        # updating alphaR and alphaL:
        tempNumFbyN = SF0 * SPHI * SXR / np.maximum(hatSXR ** 2, eps)
        tempDenFbyN = SF0 * SPHI / np.maximum(hatSXR, eps)
        alphaR = np.maximum(alphaR *
                            (np.sum(tempNumFbyN) /
                            np.sum(tempDenFbyN)) ** (omega*.1), eps)
        tempNumFbyN = SF0 * SPHI * SXL / np.maximum(hatSXL ** 2, eps)
        tempDenFbyN = SF0 * SPHI / np.maximum(hatSXL, eps)
        alphaL = np.maximum(alphaL *
                            (np.sum(tempNumFbyN) /
                            np.sum(tempDenFbyN)) ** (omega*.1), eps)
        alphaR = alphaR / np.maximum(alphaR + alphaL, .001)
        alphaL = np.copy(1 - alphaR)

            
        hatSXR = np.maximum((alphaR**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaR**2),HM), eps)
        hatSXL = np.maximum((alphaL**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaL**2),HM), eps)
        
        ## recoError[counterError] = ISDistortion(SXR, hatSXR) \
        ##                           + ISDistortion(SXL, hatSXL)
        ## 
        ## if verbose:
        ##     print "Reconstruction error difference after ALPHA : ",
        ##     print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
            

        # updating betaR and betaL
        betaR = np.diag(np.diag(np.maximum(betaR *
                                   ((np.dot(np.dot(WM.T, SXR / np.maximum(hatSXR ** 2, eps)), HM.T)) /
                                   (np.dot(np.dot(WM.T, 1 / np.maximum(hatSXR, eps)), HM.T))) ** (omega*.1), eps)))
        betaL = np.diag(np.diag(np.maximum(betaL *
                                   ((np.dot(np.dot(WM.T, SXL / np.maximum(hatSXL ** 2, eps)), HM.T)) /
                                   (np.dot(np.dot(WM.T, 1 / np.maximum(hatSXL, eps)), HM.T))) ** (omega*.1), eps)))
        betaR = betaR / np.maximum(betaR + betaL, eps)
        betaL = np.copy(np.eye(R) - betaR)

        hatSXR = np.maximum((alphaR**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaR**2),HM), eps)
        hatSXL = np.maximum((alphaL**2) * SF0 * SPHI + \
                            np.dot(np.dot(WM, betaL**2),HM), eps)
        
        ## recoError[counterError] = ISDistortion(SXR, hatSXR) \
        ##                           + ISDistortion(SXL, hatSXL)
        ## 
        ## if verbose:
        ##     print "Reconstruction error difference after BETA  : ",
        ##     print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
    return alphaR, alphaL, HGAMMA, HPHI, HF0, betaR, betaL, HM, WM, recoError

def stereo_NMF(SXR, SXL,
               numberOfAccompanimentSpectralShapes,
               WM0=None, HM0=None,
               numberOfIterations=50, updateRulePower=1.0,
               verbose=False, displayEvolution=False):
    
    eps = 10 ** (-20)
    
    if displayEvolution:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()
    
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SXR.shape
    if (F, N) != SXL.shape:
        print "The input STFT matrices do not have the same dimension.\n"
        print "Please check what happened..."
        raise ValueError("Dimension of STFT matrices must be the same.")
    
    if HM0 is None:
        HM0 = np.abs(randn(R, N))
    else:
        if np.array(HM0).shape[0] == R and np.array(HM0).shape[1] == N:
            HM0 = np.array(HM0)
        else:
            print "Wrong dimensions for given HM0, \n"
            print "random initialization used instead"
            HM0 = np.abs(randn(R, N))
    HM = np.copy(HM0)
    
    if WM0 is None:
        WM0 = np.abs(randn(F, R))
    else:
        if np.array(WM0).shape[0] == F and np.array(WM0).shape[1] == R:
            WM0 = np.array(WM0)
        else:
            print "Wrong dimensions for given WM0, \n"
            print "random initialization used instead"
            WM0 = np.abs(randn(F, R))
    WM = np.copy(WM0)
    
    betaR = np.diag(np.random.rand(R))
    betaL = np.eye(R) - betaR
    
    hatSXR = np.maximum(np.dot(np.dot(WM, betaR**2), HM), eps)
    hatSXL = np.maximum(np.dot(np.dot(WM, betaL**2), HM), eps)
    
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])
    
    recoError = np.zeros([numberOfIterations * 3 + 1])
    recoError[0] = ISDistortion(SXR, hatSXR) + ISDistortion(SXL, hatSXL)
    if verbose:
        print "Reconstruction error at beginning: ", recoError[0]
    counterError = 1
    if displayEvolution:
        h1 = plt.figure(1)
        
        
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        if verbose:
            print "iteration ", n, " over ", numberOfIterations
            
        if displayEvolution:
            h1.clf()
            imageM(db(hatSXR))
            plt.clim([np.amax(db(hatSXR))-100, np.amax(db(hatSXR))])
            plt.draw()
        
        # updating HM
        HM = HM * \
             ((np.dot(np.dot((betaR**2), WM.T), SXR /
                      np.maximum(hatSXR ** 2, eps)) +
               np.dot(np.dot((betaL**2), WM.T), SXL /
                      np.maximum(hatSXL ** 2, eps))
               ) /
              np.maximum(np.dot(np.dot((betaR**2), WM.T), 1 /
                                np.maximum(hatSXR, eps)) +
                         np.dot(np.dot((betaL**2), WM.T), 1 /
                                np.maximum(hatSXL, eps)),
                         eps)) ** omega
        
        hatSXR = np.maximum(np.dot(np.dot(WM, betaR**2),HM), eps)
        hatSXL = np.maximum(np.dot(np.dot(WM, betaL**2),HM), eps)
        
        recoError[counterError] = ISDistortion(SXR, hatSXR) \
                                  + ISDistortion(SXL, hatSXL)
        
        if verbose:
            print "Reconstruction error difference after HM    : ",\
                  recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        # updating WM
        WM = WM * \
             ((np.dot(SXR / np.maximum(hatSXR ** 2, eps),
                      np.dot(HM.T, betaR ** 2)) +
               np.dot(SXL / np.maximum(hatSXL ** 2, eps),
                      np.dot(HM.T, betaL ** 2))
               ) /
              (np.dot(1 / np.maximum(hatSXR, eps),
                      np.dot(HM.T, betaR ** 2)) +
               np.dot(1 / np.maximum(hatSXL, eps),
                      np.dot(HM.T, betaL ** 2))
               )) ** omega
        
        sumWM = np.sum(WM, axis=0)
        WM[:, sumWM>0] = (WM[:, sumWM>0] /
                          np.outer(np.ones(F),sumWM[sumWM>0]))
        HM = HM * np.outer(sumWM, np.ones(N))
        
        hatSXR = np.maximum(np.dot(np.dot(WM, betaR**2), HM), eps)
        hatSXL = np.maximum(np.dot(np.dot(WM, betaL**2), HM), eps)
        
        recoError[counterError] = ISDistortion(SXR, hatSXR) \
                                  + ISDistortion(SXL, hatSXL)
        
        if verbose:
            print "Reconstruction error difference after WM    : ",
            print recoError[counterError] - recoError[counterError - 1]
            
        counterError += 1
        
        # updating betaR and betaL
        betaR = np.diag(np.diag(np.maximum(betaR *
                        ((np.dot(np.dot(WM.T, SXR / np.maximum(hatSXR ** 2,
                                                               eps)),
                                 HM.T)) /
                         (np.dot(np.dot(WM.T, 1 / np.maximum(hatSXR,
                                                             eps)),
                                 HM.T))) ** (omega*.1), eps)))
        betaL = np.diag(np.diag(np.maximum(betaL *
                        ((np.dot(np.dot(WM.T, SXL / np.maximum(hatSXL ** 2,
                                                               eps)),
                                 HM.T)) /
                         (np.dot(np.dot(WM.T, 1 / np.maximum(hatSXL,
                                                             eps)),
                                 HM.T))) ** (omega*.1), eps)))
        betaR = betaR / np.maximum(betaR + betaL, eps)
        betaL = np.copy(np.eye(R) - betaR)
        
        hatSXR = np.maximum(np.dot(np.dot(WM, betaR**2), HM), eps)
        hatSXL = np.maximum(np.dot(np.dot(WM, betaL**2), HM), eps)
        
        recoError[counterError] = ISDistortion(SXR, hatSXR) \
                                  + ISDistortion(SXL, hatSXL)
        
        if verbose:
            print "Reconstruction error difference after BETA  : ",
            print recoError[counterError] - recoError[counterError - 1]
        
        counterError += 1
        
    return betaR, betaL, HM, WM
