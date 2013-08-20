#!/usr/bin/python

"""separateLeadFunctions.py

Description
===========

This module provides functions that are useful for the SeparateLeadStereo
modules, essentially time-frequency transformations (and inverse), as well
as generation of dictionary matrices.

Usage
=====

See each function docstring for more information.

TODO: expend this?
TODO: move all these functions in different modules, in :py:mod:`pyfasst.tools` for instance

License
=======

Copyright (C) 2011-2013 Jean-Louis Durrieu

"""

# copyright (C) 2011 Jean-Louis Durrieu

import numpy as np
# temporary: in last move, put the right module as in a package once
# debugged
import sys, os

from ..tftransforms import minqt
from ..tftransforms import nsgt
from .. import audioObject as ao # for all these fancy transforms

from ..tools.utils import *
from ..tools.distances import ISDistortion

### SOME USEFUL, INSTRUMENTAL, FUNCTIONS

##def nextpow2(i):
##    """
##    Find 2^n that is equal to or greater than.
    
##    code taken from the website:
##    http://www.phys.uu.nl/~haque/computing/WPark_recipes_in_python.html
##    """
##    n = 2
##    while n < i:
##        n = n * 2
##    return n

##def ISDistortion(X,Y):
##    """
##    value = ISDistortion(X, Y)
  
##    Returns the value of the Itakura-Saito (IS) divergence between
##    matrix X and matrix Y. X and Y should be two NumPy arrays with
##    same dimension.
##    """
##    return sum((-np.log(X / Y) + (X / Y) - 1))

# DEFINING SOME WINDOW FUNCTIONS

##def sinebell(lengthWindow):
##    """
##    window = sinebell(lengthWindow)
    
##    Computes a "sinebell" window function of length L=lengthWindow
    
##    The formula is:
##        window(t) = sin(pi * t / L), t = 0..L-1
##    """
##    window = np.sin((np.pi * (np.arange(lengthWindow))) \
##                    / (1.0 * lengthWindow))
##    return window

##def hann(args):
##    """
##    window = hann(args)
    
##    Computes a Hann window, with NumPy's function hanning(args).
##    """
##    return np.hanning(args)

# FUNCTIONS FOR TIME-FREQUENCY REPRESENTATION

def stft(data, window=sinebell(2048), hopsize=256.0, nfft=2048.0, \
         fs=44100.0, start=0, stop=None):
    """\
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
    
    :param data:
            one-dimensional time-series to be analyzed
    :param window:
            analysis window
    :param hopsize:
            hopsize for the analysis
    :param nfft:
            number of points for the Fourier
            computation (the user has to provide an
            even number)
    :param fs: sampling rate of the signal
        
    Outputs:

    :returns:
        * X
            STFT of data
        * F
            values of frequencies at each Fourier
            bins
        * N
            central time at the middle of each
            analysis window
    
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    # !!! adding zeros to the beginning of data, such that the first
    # window is centered on the first sample of data
    data = np.concatenate((np.zeros(lengthWindow / 2.0),
                           data,
                           np.zeros(lengthWindow / 2.0)))
    lengthData = data.size
    
    # adding one window for the last frame (same reason as for the
    # first frame)
    numberFrames = np.ceil((lengthData - lengthWindow) / hopsize \
                           + 1) + 1  
    newLengthData = (numberFrames - 1) * hopsize + lengthWindow
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros([newLengthData - lengthData])))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an
    # even number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2.0 + 1
    
    if stop is None:
        stop = numberFrames
    
    STFT = np.zeros([numberFrequencies,
                     stop-start],#numberFrames],
                    dtype=complex)
    
    for n in np.arange(start, stop):#numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[beginFrame:endFrame]
        STFT[:,n-start] = np.fft.rfft(frameToProcess, nfft);
    
    F = np.arange(numberFrequencies) / nfft * fs
    N = np.arange(numberFrames) * hopsize / fs
    
    return STFT, F, N

def istft(X, analysisWindow=None,
          window=sinebell(2048), hopsize=256.0, nfft=2048.0,
          originalDataLen=None, start=-1, stop=None):
    """\
    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.
    
    Inputs:
    
    :param X:
            STFT of the signal, to be "inverted"
    :param window: 
            synthesis window
            (should be the "complementary" window
            for the analysis window)
    :param hopsize: 
            hopsize for the analysis
    :param nfft: 
            number of points for the Fourier
            computation
            (the user has to provide an even number)
                                
    Outputs:

    :returns:
        * `data` -  
          time series corresponding to the given
          STFT the first half-window is removed,
          complying with the STFT computation
          given in the function 'stft'
    
    """
    if analysisWindow is None:
        analysisWindow = window
        
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = np.array(X.shape)
    lengthData = hopsize * (numberFrames - 1) + lengthWindow
    
    normalisationSeq = np.zeros(lengthData)
    
    data = np.zeros(lengthData)
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], nfft)
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = data[beginFrame:endFrame] \
                                    + window * frameTMP
    
    # remove the extra bit before data that was - supposedly - added
    # in the stft computation:
    normalisationSeq[:lengthWindow] = (
        normalisationSeq[lengthWindow:2*lengthWindow])
    normalisationSeq[-lengthWindow:] = (
        normalisationSeq[(-2*lengthWindow):(-lengthWindow)])
    if np.any(normalisationSeq==0):
        print "there were some 0s in there..."# DEBUG
    normalisationSeq[normalisationSeq==0] = 1.
    data /= normalisationSeq
    
    # this could be used somewhere, but better do the cutting outside:
    ##if start == 0 and False:
    ##    data = data[(lengthWindow / 2.0):]
    
    if originalDataLen is not None:
        data = data[:originalDataLen] 
    return data

# DEFINING THE FUNCTIONS TO CREATE THE 'BASIS' WF0

def generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048, stepNotes=4, \
                         lengthWindow=2048, Ot=0.5, perF0=1, \
                         depthChirpInSemiTone=0.5, loadWF0=True,
                         analysisWindow='hanning'):
    """\
    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:
    
    Inputs:

    :param minF0:
            the minimum value for the fundamental
            frequency (F0)
    :param maxF0:                
            the maximum value for F0
    :param Fs:                   
            the desired sampling rate
    :param Nfft:                 
            the number of bins to compute the Fourier
            transform
    :param stepNotes:            
            the number of F0 per semitone
    :param lengthWindow:         
            the size of the window for the Fourier
            transform
    :param Ot:                   
            the glottal opening coefficient for
            KLGLOTT88
    :param perF0:                
            the number of chirps considered per F0
            value
    :param depthChirpInSemiTone: 
            the maximum value, in semitone, of the
            allowed chirp per F0
                             
    Outputs:

    :returns:
    
        * `F0Table` -  
          the vector containing the values of the fundamental
          frequencies in Hertz (Hz) corresponding to the
          harmonic combs in WF0, i.e. the columns of WF0
          
        * `WF0` -      
          the basis matrix, where each column is a harmonic comb
          generated by KLGLOTT88 (with a sinusoidal model, then
          transformed into the spectral domain)
    
    """
    # generating a filename to keep data:
    filename = str('').join(['wf0_',
                             '_minF0-', str(minF0),
                             '_maxF0-', str(maxF0),
                             '_Fs-', str(Fs),
                             '_Nfft-', str(Nfft),
                             '_stepNotes-', str(stepNotes),
                             '_Ot-', str(Ot),
                             '_perF0-', str(perF0),
                             '_depthChirp-', str(depthChirpInSemiTone),
                             '_analysisWindow-', analysisWindow,
                             '.npz'])
    
    if os.path.isfile(filename) and loadWF0:
        print "Reading WF0 and F0Table from stored arrays."
        struc = np.load(filename)
        return struc['F0Table'], struc['WF0']
    
    print "First time WF0 computed with these parameters, please wait..."
    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(Fs)
    stepNotes=np.double(stepNotes)
    
    # computing the F0 table:
    numberOfF0 = np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1
    F0Table=minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))
    
    numberElementsInWF0 = numberOfF0 * perF0
    
    # computing the desired WF0 matrix
    WF0 = np.zeros([Nfft, numberElementsInWF0],dtype=np.double)
    for fundamentalFrequency in np.arange(numberOfF0):
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0,\
                                 analysisWindowType=analysisWindow)
        # 20100924 trying with hann window
        WF0[:,fundamentalFrequency * perF0] = np.abs(odgdSpec) ** 2
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2 
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(odgdSpec) ** 2
    
    np.savez(filename, F0Table=F0Table, WF0=WF0)
    
    return F0Table, WF0

def generate_WF0_MinQT_chirped(minF0, maxF0, cqtfmax, cqtfmin, cqtbins=48.,
                               Fs=44100., Nfft=2048, stepNotes=4, \
                               lengthWindow=2048, Ot=0.5, perF0=1, \
                               depthChirpInSemiTone=0.5, loadWF0=True,
                               analysisWindow='hanning',
                               atomHopFactor=0.25,
                               cqtWinFunc=np.hanning, verbose=False):
    """\
    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:

    Inputs:

    :param minF0:           
            the minimum value for the fundamental
            frequency (F0)
    :param     maxF0:            
            the maximum value for F0
    :param     cqtfmax: ...
    :param     Fs:                   
            the desired sampling rate
    :param     Nfft:                 
            the number of bins to compute the Fourier
            transform
    :param     stepNotes:            
            the number of F0 per semitone
    :param     lengthWindow:         
            the size of the window for the Fourier
            transform
    :param     Ot:                   
            the glottal opening coefficient for
            KLGLOTT88
    :param     perF0:                
            the number of chirps considered per F0
            value
    :param     depthChirpInSemiTone: 
            the maximum value, in semitone, of the
            allowed chirp per F0
                             
    Outputs:

    :returns:
    
        * `F0Table` - 
          the vector containing the values of the fundamental
          frequencies in Hertz (Hz) corresponding to the
          harmonic combs in WF0, i.e. the columns of WF0

        * `WF0` -      
          the basis matrix, where each column is a harmonic comb
          generated by KLGLOTT88 (with a sinusoidal model, then
          transformed into the spectral domain)
    
    20120828T2358 Horribly slow...
    """
    
    # note: cqtfmax should actually be computed so as to guarantee
    #       the desired Nfft: - not necessary for minqt anymore
    # cqtfmax = np.ceil(3. * Fs / (Nfft * (2**(1./cqtbins) - 1)))
    # strange things happening to FFTLen...
    if verbose>1: print "cqtfmax set to", cqtfmax
    mqt = minqt.MinQTransfo(linFTLen=Nfft,
                            fmin=cqtfmin,
                            fmax=cqtfmax,
                            bins=cqtbins,
                            fs=Fs,
                            perfRast=1,
                            verbose=verbose,
                            winFunc=cqtWinFunc,
                            atomHopFactor=atomHopFactor)
    # getting the right window length:
    #    in particular, it should not be less than the biggest window
    #    used by the minqt transform:
    lengthWindow = np.maximum(lengthWindow,
                              mqt.cqtkernel.FFTLen *
                              (2**(mqt.octaveNr-1))) 
    
    # generating a filename to keep data:
    filename = str('').join(['wf0minqt_',
                             '_minF0-', str(minF0),
                             '_maxF0-', str(maxF0),
                             '_cqtfmax-', str(cqtfmax),
                             '_cqtfmin-', str(cqtfmin),
                             '_cqtbins-', str(cqtbins),
                             '_Fs-', str(int(Fs)),
                             '_Nfft-', str(int(Nfft)),
                             '_atomHopFactor-%.2f' %(atomHopFactor),
                             '_stepNotes-', str(int(stepNotes)),
                             '_Ot-', str(Ot),
                             '_perF0-', str(int(perF0)),
                             '_depthChirp-', str(depthChirpInSemiTone),
                             '_analysisWindow-', analysisWindow,
                             '_lengthWindow-%d' %(int(lengthWindow)),
                             '_cqtwinfunc-', cqtWinFunc.__name__,
                             '.npz'])
    
    if os.path.isfile(filename) and loadWF0:
        print "Reading WF0 and F0Table from stored arrays in %s." %filename
        struc = np.load(filename)
        return struc['F0Table'], struc['WF0'], struc['mqt'].tolist()
    else:
        print "No such file: %s." %filename
        
    
    print "First time WF0 computed with these parameters, please wait..."
    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(Fs)
    stepNotes=np.double(stepNotes)
    
    # computing the F0 table:
    numberOfF0 = np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1
    F0Table=minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))
    
    numberElementsInWF0 = numberOfF0 * perF0
    
    if verbose>2:
        print mqt.cqtkernel
        print mqt.fmin, mqt.fmax, mqt.linFTLen, mqt.octaveNr, mqt.linBins
    
    # computing the desired WF0 matrix
    WF0 = np.zeros([mqt.freqbins,
                    numberElementsInWF0],
                   dtype=np.double)
    
    # slow... try faster : concatenate the odgd, compute one big cqt of that
    # result and extract only the desired frames:
    ##odgds = np.array([])
    for fundamentalFrequency in np.arange(numberOfF0):
        if verbose>0:
            print "    f0 n.", fundamentalFrequency+1, "/", numberOfF0
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0,\
                                 analysisWindowType=analysisWindow)
        mqt.computeTransform(data=odgd)
        # getting the cqt transform at the middle of the window:
        midindex = np.argmin((mqt.datalen_init / 2. - mqt.time_stamps)**2)
        if verbose>1: print midindex, mqt.transfo.shape, WF0.shape
        WF0[:,fundamentalFrequency * perF0] = np.abs(mqt.transfo[:,midindex])**2
        # del mqt.transfo # maybe needed but might slow down even more...
        ##odgds = np.concatenate([odgds, odgd/(np.abs(odgd).max()*1.2)])
        ##print odgds.shape, odgd.shape
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2 
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            mqt.computeTransform(data=odgd)
            # getting the cqt transform at the middle of the window:
            midindex = np.argmin((mqt.datalen_init / 2.
                                  - mqt.time_stamps)**2)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(mqt.transfo[:,midindex]) ** 2
            # del mqt.transfo # idem
            ##odgds = np.concatenate([odgds, odgd/(np.abs(odgd).max()*1.2)])
    ##hybt.computeHybrid(data=odgds)
    ##midindex = np.argmin((lengthWindow / 2. + lengthWindow
    ##                      * np.vstack(np.arange(numberElementsInWF0))
    ##                      - hybt.time_stamps)**2, axis=1)
    ##if verbose>1: print midindex
    ##WF0 = np.abs(hybt.spCQT[:,midindex]) ** 2
    
    np.savez(filename, F0Table=F0Table, WF0=WF0, mqt=mqt)
    
    return F0Table, WF0, mqt #, hybt, odgds

def generate_WF0_NSGTMinQT_chirped(minF0, maxF0, cqtfmax, cqtfmin, cqtbins=48.,
                                   Fs=44100., Nfft=2048, stepNotes=4, \
                                   lengthWindow=2048, Ot=0.5, perF0=1, \
                                   depthChirpInSemiTone=0.5, loadWF0=True,
                                   analysisWindow='hanning',
                                   atomHopFactor=0.25,
                                   cqtWinFunc=np.hanning, verbose=False):
    """
    ::
    
        F0Table, WF0 = generate_WF0_MinCQT_chirped(minF0, maxF0, Fs, Nfft=2048,
            stepNotes=4, lengthWindow=2048,
            Ot=0.5, perF0=2,
            depthChirpInSemiTone=0.5)
            
                                        
    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:

    Inputs:
    
    :param minF0:
        the minimum value for the fundamental
        frequency (F0)
    :param maxF0:
        the maximum value for F0
    :param cqtfmax:
        ...
    :param Fs:
        the desired sampling rate
    :param Nfft:
        the number of bins to compute the Fourier
        transform
    :param stepNotes:
        the number of F0 per semitone
    :param lengthWindow:
        the size of the window for the Fourier
        transform
    :param Ot:
        the glottal opening coefficient for
        KLGLOTT88
    :param perF0:
        the number of chirps considered per F0
        value
    :param depthChirpInSemiTone:
        the maximum value, in semitone, of the
        allowed chirp per F0
                             
    Outputs:
    
    :returns:
        * `F0Table` - the vector containing the values of the fundamental
          frequencies in Hertz (Hz) corresponding to the
          harmonic combs in WF0, i.e. the columns of WF0
          
        * `WF0` - the basis matrix, where each column is a harmonic comb
          generated by KLGLOTT88 (with a sinusoidal model, then
          transformed into the spectral domain)
    
    20120828T2358 Horribly slow...
    """
    # generating a filename to keep data:
    filename = str('').join(['wf0nsgtminqt_',
                             '_minF0-', str(minF0),
                             '_maxF0-', str(maxF0),
                             '_cqtfmax-', str(cqtfmax),
                             '_cqtfmin-', str(cqtfmin),
                             '_cqtbins-', str(cqtbins),
                             '_Fs-', str(int(Fs)),
                             '_Nfft-', str(int(Nfft)),
                             '_atomHopFactor-%.2f' %(atomHopFactor),
                             '_stepNotes-', str(int(stepNotes)),
                             '_Ot-', str(Ot),
                             '_perF0-', str(int(perF0)),
                             '_depthChirp-', str(depthChirpInSemiTone),
                             '_analysisWindow-', analysisWindow,
                             '_cqtwinfunc-', cqtWinFunc.__name__,
                             '.npz'])
    
    if os.path.isfile(filename) and loadWF0:
        print "Reading WF0 and F0Table from stored arrays in %s." %filename
        struc = np.load(filename)
        return struc['F0Table'], struc['WF0'], struc['mqt'].tolist()
    else:
        print "No such file: %s." %filename
        
    
    print "First time WF0 computed with these parameters, please wait..."
    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(Fs)
    stepNotes=np.double(stepNotes)
    
    # computing the F0 table:
    numberOfF0 = np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1
    F0Table=minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))
    
    numberElementsInWF0 = numberOfF0 * perF0
    
    # note: cqtfmax should actually be computed so as to guarantee
    #       the desired Nfft:
    # cqtfmax = np.ceil(3. * Fs / (Nfft * (2**(1./cqtbins) - 1)))
    # strange things happening to FFTLen...
    if verbose>1: print "cqtfmax set to", cqtfmax
    
    mqt = nsgt.nsgtMinQT(ftlen=Nfft,
                         cqtfmin=cqtfmin,
                         cqtfmax=cqtfmax,
                         bpo=cqtbins,
                         fs=Fs,
                         datalength=lengthWindow,
                         )
    if verbose>2:
        print mqt.cqtkernel
        print mqt.fmin, mqt.fmax, mqt.linFTLen, mqt.octaveNr, mqt.linBins
    
    # computing the desired WF0 matrix
    WF0 = np.zeros([mqt.cqtkernel.bins*mqt.octaveNr+
                    mqt.cqtkernel.linBins,
                    numberElementsInWF0],
                   dtype=np.double)
    
    # slow... try faster : concatenate the odgd, compute one big cqt of that
    # result and extract only the desired frames:
    ##odgds = np.array([])
    for fundamentalFrequency in np.arange(numberOfF0):
        if verbose>0:
            print "    f0 n.", fundamentalFrequency+1, "/", numberOfF0
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0,\
                                 analysisWindowType=analysisWindow)
        mqt.computeTransform(data=odgd)
        # getting the cqt transform at the middle of the window:
        midindex = np.argmin((mqt.datalen_init / 2. - mqt.time_stamps)**2)
        if verbose>1: print midindex, mqt.transfo.shape, WF0.shape
        WF0[:,fundamentalFrequency * perF0] = np.abs(mqt.transfo[:,midindex])**2
        # del mqt.transfo # maybe needed but might slow down even more...
        ##odgds = np.concatenate([odgds, odgd/(np.abs(odgd).max()*1.2)])
        ##print odgds.shape, odgd.shape
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2 
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            mqt.computeTransform(data=odgd)
            # getting the cqt transform at the middle of the window:
            midindex = np.argmin((mqt.datalen_init / 2.
                                  - mqt.time_stamps)**2)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(mqt.transfo[:,midindex]) ** 2
            # del mqt.transfo # idem
            ##odgds = np.concatenate([odgds, odgd/(np.abs(odgd).max()*1.2)])
    ##hybt.computeHybrid(data=odgds)
    ##midindex = np.argmin((lengthWindow / 2. + lengthWindow
    ##                      * np.vstack(np.arange(numberElementsInWF0))
    ##                      - hybt.time_stamps)**2, axis=1)
    ##if verbose>1: print midindex
    ##WF0 = np.abs(hybt.spCQT[:,midindex]) ** 2
    
    np.savez(filename, F0Table=F0Table, WF0=WF0, mqt=mqt)
    
    return F0Table, WF0, mqt #, hybt, odgds

def generate_WF0_TR_chirped(transform, minF0, maxF0, stepNotes=4,
                            Ot=0.5, perF0=1, 
                            depthChirpInSemiTone=0.5, loadWF0=True,
                            verbose=False):
    """\
    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:
    
    Inputs:
    
    :param minF0:
        the minimum value for the fundamental
        frequency (F0)
    :param maxF0:
        the maximum value for F0
    :param cqtfmax:  ...
    :param Fs: the desired sampling rate
    :param Nfft:
        the number of bins to compute the Fourier
        transform
    :param stepNotes:
        the number of F0 per semitone
    :param lengthWindow:
        the size of the window for the Fourier
        transform
    :param Ot:
        the glottal opening coefficient for
        KLGLOTT88
    :param perF0:
        the number of chirps considered per F0
        value
    :param depthChirpInSemiTone:
        the maximum value, in semitone, of the
        allowed chirp per F0

    Outputs:
    
    :returns:
      * `F0Table` - 
        the vector containing the values of the fundamental
        frequencies in Hertz (Hz) corresponding to the
        harmonic combs in WF0, i.e. the columns of WF0
      * `WF0` - 
        the basis matrix, where each column is a harmonic comb
        generated by KLGLOTT88 (with a sinusoidal model, then
        transformed into the spectral domain)
        
    Notes:
    20120828T2358 Horribly slow...
    """
    if hasattr(transform, 'octaveNr'):
        lengthWindow = (
            transform.cqtkernel.FFTLen * (2**(transform.octaveNr-1))) 
    elif hasattr(transform, 'cqtkernel'):
        lengthWindow = transform.cqtkernel.linFTLen
    else:
        try:
            lengthWindow = (transform.freqbins - 1) * 2 * 2 # just to be sure
        except AttributeError:
            raise AttributeError(
                'There is something utterly wrong with the desired '+
                'TF representation...\n'+
                'No freqbins attribute!')
    
    # generating a filename to keep data:
    attributesToKeep = ('fmin', 'fmax', 'bins', 'fs', 'winFunc',
                        'freqbins', 'atomHopFactor')
    attributes = [(k.lower()+'-'+str(v) if (k in attributesToKeep and
                                            np.isscalar(v)) else
                   v.__name__ if (k in attributesToKeep and
                                  type(v)==type(lambda x:x)) else
                   '') for k,v in transform.__dict__.items()]
    significantAttributes = [] # keeping only non-empty attributes
    for att in attributes:
        if att != '':
            significantAttributes.append(att)
    attributes = significantAttributes
    attributes.sort()
    print attributes #DEBUG
    
    filename = str('').join(['wf0_%s_' %transform.transformname,
                             '_minF0-', str(minF0),
                             '_maxF0-', str(maxF0),
                             '_stepNotes-', str(int(stepNotes)),
                             '_Ot-', str(Ot),
                             '_perF0-', str(int(perF0)),
                             '_depthChirp-', str(depthChirpInSemiTone),
                             '_lengthWindow-%d' %lengthWindow,
                             '_', str('_').join(attributes),
                              '.npz'])
    #filename = str('').join(['wf0_%s_' %transform.transformname,
    #                         '_', str('_').join(attributes),
    #                         '.npz'])
    ##np.savez(filename, test=None) # to check size of filename on write
    
    # print len(filename), filename #DEBUG
    
    if os.path.isfile(filename) and loadWF0:
        print "Reading WF0 and F0Table from stored arrays in %s." %filename
        struc = np.load(filename)
        return struc['F0Table'], struc['WF0'], struc['tft'].tolist()
    else:
        print "No such file: %s." %filename
        
    
    print "First time WF0 computed with these parameters, please wait..."
    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(transform.fs)
    stepNotes=np.double(stepNotes)

    if hasattr(transform, 'cqtkernel'):
        if hasattr(transform.cqtkernel, 'linFTLen'):
            Nfft = transform.cqtkernel.linFTLen
        else:
            Nfft = transform.cqtkernel.FFTLen
    else:
        Nfft = (transform.freqbins - 1) * 2
    analysisWindow = transform.winFunc(lengthWindow)
    
    # computing the F0 table:
    numberOfF0 = np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1
    F0Table = minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))
    
    numberElementsInWF0 = numberOfF0 * perF0
    
    # computing the desired WF0 matrix
    WF0 = np.zeros([transform.freqbins,
                    numberElementsInWF0],
                   dtype=np.double)
    
    # slow... try faster : concatenate the odgd, compute one big cqt of that
    # result and extract only the desired frames:
    ##odgds = np.array([])
    for fundamentalFrequency in np.arange(numberOfF0):
        if verbose>0:
            print "    f0 n.", fundamentalFrequency+1, "/", numberOfF0
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0,\
                                 analysisWindowType=analysisWindow)
        transform.computeTransform(data=odgd)
        # getting the cqt transform at the middle of the window:
        midindex = np.argmin((transform.datalen_init / 2.
                              - transform.time_stamps)**2)
        if verbose>1: print midindex, transform.transfo.shape, WF0.shape
        WF0[:,fundamentalFrequency * perF0] = np.abs(
            transform.transfo[:,midindex])**2
        # del mqt.transfo # maybe needed but might slow down even more...
        ##odgds = np.concatenate([odgds, odgd/(np.abs(odgd).max()*1.2)])
        ##print odgds.shape, odgd.shape
        if verbose>10: # super debug
            import matplotlib.pyplot as plt
            plt.ion()
            plt.figure(111)
            plt.clf()
            plt.imshow(np.log(np.abs(transform.transfo)**2))
            raw_input('ayayay')
            
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2 
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            transform.computeTransform(data=odgd)
            # getting the cqt transform at the middle of the window:
            midindex = np.argmin((transform.datalen_init / 2.
                                  - transform.time_stamps)**2)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(transform.transfo[:,midindex]) ** 2
            # del mqt.transfo # idem
            ##odgds = np.concatenate([odgds, odgd/(np.abs(odgd).max()*1.2)])
    ##hybt.computeHybrid(data=odgds)
    ##midindex = np.argmin((lengthWindow / 2. + lengthWindow
    ##                      * np.vstack(np.arange(numberElementsInWF0))
    ##                      - hybt.time_stamps)**2, axis=1)
    ##if verbose>1: print midindex
    ##WF0 = np.abs(hybt.spCQT[:,midindex]) ** 2
    
    np.savez(filename, F0Table=F0Table, WF0=WF0, tft=transform)
    
    return F0Table, WF0, transform #, hybt, odgds

def generate_ODGD_spec(F0, Fs, lengthOdgd=2048, Nfft=2048, Ot=0.5, \
                       t0=0.0, analysisWindowType='sinebell'): 
    """
    generateODGDspec:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """
    
    # converting input to double:
    F0 = np.double(F0)
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType=='sinebell':
        analysisWindow = sinebell(lengthOdgd)
    elif analysisWindowType=='hanning' or \
             analysisWindowType=='hanning':
        analysisWindow = hann(lengthOdgd)
    elif analysisWindowType=='rectangular':
        analysisWindow = np.ones(lengthOdgd)
    elif len(analysisWindowType)==lengthOdgd:
        analysisWindow = analysisWindowType
    else:
        raise ValueError("Analysis window not understood.")
        
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / F0)
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot
    
    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 \
                 * (np.exp(-temp_array) \
                    + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                    - (6 * (1 - np.exp(-temp_array)) \
                       / (temp_array ** 2))) \
                       / temp_array
    
    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    #print timeStamps.shape#DEBUG
    #print lengthOdgd#DEBUG
    odgd = np.exp(np.outer(2.0 * 1j * np.pi * F0 * frequency_numbers, \
                           timeStamps)) \
                           * np.outer(amplitudes, np.ones(lengthOdgd))
    odgd = np.sum(odgd, axis=0)
    
    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)
    
    return odgd, odgdSpectrum

def generate_ODGD_spec_inharmo(F0, Fs, lengthOdgd=2048, Nfft=2048, Ot=0.5, \
                               t0=0.0, analysisWindowType='sinebell',
                               inharmonicity=0.5): 
    """
    generateODGDspec:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """
    
    # converting input to double:
    F0 = np.double(F0)
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType=='sinebell':
        analysisWindow = sinebell(lengthOdgd)
    elif analysisWindowType=='hanning' or \
               analysisWindowType=='hanning':
            analysisWindow = hann(lengthOdgd)
    elif analysisWindowType=='rectangular':
        analysisWindow = np.ones(lengthOdgd)
    else:
        raise ValueError("Analysis window not understood.")
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / F0)
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot
    
    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 \
                 * (np.exp(-temp_array) \
                    + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                    - (6 * (1 - np.exp(-temp_array)) \
                       / (temp_array ** 2))) \
                       / temp_array
    
    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    odgd = np.exp(np.outer(2.0 * 1j * np.pi * F0 * frequency_numbers, \
                           timeStamps)) \
                           * np.outer(amplitudes, np.ones(lengthOdgd))
    odgd = np.sum(odgd, axis=0)
    
    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)
    
    return odgd, odgdSpectrum

def generate_ODGD_spec_chirped(F1, F2, Fs, lengthOdgd=2048, Nfft=2048, \
                               Ot=0.5, t0=0.0, \
                               analysisWindowType='sinebell'):
    """
    generateODGDspecChirped:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """
    
    # converting input to double:
    F1 = np.double(F1)
    F2 = np.double(F2)
    F0 = np.double(F1 + F2) / 2.0
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType == 'sinebell':
        analysisWindow = sinebell(lengthOdgd)
    elif analysisWindowType == 'hanning' or \
             analysisWindowType == 'hann':
        analysisWindow = hann(lengthOdgd)
    elif analysisWindowType == 'rectangular':
        analysisWindow = np.ones(lengthOdgd)
    else:
        raise ValueError("Analysis window not understood.")
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / np.max([F1, F2]))
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot
    
    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 * \
                 (np.exp(-temp_array) \
                  + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                  - (6 * (1 - np.exp(-temp_array)) \
                     / (temp_array ** 2))) \
                  / temp_array
    
    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    odgd = np.exp(2.0 * 1j * np.pi \
                  * (np.outer(F1 * frequency_numbers,timeStamps) \
                     + np.outer((F2 - F1) \
                                * frequency_numbers,timeStamps ** 2) \
                     / (2 * lengthOdgd / Fs))) \
                     * np.outer(amplitudes,np.ones(lengthOdgd))
    odgd = np.sum(odgd,axis=0)
    
    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)
    
    return odgd, odgdSpectrum

def generateHannBasis(numberFrequencyBins, sizeOfFourier, Fs, \
                      frequencyScale='linear', numberOfBasis=20, \
                      overlap=.75):
    """Generates a collection of Hann functions, spaced across the
    frequency axis, as desired by the user (and if implemented),
    targetting the given number of basis and adapting the extent (or bandwidth)
    of each function (over the frequencies) to that number and according
    to the desired overlap between these windows.
    
    
    """
    isScaleRecognized = False
    if frequencyScale == 'linear':
        # number of windows generated:
        numberOfWindowsForUnit = np.ceil(1.0 / (1.0 - overlap))
        # recomputing the overlap to exactly fit the entire
        # number of windows:
        overlap = 1.0 - 1.0 / np.double(numberOfWindowsForUnit)
        # length of the sine window - that is also to say: bandwidth
        # of the sine window:
        lengthSineWindow = np.ceil(numberFrequencyBins \
                                   / ((1.0 - overlap) \
                                      * (numberOfBasis - 1) + 1 \
                                      - 2.0 * overlap))
        # even window length, for convenience:
        lengthSineWindow = 2.0 * np.floor(lengthSineWindow / 2.0) 
        
        # for later compatibility with other frequency scales:
        mappingFrequency = np.arange(numberFrequencyBins) 
        
        # size of the "big" window
        sizeBigWindow = 2.0 * numberFrequencyBins
        
        # centers for each window
        ## the first window is centered at, in number of window:
        firstWindowCenter = -numberOfWindowsForUnit + 1
        ## and the last is at
        lastWindowCenter = numberOfBasis - numberOfWindowsForUnit + 1
        ## center positions in number of frequency bins
        sineCenters = np.round(\
            np.arange(firstWindowCenter, lastWindowCenter) \
            * (1 - overlap) * np.double(lengthSineWindow) \
            + lengthSineWindow / 2.0)
        
        # For future purpose: to use different frequency scales
        isScaleRecognized = True
        
    # For frequency scale in logarithm (such as ERB scales) 
    if frequencyScale == 'log':
        isScaleRecognized = False
        
    # checking whether the required scale is recognized
    if not(isScaleRecognized):
        print "The desired feature for frequencyScale is not recognized yet..."
        return 0
    
    # the shape of one window:
    prototypeSineWindow = hann(lengthSineWindow)
    # adding zeroes on both sides, such that we do not need to check
    # for boundaries
    bigWindow = np.zeros([sizeBigWindow * 2, 1])
    bigWindow[(sizeBigWindow - lengthSineWindow / 2.0):\
              (sizeBigWindow + lengthSineWindow / 2.0)] \
              = np.vstack(prototypeSineWindow)
    
    WGAMMA = np.zeros([numberFrequencyBins, numberOfBasis])
    
    for p in np.arange(numberOfBasis):
        WGAMMA[:, p] = np.hstack(bigWindow[np.int32(mappingFrequency \
                                                    - sineCenters[p] \
                                                    + sizeBigWindow)])
        
    return WGAMMA
