#!/usr/bin/python

# copyright (C) 2011 - 2013 Jean-Louis Durrieu
# 

"""Constant-Q transform after the work by C. Scholkhuber and A. Klapuri
2010 [SK2010]_

Adaptation of the Constant Q transform as presented in

.. [SK2010] Schoerkhuber, C. and Klapuri, A.,
   \"Constant-Q transform toolbox for music processing,\"
   submitted to the 7th Sound and Music Computing Conference, Barcelona, Spain.

Comments beginning with '%' and '%%' are retained from the original Matlab
code. 

Python/Numpy/Scipy by
Jean-Louis Durrieu, EPFL, 2012 - 2013
"""

import numpy as np
import scipy.signal as spsig  # for the windows and filters
# import scipy.sparse as spspa
import scipy.interpolate as spinterp
# from .. import audioObject as ao # for the stft and istft
from stft import stft, istft

from ..tools.utils import nextpow2, sqrt_blackmanharris

class CQTKernel(object):
    """The CQT Kernel contains everything that can be
    precomputed for Constant-Q transforms. This
    relies on [SK2010]_, and therefore computes a Kernel
    for a single octave. It is then efficiently used to
    compute the decomposition on the different octaves by
    downsampling the signal.

    Parameters:

     fmax
      The maximum desired central frequency
      
     bins
      The number of bins per octave
     
     fs
      Sampling rate of the audio files
      
     q
      parameter that controls the quality
     
     atomHopFactor
      hopsize rate (0.25 is a hopsize of 25% the size of the windows)
      between successive analysis windows
     
     thresh
      threshold value for sparsifying the kernel
      (Note: in this implementation, we do not use the sparsity, more
      efficiency could be achieved by considering it)
      
     winFunc (python function that outputs an array)
      the analysis window function

     perfRast
      whether computing rasterized version or not
      (if so, the decompositions at all scales will have the same
      number of frames, otherwise, each lower analysis octave will have
      half as many frames as the direct upper analysis octave.)
     
    Attributes:
    
     sparKernel
     weight
     atomHOP
     FFTLen
     fftOLP
     fftHOP
     bins
     winNr
     Nk_max
     Q 
     fmin
     fmax
     frequencies
     perfRast
     first_center
     fs 
     winFunc
     thresh
     q
    
    
    """
    def __init__(self, fmax, bins, fs,
                 q = 1,
                 atomHopFactor = 0.25,
                 thresh = 0.0005, 
                 winFunc = sqrt_blackmanharris,
                 perfRast = 0
                 ):
        """
        Generate the CQT kernel for one octave
        """
        # %% define
        # checking that fmax<fs/2
        if fmax >= fs/2.:
            raise ValueError("fmax (%s) is too big for fs (%s)"
                             %(str(fmax), str(fs)))
        # fmax = np.minimum(fmax, fs/2.)
        fmin = (fmax / 2.) * (2 ** (1./bins))
        Q = 1. / (2 ** (1./bins) - 1)
        Q = Q * q
        Nk_max = Q * fs / fmin
        Nk_max = np.round(Nk_max) # length of the largest atom [samples]
        
        # %% Compute FFT size, FFT hop, atom hop, ...
        # %  length of the shortest atom [samples]
        Nk_min = np.round( Q * fs /
                           (fmin * (2**((bins-1.)/bins))) )
        # %atom hop size
        # DJL: nextpow2, so that at all octaves, the frames are centered on
        #     integer valued samples of the (decimated) signal.
        # atomHOP = np.round(Nk_min * atomHopFactor)
        atomHOP = nextpow2(Nk_min * atomHopFactor)/2
        # %first possible center position within the frame
        first_center = np.ceil(Nk_max / 2.)
        # %lock the first center to an integer multiple of the atom hop size
        first_center = atomHOP * np.ceil(first_center * 1. / atomHOP)
        # %use smallest possible FFT size (increase sparsity)
        FFTLen = nextpow2(first_center + np.ceil(Nk_max / 2.))
        ## DEBUG DEBUG DEBUG
        ## print "FFTLen", FFTLen, "Nk_min", Nk_min
        
        if perfRast:
            # %number of temporal atoms per FFT Frame
            # DJL: 20130314: why is this different from
            #     non rasterized transform? adding 1 again, but has to be
            #     further studied...
            winNr = np.floor((FFTLen - np.ceil(Nk_max / 2.) -
                              first_center) / atomHOP) + 1
            if winNr == 0:
                FFTLen = FFTLen * 2
                winNr = np.floor((FFTLen - np.ceil(Nk_max / 2.)-
                                  first_center) / atomHOP)
        else:
            # %number of temporal atoms per FFT Frame
            winNr = np.floor((FFTLen - np.ceil(Nk_max / 2.) -
                              first_center) / atomHOP) + 1 
            
        last_center = first_center + (winNr - 1.) * atomHOP
        # % hop size of FFT frames
        fftHOP = (last_center + atomHOP) - first_center
        # %overlap of FFT frames in percent
        fftOLP = (FFTLen - fftHOP) * (1. / FFTLen)*100.
        
        # %% init variables
        tempKernel = np.zeros(FFTLen, dtype=np.complex)
        sparKernel = np.zeros([bins * winNr, FFTLen],
                              dtype=np.complex)
        
        frequencies = []
        
        # %% Compute kernel
        atomInd = 0
        for k in np.arange(bins): 
            
            Nk = np.round(Q * fs /
                          (fmin * (2**((k*1.)/bins))))
            # print Nk
            winFct = winFunc(Nk)
            
            fk = fmin * (2**((k*1.)/bins))
            frequencies.append(fk)
            tempKernelBin = (
                (winFct *1. / Nk) *
                np.exp(2 * np.pi * 1j * fk * np.arange(Nk) / fs)
                )
            atomOffset = first_center - np.ceil(Nk / 2.)
            
            for i in np.arange(winNr): 
                shift = atomOffset + i * atomHOP
                # print shift
                tempKernel[shift:(Nk+shift)] = tempKernelBin 
                atomInd = atomInd + 1
                specKernel= np.fft.fft(tempKernel)
                specKernel[np.abs(specKernel)<=thresh] = 0
                # sparsifying this?
                # sparKernel = sparse([sparKernel; specKernel])
                sparKernel[int(i + k * winNr)] = specKernel
                # %reset window   
                tempKernel = np.zeros(FFTLen, dtype=np.complex)  
                
        sparKernel = (sparKernel.T) * 1. / FFTLen
        
        # %% Normalize the magnitudes of the atoms
        wx1 = np.argmax(sparKernel[:, 0])
        wx2 = np.argmax(sparKernel[:,-1])
        wK = sparKernel[wx1:wx2,:]
        wK = np.diag(np.dot(wK, np.conjugate(wK.T)))
        wK = wK[int(np.round(1./q)):\
                int(len(wK) - np.round(1./q) - 1)]
        weight = 1. / np.mean(np.abs(wK))
        weight *= (fftHOP * 1. / FFTLen)
        # %sqrt because the same weight is applied in icqt again
        weight = np.sqrt(weight)
        sparKernel *= weight
        
        self.sparKernel = np.ascontiguousarray(sparKernel)
        self.weight = weight
        self.atomHOP = atomHOP
        self.FFTLen = FFTLen
        self.fftOLP = fftOLP
        self.fftHOP = fftHOP
        self.bins = bins
        self.winNr = winNr
        self.Nk_max = Nk_max
        self.Q = Q
        self.fmin = fmin
        self.fmax = fmax
        self.frequencies = frequencies
        self.perfRast = perfRast
        self.first_center = first_center
        self.fs = fs
        self.winFunc = winFunc
        self.thresh = thresh
        self.q = q
        
    def __str__(self):
        description = "CQT Kernel structure, containing:\n"
        for k, v in self.__dict__.items():
            description += str(k) + ': ' + str(v) + '\n'
        return description


class HybridCQTKernel(CQTKernel):
    """Hybrid CQT/Linear kernel
    """
    def __init__(self, **kwargs):
        super(HybridCQTKernel, self).__init__(**kwargs)
        self.computeMissingLinearFreqKernel()
        
    def computeMissingLinearFreqKernel(self):
        """Compute the missing (high) frequency
        components, and make a similar Kernel for them.
        
        We can use this for the first octave (the highest frequency octave)
        to extend the high frequencies. Actually, this can be used to compute
        a hybrid CQT transform on the low frequencies, while keeping linear
        freqs in the high spectrum, and still benefiting from the invertibility
        of the CQT transform by Schoerkhuber and Klapuri
        """
        # %% init variables
        tempKernel = np.zeros(self.FFTLen,
                              dtype=np.complex)
        Nk = self.Nk_max # self.FFTLen # self.Nk_max 
        pfirst = int(self.fmax*(2.**(1./self.bins))/self.fs*Nk)
        # *(2.**((self.bins-1)/self.bins))/self.fs*Nk)
        plast = int(Nk/2.) + 1
        first_center = np.ceil(Nk / 2.)
        
        self.linBins = plast - pfirst
        
        sparKernel = np.zeros([self.linBins * self.winNr, self.FFTLen],
                              dtype=np.complex)
        # %% Compute kernel
        for p in np.arange(pfirst, plast):
            # print Nk
            winFct = self.winFunc(Nk)
            
            fk = p * 1. / Nk * self.fs# fmin * (2**((k*1.)/bins))
            self.frequencies.append(fk)
            tempKernelBin = ( 
                (winFct * 1. / Nk) *
                np.exp(2 * np.pi * 1j * p * np.arange(Nk) / Nk)
                ) # Fourier exponential function!
            atomOffset = first_center - np.ceil(Nk / 2.);
            
            for i in np.arange(self.winNr): 
                shift = atomOffset + i * self.atomHOP
                ## print shift, Nk
                tempKernel[shift:(Nk+shift)] = tempKernelBin 
                specKernel= np.fft.fft(tempKernel)
                specKernel[np.abs(specKernel)<=self.thresh] = 0
                # sparsifying this?
                ## sparKernel = sparse([sparKernel; specKernel])
                ## print i, p, int(i + p * self.winNr)
                sparKernel[int(i + (p-pfirst) * self.winNr)] = specKernel
                # %reset window
                tempKernel = np.zeros(self.FFTLen, dtype=np.complex)  
                
        sparKernel = (sparKernel.T) * 1. / self.FFTLen
        
        # %% Normalize the magnitudes of the atoms
        wx1 = np.argmax(sparKernel[:,0])
        wx2 = np.argmax(sparKernel[:,-1])
        wK = sparKernel[wx1:wx2,:]
        wK = np.diag(np.dot(wK, np.conjugate(wK.T)))
        wK = wK[int(np.round(1./self.q)):\
                int(len(wK) - np.round(1./self.q) - 1)]
        weight = 1. / np.mean(np.abs(wK))
        weight *= (self.fftHOP * 1. / self.FFTLen)
        # %sqrt because the same weight is applied in icqt again
        weight = np.sqrt(weight)
        sparKernel *= weight
        self.linearSparKernel = np.ascontiguousarray(sparKernel)


class MinQTKernel(CQTKernel):
    """Min Q Transform Kernel
    """
    def __init__(self, bins, fmax, fs, linFTLen=2048,
                 **kwargs):
        """Initiates and computes the MinQT kernel,
        by determining the frequency of split - fmax - and having 2 different
        frequency ranges (log for CQT part and lin for linear part)
        """
        Q = 1./(2**(1./bins)-1)
        Kmax = int(np.ceil(Q))
        fmax = 2**(-1./bins) * Kmax * fs * 1. / linFTLen
        self.Q = Q # minimum Q value for the whole transform
        self.Kmax = Kmax
        self.linFTLen = linFTLen
        self.fs = fs
        self.fmax = fmax
        self.bins = bins
        super(MinQTKernel, self).__init__(fmax=self.fmax, fs=self.fs,
                                          bins=self.bins,
                                          **kwargs)
        # instead of linear Kernel, compute with FFT
        #    the number of additional linear freq. bins:
        self.linBins = linFTLen/2 - Kmax + 1
        self.linWindow = self.winFunc(linFTLen)
        ## legacy: don't use the following kernel, this is too slow!
        # self.computeMissingLinearFreqKernel()
        
    def computeMissingLinearFreqKernel(self):
        """Compute the missing (high) frequency
        components, and make a similar Kernel for them.
        
        We can use this for the first octave (the highest frequency octave)
        to extend the high frequencies. Actually, this can be used to compute
        a hybrid CQT transform on the low frequencies, while keeping linear
        freqs in the high spectrum, and still benefiting from the invertibility
        of the CQT transform by Schoerkhuber and Klapuri
        """
        # %% init variables
        tempKernel = np.zeros(self.linFTLen,
                              dtype=np.complex)
        N = self.linFTLen
        pfirst = int(self.Kmax)
        plast = int(N/2 + 1)
        first_center = np.ceil(N / 2.)
        
        self.linBins = plast - pfirst
        
        sparKernel = np.zeros([self.linBins, self.linFTLen],
                              dtype=np.complex)
        # %% Compute kernel
        winFct = self.winFunc(N)
        for p in np.arange(pfirst, plast):
            
            fk = p * 1. / N * self.fs# fmin * (2**((k*1.)/bins))
            self.frequencies.append(fk)
            tempKernelBin = ( 
                (winFct * 1. / N) *
                np.exp(2 * np.pi * 1j * p * np.arange(N) / N)
                ) # Fourier exponential function!
            
            # to have time aligned atoms:
            atomOffset = first_center - np.ceil(N / 2.);
            
            i = 0 # only one window per frame of FT for linear part
            shift = atomOffset 
            tempKernel[shift:(Nk+shift)] = tempKernelBin 
            specKernel= np.fft.fft(tempKernel)
            specKernel[np.abs(specKernel)<=self.thresh] = 0
            # sparsifying this?
            ## sparKernel = sparse([sparKernel; specKernel])
            ## print i, p, int(i + p * self.winNr)
            sparKernel[int(i + (p-pfirst) * self.winNr)] = specKernel
            # %reset window
            tempKernel = np.zeros(self.FFTLen, dtype=np.complex)  
                
        sparKernel = (sparKernel.T) * 1. / self.FFTLen
        
        # %% Normalize the magnitudes of the atoms
        wx1 = np.argmax(sparKernel[:,0])
        wx2 = np.argmax(sparKernel[:,-1])
        wK = sparKernel[wx1:wx2,:]
        wK = np.diag(np.dot(wK, np.conjugate(wK.T)))
        wK = wK[int(np.round(1./self.q)):\
                int(len(wK) - np.round(1./self.q) - 1)]
        weight = 1. / np.mean(np.abs(wK))
        weight *= (self.fftHOP * 1. / self.FFTLen)
        # %sqrt because the same weight is applied in icqt again
        weight = np.sqrt(weight)
        sparKernel *= weight
        self.linearSparKernel = np.ascontiguousarray(sparKernel)


class CQTransfo(object):
    """Constant Q Transform"""
    transformname = 'cqt'
    def __init__(self,
                 fmin, fmax, bins, fs,
                 q=1,
                 atomHopFactor=0.25,
                 thresh=0.0005, 
                 winFunc=sqrt_blackmanharris,
                 perfRast=0,
                 cqtkernel=None,
                 lowPassCoeffs=None,
                 data=None,
                 verbose=0, **kwargs): 
        """cqt
        from the reference implementation by C. Scholkhuber and A. Klapuri
        Matlab program downloaded from
        http://www.elec.qmul.ac.uk/people/anssik/cqt/
        on 22/08/2012
        """
        self.verbose = verbose
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins # bins per octave
        self.bpo = bins
        self.fs = fs
        self.q = q
        self.atomHopFactor = atomHopFactor
        self.thresh = thresh
        if winFunc is None:
            winFunc = sqrt_blackmanharris
        self.winFunc = winFunc
        self.perfRast = perfRast
        # %  % define
        self.octaveNr = np.ceil(np.log2(fmax * 1. / fmin))
        self.freqbins = bins * self.octaveNr
        # %set fmin to actual value
        self.fmin = (fmax / (2**self.octaveNr)) * 2**(1./bins)
        # %% design lowpass filter
        if lowPassCoeffs is None:
            self.LPorder = 6 # %order of the anti-aliasing filter
            self.cutoff = 0.5
            # %design f_nyquist/2-lowpass filter
            self.B, self.A = spsig.butter(N=self.LPorder,
                                          Wn=self.cutoff,
                                          btype='low')
            
        # %% design kernel for one octave 
        if cqtkernel is None:
            self.cqtkernel = CQTKernel(
                fmax=fmax,
                bins=bins,
                fs=fs,
                q=q,
                atomHopFactor=atomHopFactor,
                thresh=thresh, 
                winFunc=winFunc,
                perfRast=perfRast)
        else:
            self.cqtkernel = cqtkernel
            
        if data is not None:
            self.computeTransform(data=data) # more general from subclasses
            
    def computeTransform(self, data):
        """Computes the desired transform
        """
        return self.computeCQT(data)
    
    def computeCQT(self, data):
        # %% calculate CQT
        if self.verbose: print "Computing CQT"
        if hasattr(self, '_spCQT'):
            del self._spCQT
        cqtkernel = self.cqtkernel
        self.datalen_init = data.shape[0]
        self.cellCQT = {} # contrary to CSAK original code, keep this always
        self.maxBlock = (
            cqtkernel.FFTLen *
            (2**(self.octaveNr-1))) # %largest FFT Block (virtual)
        self.suffixZeros = self.maxBlock
        self.prefixZeros = self.maxBlock
        # %zeropadding
        x = (
            np.concatenate([np.zeros(self.prefixZeros),
                            data,
                            np.zeros(self.suffixZeros)])
            )
        OVRLP = cqtkernel.FFTLen - cqtkernel.fftHOP
        # %conjugate spectral kernel for cqt transformation
        K = np.ascontiguousarray(np.conjugate(cqtkernel.sparKernel.T))
        self.nframes = []
        
        if not self.cqtkernel.perfRast: # this is working well enough but...
            for i in np.arange(self.octaveNr):
                if self.verbose:
                    print "    Octave n.", i, "out of", self.octaveNr
                # %generating FFT blocks
                nframes = (
                    np.floor(((x.size - cqtkernel.FFTLen) /
                              cqtkernel.fftHOP) + 1)
                    )
                self.nframes += [nframes,]
                self.cellCQT[i] = np.zeros([self.cqtkernel.bins *
                                            self.cqtkernel.winNr,
                                            nframes],
                                           dtype=np.complex)
                for n in np.arange(nframes):
                    if self.verbose>1:
                        print "        nframe", n, "out of", nframes
                    # %applying fft to each column (each FFT frame)
                    framestart = n*cqtkernel.fftHOP
                    framestop = framestart + cqtkernel.FFTLen
                    X = np.fft.fft(x[framestart:framestop],
                                   n=cqtkernel.FFTLen)
                    # %calculating cqt coefficients for all FFT frames
                    # for this octave
                    self.cellCQT[i][:,n] = np.dot(K, X)
                if i != self.octaveNr:
                    x = spsig.filtfilt(self.B,self.A,x,) # %antialiasing filter
                    x = x[::2]# %drop samplerate by 2
        else: # ... this loop seems overly highly unoptimized! 
            atomNr = self.cqtkernel.winNr
            # number of hops that are not computed at the beginning of the
            # data (beginning the first "window" starts at first_center,
            # and not at 0):
            emptyHops = self.cqtkernel.first_center *1./self.cqtkernel.atomHOP
            ahop = self.cqtkernel.atomHOP # in samples, for the first octave
            for i in np.arange(self.octaveNr):
                if self.verbose:
                    print "    Octave n.", i+1, "out of", self.octaveNr
                inc = ahop / (2.**i)
                # bin frequency numbers in the CQT transfo:
                binVec = np.int32(
                    self.cqtkernel.bins * (self.octaveNr-i-1) +
                    np.arange(self.cqtkernel.bins)
                    )
                # the number of frames in total to skip at the beginning of the
                # CQT because of downsampling at each octave (so the higher
                # the octave number, the less to skip)
                # and since we want all the windows to be aligned,
                # in order to have the first window of the lowest frequency
                # octave to be centered with the first "effective" window in
                # high freqs, we need to add as many 0s as follows, for each
                # octave:
                drop = emptyHops * (2**(self.octaveNr-i-1) - 1)
                # There are too many frames computed at each octave, and we
                # have to drop them so as to obtain aligned frames at all
                # octaves.
                if self.verbose>2:
                    print "        drop", drop # DEBUG
                nframes = (
                    np.floor(((x.size - cqtkernel.FFTLen)/cqtkernel.fftHOP)+1)
                    )
                self.nframes += [nframes,]
                if i==0:
                    self._spCQT = np.zeros([self.cqtkernel.bins
                                            * self.octaveNr,
                                            nframes * atomNr],
                                           dtype=np.complex)
                
                CQTframe = np.zeros([self.cqtkernel.bins * atomNr, nframes],
                                    dtype=np.complex)
                XX = np.zeros([self.cqtkernel.FFTLen, nframes],
                              dtype=np.complex)
                
                nshifts = int(2**i)
                for n in np.arange(nframes):
                    if self.verbose>2:
                        print "            nframe", n+1, "out of", nframes
                    # %applying fft to each column (each FFT frame)
                    framestart = n * self.cqtkernel.fftHOP
                    framestop = framestart + self.cqtkernel.FFTLen
                    ##X = np.fft.fft(x[framestart:framestop],
                    ##               n=cqtkernel.FFTLen)
                    XX[:,n]= np.fft.fft(x[framestart:framestop],
                                        n=cqtkernel.FFTLen)
                for nshift in np.arange(2**i):
                    # each shift in this loop corresponds to a different
                    # "starting point", aiming at compensating the arbitrary
                    # time stamps resulting from the subsampling, before
                    # each octave.
                    # 
                    if self.verbose>1:
                        print "        shift n.", nshift+1, "out of", (2**i)
                    shift = nshift * inc
                    # in original matlab code, this is conjugated,
                    # but K is here already conjugated
                    phShiftVec = np.exp(1j * 2 * np.pi *
                                        np.arange(self.cqtkernel.FFTLen) *
                                        shift / self.cqtkernel.FFTLen)
                    Kshift = K * phShiftVec
                    
                    # %calculating cqt coefficients for all FFT frames
                    # for this octave
                    CQTframe = np.dot(Kshift, XX, out=CQTframe)
                    if nshift==0:
                        self.cellCQT[i] = np.copy(CQTframe)
                    if atomNr>1:
                        if self.verbose>1:
                            print "            filling in CQT matrix, "+\
                                  "many windows per frame"
                        for nb, b in enumerate(binVec):
                            # nb is the bin number in the current octave
                            #    representation
                            # b is the bin number in the final CQT
                            #    representation
                            for a in range(int(atomNr)):
                                # a is the number of the "sub-window"
                                # as computed by the above matrix product
                                #    Kshift * XX
                                self._spCQT[
                                    b,
                                    int(nshift+a*nshifts):\
                                    int((nframes)*atomNr*nshifts):\
                                    int(atomNr*nshifts)] = (
                                        CQTframe[nb*atomNr+a]
                                    )
                            # doing the following shift only after all
                            # shifts have been computed, to realign the time
                            # series
                            ##self._spCQT[b, :(self._spCQT.shape[1]
                            ##                 -int(drop*nshifts))] = (
                            ##   self._spCQT[b, int(drop*nshifts):]
                            ##    )
                    else:
                        # TODO: not sure whether this piece of code works:
                        #    needs testing!
                        if self.verbose>1:
                            print "            filling in CQT matrix, "+\
                                  "one window per frame"
                        self._spCQT[binVec,
                                    int(nshift):(nframes*nshifts):nshifts] = (
                            CQTframe[:,:] #CQTframe[:,int(drop*nshifts):]
                            )
                
                # aligning the time series, with the right drop:
                for nb, b in enumerate(binVec):
                    self._spCQT[b,:(self._spCQT.shape[1]-int(drop*nshifts))]=(
                        self._spCQT[b, int(drop*nshifts):]
                        )
                
                if i != self.octaveNr-1:
                    x = spsig.filtfilt(self.B,self.A,x) # %anti aliasing filter
                    x = x[::2]# %drop samplerate by 2

    def _set_transfo(self, X):
        """for now, we return the spCQT: matrix form of the transform
        """
        Xshape = X.shape
        # spShape = self._spCQT.shape
        if Xshape[0] == self.freqbins: # spShape:
            self._spCQT = np.copy(X)
            self.spCQT2CellCQT()
        else:
            raise ValueError('Transfo not of the right size: '+str(X.shape)+
                             ' instead of '+str(self.freqbins))
    
    def _get_transfo(self):
        """for now, we return the spCQT: matrix form of the transform
        """
        return self._get_spCQT()
    
    def _del_transfo(self):
        """erasing everything
        """
        del self._spCQT
        del self.cellCQT
    
    transfo = property(fget=_get_transfo,
                       fdel=_del_transfo,
                       fset=_set_transfo,
                       doc="returns the computed transform")
    
    def _get_time_stamps(self):
        """getting the time stamps for spCQT,
        which also includes the prefix zeros that were pre-pended to the data.
        (in samples)
        """
        nframes = self.nframes[0] * self.cqtkernel.winNr  # self.spCQT.shape[1]
        time_stamps = (
            np.arange(nframes) * self.cqtkernel.atomHOP
            + self.cqtkernel.first_center *
            2**(self.octaveNr-1)
            - self.prefixZeros
            )
        
        return time_stamps
    
    time_stamps = property(fget=_get_time_stamps,
                           doc="time stamps for spCQT")
    
    def _get_frequencies(self):
        return self._compute_frequencies()
    
    freq_stamps = property(fget=_get_frequencies,
                           doc="frequency stamps for spCQT")
    
    def _compute_frequencies(self):
        freqs = (
            self.cqtkernel.fmin
            * 2 ** (np.arange(self.cqtkernel.bins *
                              self.octaveNr) / self.cqtkernel.bins
                    - (self.octaveNr-1))
            )
        return freqs

    def _get_qvalues(self):
        Q = (
            self.freq_stamps[:-1] /
            np.diff(self.freq_stamps))
        return Q
    
    qValues = property(fget= _get_qvalues,
                       doc="$Q$ values, approximated")
    
    # make spCQT a property: if not using rasterized version, then this matrix
    # needs to be explicitly computed (contrary to cellCQT)
    #
    # _get_spCQT does therefore not need to be called explicitly (but could)
    def _get_spCQT(self):
        if not hasattr(self, '_spCQT') and hasattr(self, 'cellCQT'):
            # compute the full cqt from self.cellCQT
            self.cellCQT2spCQT()
        elif not hasattr(self, '_spCQT') and not hasattr(self, 'cellCQT'):
            raise AttributeError("Some CQT should be computed before"+\
                                 " getting it.")
        return self._spCQT
    
    def _set_spCQT(self, value):
        self._spCQT = value
        if self.verbose>1:
            print "    Setting spCQT, resetting cellCQT"
        self.spCQT2CellCQT()
    
    spCQT = property(fget=_get_spCQT,
                     fset=_set_spCQT,
                     doc=("spCQT: the constant Q transform, "+
                          "in a readable format."))
    
    def _get_cellCQT(self):
        self.checkCQTexists()
        
        if hasattr(self, 'cellCQT'):
            return self.cellCQT
        
        # now, if not, according to check CQTexists, self._spCQT exists.
        self.spCQT2CellCQT()
        return self.cellCQT
    
    def plotCellCQT(self):
        if not hasattr(self, 'cellCQT'):
            raise AttributeError("This CQT instance has no cellCQT. "+
                                 "\nUse computeCQT for this.")
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure()
        valueLims = [0,-1e10]
        for i in np.arange(self.octaveNr):
            cqt = self.cellCQT[i]
            ax = fig.add_subplot(
                self.octaveNr, 1, i+1)
            ax.imshow(np.log(np.abs(cqt)),
                      interpolation='nearest',
                      origin='lower')
            if np.log(np.abs(cqt)).max()>valueLims[1]:
                valueLims[1] = np.log(np.abs(cqt)).max()
                valueLims[0] = valueLims[1] - 10
        for ax in fig.get_axes():
            ax.get_images()[0].set_clim(valueLims)
            
    def checkCQTexists(self):
        if not hasattr(self, '_spCQT') and not hasattr(self, 'cellCQT'):
            raise AttributeError("Compute the CQT with computeCQT before"+\
                                 "requesting any of its representation")
        return True
    
    def invertFromSpCQT(self):
        """Assuming we have self.spCQT, and not self.cellCQT, we recompute
        self.cellCQT from self.spCQT, and then invert as usual.
        
        NB: here, self.cellCQT is written over, if it existed. 
        """
        if self.verbose:
            print "Computing the cell representation from the matrix "+\
                  "representation..."
        self.spCQT2CellCQT()
        if self.verbose:
            print "... and then computing the inverse from the newly "+\
                  "generated cells."
        return self.invertFromCellCQT()

    def invertFromSpCQTRast(self):
        """this inverts the transform, if perfRast, then this means
        we can invert each hop of the different octaves. 
        """
        self._check_attr_inversion()
        # inverting Kernel:
        K = np.ascontiguousarray(self.cqtkernel.sparKernel)
        y = np.zeros(np.ceil(self.datalen_init /
                             (2.**(self.octaveNr-1))))
        atomNr = self.cqtkernel.winNr
        emptyHops = self.cqtkernel.first_center * 1. / self.cqtkernel.atomHOP
        ahop = self.cqtkernel.atomHOP # in samples, for the first octave
        tmpSpCQT = np.copy(self._spCQT) # copying so we can modify it later
        # from low to high octave:
        for noct in range(int(self.octaveNr-1), -1, -1):
            if self.verbose:
                print "    Octave n.", noct+1, "out of", self.octaveNr
            inc = ahop / (2.**noct)
            nshifts = int(2**noct)
            
            nframes = int(self.nframes[noct])
            
            ylen = (
                self.cqtkernel.fftHOP * (nframes-1)
                + self.cqtkernel.FFTLen
                + nshifts * inc # adding the different shifts
                )
            if ylen > y.size:
                y = np.concatenate([y, np.zeros(ylen-y.size)])
            
            for nshift in np.arange(nshifts):
                if self.verbose>1:
                    print "        shift n.", nshift+1, "out of", nshifts
                self.spCQT2CellCQT()
                
                Y = np.dot(K, self.cellCQT[noct])
                yoct = np.zeros(self.cqtkernel.FFTLen)
                for n in range(nframes):
                    frastart = int(n * self.cqtkernel.fftHOP + nshift * inc)
                    frastop = frastart + self.cqtkernel.FFTLen
                    # divide by nshifts because we know there are as many 
                    # redundant shifts to overlap and add.
                    yoct = (
                        np.fft.ifft(Y[:,n], n=self.cqtkernel.FFTLen)
                        / np.double(nshifts))
                    yoct = 2. * np.real(yoct)
                    y[frastart:frastop] += yoct
                
                # shift the spCQT for that octave by one to the left,
                # we end up with the windows which are at inc sample
                # (for this decimated octave version of y):
                self._spCQT[int(self.cqtkernel.bins*(self.octaveNr-noct-1)):
                         int(self.cqtkernel.bins*(self.octaveNr-noct)),
                         :-1] = (
                    self._spCQT[int(self.cqtkernel.bins*(self.octaveNr-noct-1)):
                             int(self.cqtkernel.bins*(self.octaveNr-noct)),
                             1:]
                    )
            
            # upsampling the signal by factor 2:
            if noct != 0:
                newy = np.zeros(y.size*2)
                newy[::2] = y
                y = spsig.filtfilt(self.B,
                                   self.A,
                                   newy)
                del newy
                y *= 2
                
        self._spCQT = np.copy(tmpSpCQT)
        del tmpSpCQT
        
        y = y[self.prefixZeros:]
        y = y[:self.datalen_init]
        return y
    
    def cellCQT2spCQT(self):
        """compute the full cqt from self.cellCQT"""
        # compute the full cqt from self.cellCQT
        # nb of frames for spCQT, as in cask.
        spCQTframes = self.nframes[0] * self.cqtkernel.winNr
        self._spCQT = np.zeros([self.cqtkernel.bins * 
                                self.octaveNr,
                                self.cellCQT[0].shape[1] *
                                self.cqtkernel.winNr],
                                # self.nframes[0] * self.cqtkernel.winNr],
                               dtype=np.complex)
        nfreqbins = self.cqtkernel.bins * self.octaveNr
        #for fn in range(nfreqbins):
        #
        atomNr = self.cqtkernel.winNr
        emptyHops = self.cqtkernel.first_center *1./self.cqtkernel.atomHOP
        ahop = self.cqtkernel.atomHOP
        
        for noct in range(int(self.octaveNr)):
            drop = emptyHops * (2**(self.octaveNr-noct-1)) - emptyHops
            #drop = 0 # DEBUG
            binVec = np.int32(
                self.cqtkernel.bins * (self.octaveNr-noct-1) +
                np.arange(self.cqtkernel.bins)
                )
            nshifts = int(2**noct)
            # self.cellCQT
            if atomNr>1:
                if self.verbose>1:
                    print "            filling in CQT matrix, "+\
                          "many windows per frame"
                for nb, b in enumerate(binVec):
                    for a in range(int(atomNr)):
                        self._spCQT[
                            b,
                            int(a*nshifts):\
                            int((self.cellCQT[noct].shape[1]) *
                                  atomNr * nshifts):\
                            int(atomNr * nshifts)] = (
                            self.cellCQT[noct][nb*atomNr+a]
                            )
                    self._spCQT[b,:(self._spCQT.shape[1]
                                    -int(drop*nshifts))] = (
                        self._spCQT[b, int(drop*nshifts):]
                        )
            else:
                if self.verbose>1:
                    print "            filling in CQT matrix, "+\
                          "one window per frame"
                ##print self._spCQT[binVec, :int((self.cellCQT[noct].shape[1]
                ##                      -drop) * nshifts):nshifts].shape#DEBUG
                ##print self.cellCQT[noct][:, int(drop):].shape#DEBUG
                self._spCQT[binVec, :int((self.cellCQT[noct].shape[1]
                                          -drop) * nshifts):nshifts] = (
                    # might raise a bound error here...
                    self.cellCQT[noct][:, int(drop):]
                    )
            # interpolating the values:
            if noct>0:
                for nb, b in enumerate(binVec):
                    yint = np.abs(self._spCQT[b][::nshifts]) 
                    xint = np.arange(start=0,
                                     stop=self._spCQT.shape[1],
                                     step=nshifts)
                    yint = np.abs(self._spCQT[b][::nshifts]) 
                    finterp = (
                        spinterp.interp1d(x=xint,
                                          y=yint,
                                          kind='linear',
                                          bounds_error=False,
                                          fill_value=0)
                        )
                    oldspcqt = np.copy(self._spCQT[b, np.int32(xint)])
                    self._spCQT[b] = (
                        finterp(np.arange(self._spCQT.shape[1])))
                    self._spCQT[b, xint] = oldspcqt
        # resizing spCQT to fit original size:
        self._spCQT = (np.ascontiguousarray(self._spCQT[:,:spCQTframes]))
    
    def spCQT2CellCQT(self):
        """ generates self.cellCQT from self.spCQT
        
        NB: after transformation of spCQT (by filtering, for instance),
        this method only keeps downsampled versions of each CQT representation
        for each octave. More elaborated computations may be necessary to
        take into account more precise time variations at low frequency
        octaves.
        """
        if not hasattr(self, 'spCQT'):
            raise AttributeError("There is no spCQT field in this object. "+\
                                 "Compute the transform or fill it in from "+\
                                 "another object before this computation!")
        self.cellCQT = {}
        # now, if not, according to check CQTexists, self._spCQT exists.
        emptyHops = self.cqtkernel.first_center *1./self.cqtkernel.atomHOP
        self.cellCQT = {}
        for noct in range(int(self.octaveNr)):
            dropped = emptyHops * (2.**(self.octaveNr-noct-1) - 1)
            X = self._spCQT[int(self.cqtkernel.bins*(self.octaveNr-noct-1)):
                            int(self.cqtkernel.bins*(self.octaveNr-noct)),
                            ::int(2**noct)]
            X = (
                np.hstack([np.zeros([self.cqtkernel.bins, dropped]),
                           X,])
                )
            X = (
                np.hstack([X,
                           np.zeros([self.cqtkernel.bins,
                                     np.ceil(X.shape[1]/
                                             self.cqtkernel.winNr)*
                                     self.cqtkernel.winNr -
                                     X.shape[1]])])
                )
            ##print X.shape #DEBUG
            if self.cqtkernel.winNr>1:
                # in order to keep the same size as the original,
                # we set the cell shapes as done in computeCQT instead:
                ##nframes = (
                ##    np.floor(
                ##    ((self.datalen_init + self.prefixZeros + self.suffixZeros
                ##      - self.cqtkernel.FFTLen) / self.cqtkernel.fftHOP) + 1))
                self.cellCQT[noct] = np.zeros([self.cqtkernel.bins *
                                               self.cqtkernel.winNr,
                                               #nframes],
                                               np.ceil(X.shape[1]/
                                                       self.cqtkernel.winNr)],
                                              dtype=np.complex)
                for nbin in range(int(self.cqtkernel.bins)):
                    ##print int(nbin*self.cqtkernel.winNr) #DEBUG
                    ##print int((nbin+1)*self.cqtkernel.winNr) #DEBUG
                    self.cellCQT[noct][int(nbin*self.cqtkernel.winNr):\
                                       int((nbin+1)*self.cqtkernel.winNr),
                                       :] = (
                        X[nbin].reshape(
                        self.cqtkernel.winNr,
                        X.shape[1]/self.cqtkernel.winNr, order='F')
                        )
            else:
                self.cellCQT[noct] = np.copy(X)
            self.cellCQT[noct] = (
                np.ascontiguousarray(self.cellCQT[noct][:,:self.nframes[noct]])
                )
            
    def invertTransform(self):
        """Invert the desired transform, here invert CQT
        from the cell CQT: like the original from [Schorkhuber2010]
        """
        return self.invertFromCellCQT()
    
    def invertFromCellCQT(self):
        """inverting the Cell CQT 
        """
        self._check_attr_inversion()
        # inverting Kernel:
        K = np.ascontiguousarray(self.cqtkernel.sparKernel)
        y = np.zeros(np.ceil(self.datalen_init /
                             (2.**(self.octaveNr-1))))
        for noct in range(int(self.octaveNr-1), -1, -1):
            Y = np.dot(K, self.cellCQT[noct])
            _, nframes = self.cellCQT[noct].shape
            ylen = (
                self.cqtkernel.fftHOP * (nframes-1)
                + self.cqtkernel.FFTLen
                )
            if ylen > y.size:
                y = np.concatenate([y, np.zeros(ylen-y.size)])
            yoct = np.zeros(self.cqtkernel.FFTLen)
            for n in range(nframes):
                frastart = n * self.cqtkernel.fftHOP
                frastop = frastart + self.cqtkernel.FFTLen
                yoct = np.fft.ifft(Y[:,n], n=self.cqtkernel.FFTLen)
                yoct = 2. * np.real(yoct)
                y[frastart:frastop] += yoct
                
            if noct != 0:
                newy = np.zeros(y.size*2)
                newy[::2] = y
                y = spsig.filtfilt(self.B,
                                   self.A,
                                   newy)
                del newy
                y *= 2
                
        y = y[self.prefixZeros:]
        y = y[:self.datalen_init]
        return y
    
    def _check_attr_inversion(self):
        """a function that raises an error if any of the needed
        attributes are not there, before trying to invert the transform
        """
        necessaryAttributes = ['datalen_init',
                               'prefixZeros',
                               'suffixZeros',
                               'octaveNr']
        for attr in necessaryAttributes:
            if not hasattr(self, attr):
                raise AttributeError(
                    "Missing attribute to compute the inverse "+\
                    "transform: %s." %attr)
        return True

class HybridCQTransfo(CQTransfo):
    """Hybrid Constant Q Transform"""
    def __init__(self, **kwargs):
        super(HybridCQTransfo, self).__init__(**kwargs)
        # but the cqtkernel is augmented as an hybrid kernel:
        self.cqtkernel = HybridCQTKernel(
            fmax=self.fmax,
            bins=self.bins,
            fs=self.fs,
            q=self.q,
            atomHopFactor=self.atomHopFactor,
            thresh=self.thresh, 
            winFunc=self.winFunc,
            perfRast=self.perfRast)
        self.freqbins = self.bins*self.octaveNr + self.cqtkernel.linBins
        
        if 'data' in kwargs.keys() and kwargs['data'] is not None:
            self.computeLinearPart(data=kwargs['data'])
    
    def cellCQT2spCQT(self):
        """the spCQT is computed from self.cellCQT
        """
        # this computes the cellCQT for the octaves of the CQT:
        #     NB: super... returns self._spCQT
        #         maybe more efficient would be to have a
        #         separate method that computes and the other
        #         one that returns it? j'ignore de le savoir...
        super(HybridCQTransfo,self).cellCQT2spCQT()
        # now compute for linear frequency part:
        #     alignment should  be the same as for octave=0
        self._spCQT = np.concatenate([
            self._spCQT,
            np.zeros([self.cqtkernel.linBins,
                      self._spCQT.shape[1]],
                     dtype=np.complex)
            ])
        
        atomNr = self.cqtkernel.winNr
        emptyHops = self.cqtkernel.first_center *1./self.cqtkernel.atomHOP
        ahop = self.cqtkernel.atomHOP
        drop = emptyHops * (2**(self.octaveNr-1)) - emptyHops
        
        binVec = np.int32(
                    self.cqtkernel.bins * self.octaveNr +
                    np.arange(self.cqtkernel.linBins)
                    )
        
        if atomNr>1:
            if self.verbose>1:
                print "            filling in CQT matrix, "+\
                      "many windows per frame"
            for nb, b in enumerate(binVec):
                for a in range(int(atomNr)):
                    self._spCQT[
                        b,
                        int(a):\
                        int(self.cellCQT['linear'].shape[1] *
                            atomNr):\
                        int(atomNr)] = (
                        self.cellCQT['linear'][nb*atomNr+a]
                        )
                self._spCQT[b,:(self._spCQT.shape[1]
                                -int(drop))] = (
                    self._spCQT[b, int(drop):]
                    )
        else:
            if self.verbose>1:
                print "            filling in CQT matrix, "+\
                      "one window per frame"
            self._spCQT[binVec, :int(self.cellCQT['linear'].shape[1]
                                     -drop)] = (
                # might raise a bound error here...
                self.cellCQT['linear'][:,int(drop):]
                )
            
    def computeTransform(self, data):
        """Computes the desired transform
        """
        super(HybridCQTransfo, self).computeTransform(data)
        self.computeLinearPart(data=data)
    
    def computeHybrid(self, data):
        """calculates a hybrid CQT/FT representation of a sound stored in data
        """
        self.computeCQT(data=data)
        # compute linear freqs part:
        self.computeLinearPart(data=data)
    
    def computeLinearPart(self, data):
        """Same as computeCQT, except it uses the linear frequency components
        in cqtkernel.linearSparKernel
        
        NB: since this should be equivalent to computing an FFT after windowing
        each frame, there may be a faster way of implementing this function.
        For now, keeping the same rules as the original CQT implementation,
        for consistency and also for avoiding problems with window synchrony
        """
        if self.verbose:
            print "Computing ``missing'' linear frequency part of the spectrum"
        
        cqtkernel = self.cqtkernel
        
        # %zeropadding
        x = (
            np.concatenate([np.zeros(self.prefixZeros),
                            data,
                            np.zeros(self.suffixZeros)])
            )
        OVRLP = cqtkernel.FFTLen - cqtkernel.fftHOP
        # %conjugate spectral kernel for cqt transformation
        K = np.ascontiguousarray(np.conjugate(cqtkernel.linearSparKernel.T))
        if not self.cqtkernel.perfRast:       
            if self.verbose:
                print "    Octave n.", 0, "out of", self.octaveNr
            nframes = self.nframes[0]
            self.cellCQT['linear'] = np.zeros([cqtkernel.linBins
                                               * cqtkernel.winNr,
                                               nframes],
                                              dtype=np.complex)
            # in theory we could avoid a loop, at the cost of
            # more memory consumption...
            XX = np.zeros([self.cqtkernel.FFTLen, nframes],
                          dtype=np.complex)
            for n in np.arange(nframes):
                if self.verbose>1:
                    print "        nframe", n, "out of", nframes
                framestart = n*cqtkernel.fftHOP
                framestop = framestart + cqtkernel.FFTLen
                XX[:,n] = np.fft.fft(x[framestart:framestop],
                               n=cqtkernel.FFTLen)
            # %calculating cqt coefficients for all FFT frames
            # for this octave
            self.cellCQT['linear'] = np.dot(K, XX, out=self.cellCQT['linear'])
        else:
            atomNr = self.cqtkernel.winNr
            emptyHops = self.cqtkernel.first_center *1./self.cqtkernel.atomHOP
            ahop = self.cqtkernel.atomHOP
            if self.verbose:
                print "    Octave n.", 0, "out of", self.octaveNr
            binVec = np.int32(
                    self.cqtkernel.bins * self.octaveNr +
                    np.arange(self.cqtkernel.linBins)
                    )
            # self._spCQT gets big...
            self._spCQT = np.vstack([
                self._spCQT,
                np.zeros([self.cqtkernel.linBins,
                          self._spCQT.shape[1]],
                         dtype=np.complex)])
            drop = emptyHops * (2**(self.octaveNr-1) - 1) # same as octave=0
            nframes = self.nframes[0]
            self.cellCQT['linear'] = np.zeros([self.cqtkernel.linBins * atomNr,
                                               nframes],
                                              dtype=np.complex)
            XX = np.zeros([self.cqtkernel.FFTLen, nframes],
                          dtype=np.complex)
            for n in np.arange(nframes):
                if self.verbose>2:
                    print "            nframe", n+1, "out of", nframes
                # %applying fft to each column (each FFT frame)
                framestart = n * self.cqtkernel.fftHOP
                framestop = framestart + self.cqtkernel.FFTLen
                ##X = np.fft.fft(x[framestart:framestop],
                ##               n=cqtkernel.FFTLen)
                XX[:,n]= np.fft.fft(x[framestart:framestop],
                                    n=cqtkernel.FFTLen)
                
            self.cellCQT['linear'] = np.dot(K, XX, out=self.cellCQT['linear'])
            if atomNr>1:
                if self.verbose>1:
                    print "            filling in CQT matrix, "+\
                          "many windows per frame"
                for nb, b in enumerate(binVec):
                    for a in range(int(atomNr)):
                        self._spCQT[
                            b,
                            int(a):\
                            int((nframes)*atomNr):\
                            int(atomNr)] = (
                            self.cellCQT['linear'][nb*atomNr+a]
                            )
            else:
                # TODO: not sure whether this piece of code works:
                #    needs testing!
                if self.verbose>1:
                    print "            filling in CQT matrix, "+\
                          "one window per frame"
                self._spCQT[binVec] = (
                    self.cellCQT['linear'] #CQTframe[:,int(drop*nshifts):]
                    )
            
            for nb, b in enumerate(binVec):
                self._spCQT[b,:(self._spCQT.shape[1]-int(drop))]=(
                    self._spCQT[b, int(drop):]
                    )
                
    def invertTransform(self):
        """invert the desired transform
        """
        return self.invertHybridCQT(self)
                
    def invertHybridCQT(self):
        """Invert the hybrid transform.
        
        Linearity allows to perform the cqt inverse first, and add the inverse
        of the linear freqs part thereafter (or the other way around).
        """
        y = self.invertFromCellCQT()
        y += self.invertLinearPart()
        return y
    
    def invertLinearPart(self):
        """This inverts the linear part of the hybrid transform
        
        NB: as for the computation of this part in transform, a windowed
        version
        of a plain FFT should do the same job, and faster.
        """
        K = np.ascontiguousarray(self.cqtkernel.linearSparKernel)
        y = np.zeros(np.ceil(
            self.prefixZeros +
            self.datalen_init +
            self.suffixZeros))
        Y = np.dot(K, self.cellCQT['linear'])
        _, nframes = Y.shape
        ylen = (
            self.cqtkernel.fftHOP * (nframes-1)
            + self.cqtkernel.FFTLen
            )
        if ylen > y.size:
            y = np.concatenate([y, np.zeros(ylen-y.size)])
        yoct = np.zeros(self.cqtkernel.FFTLen)
        for n in range(nframes):
            frastart = n * self.cqtkernel.fftHOP
            frastop = frastart + self.cqtkernel.FFTLen
            yoct = np.fft.ifft(Y[:,n])
            yoct = 2. * np.real(yoct)
            y[frastart:frastop] += yoct
        
        y = y[self.prefixZeros:]
        y = y[:self.datalen_init]
        return y

    def spCQT2CellCQT(self):
        # populating the CQT part of cellCQT
        super(HybridCQTransfo, self).spCQT2CellCQT()
        # now computing the linear frequency part
        emptyHops = self.cqtkernel.first_center * 1. / self.cqtkernel.atomHOP
        dropped = emptyHops * (2.**(self.octaveNr - 1) - 1)
        X = self._spCQT[int(self.cqtkernel.bins * self.octaveNr):
                        int(self.cqtkernel.bins * self.octaveNr
                            + self.cqtkernel.linBins)]
        X = (
            np.hstack([np.zeros([self.cqtkernel.linBins, dropped]),
                       X,])
            )
        X = (
            np.hstack([X,
                       np.zeros([self.cqtkernel.linBins,
                                 np.ceil(X.shape[1] / 
                                         self.cqtkernel.winNr)*
                                 self.cqtkernel.winNr -
                                 X.shape[1]])])
            )
        if self.cqtkernel.winNr>1:
            self.cellCQT['linear'] = np.zeros([self.cqtkernel.linBins *
                                               self.cqtkernel.winNr,
                                               np.ceil(X.shape[1]/
                                                       self.cqtkernel.winNr)],
                                              dtype=np.complex)
            for nbin in range(int(self.cqtkernel.linBins)):
                ##print int(nbin*self.cqtkernel.winNr) #DEBUG
                ##print int((nbin+1)*self.cqtkernel.winNr) #DEBUG
                self.cellCQT['linear'][int(nbin*self.cqtkernel.winNr):\
                                       int((nbin+1)*self.cqtkernel.winNr)] = (
                    X[nbin].reshape(
                        self.cqtkernel.winNr,
                        X.shape[1]/self.cqtkernel.winNr, order='F')
                    )
        else:
            self.cellCQT['linear'] = np.copy(X)
            
        # resizing to fit original size:
        self.cellCQT['linear'] = (
            np.ascontiguousarray(self.cellCQT['linear'][:,:self.nframes[0]])
            )
        
    def _compute_frequencies(self):
        freqs = super(HybridCQTransfo, self)._compute_frequencies()
        linfreqs = (
            np.arange(self.cqtkernel.linBins, dtype=np.float64) *
            self.cqtkernel.fs /
            self.cqtkernel.Nk_max# self.cqtkernel.FFTLen
            + self.fmax * 2 ** (1. / self.cqtkernel.bins)
            )
        return np.concatenate([freqs, linfreqs])

class MinQTransfo(CQTransfo):
    """Minimum Q Transform"""
    transformname = 'minqt'
    def __init__(self, fmax, bins, linFTLen, fs, fmin=70,
                 **kwargs):
        """MinQTransform : MinQT
        """
        # not computing the cqtkernel there, hence a "dummy" one:
        super(MinQTransfo, self).__init__(fmax=fmax, fs=fs,
                                          bins=bins,cqtkernel=0,
                                          fmin=fmin,
                                          **kwargs)
        # but the cqtkernel is augmented as the MinQT kernel:
        self.cqtkernel = MinQTKernel(
            linFTLen=linFTLen,
            fmax=fmax,
            bins=bins,
            fs=fs,
            q=self.q,
            atomHopFactor=self.atomHopFactor,
            thresh=self.thresh, 
            winFunc=self.winFunc,
            perfRast=self.perfRast)
        
        self.octaveNr = np.ceil(np.log2(self.cqtkernel.fmax * 1. / fmin))
        self.fmin = (self.cqtkernel.fmax / (2.**self.octaveNr)) * 2**(1./bins)
        
        self.freqbins = (
            self.octaveNr * self.cqtkernel.bins
            + self.cqtkernel.linBins
            )
        
        if 'data' in kwargs.keys() and kwargs['data'] is not None:
            self.computeLinearPart(data=kwargs['data'])
    
    def computeTransform(self, data):
        """Computes the desired transform
        """
        super(MinQTransfo, self).computeTransform(data)
        self.computeLinearPart(data=data)
        
    def computeLinearPart(self, data):
        """Compute the linear frequency part with an STFT, and
        taking only the desired frequencies.
        """
        if self.verbose:
            print "Computing ``missing'' linear frequency part of the spectrum"

        # to get the matrices aligned, needs to have the first frame
        # of STFT centered on self.cqtkernel.first_center (?)
        # %zeropadding
        x = (
            np.concatenate([np.zeros(self.prefixZeros),
                            data,
                            np.zeros(self.suffixZeros)])
            )
        # since qo.stft centers the first frame on the first
        # sample of the data, we need to remove half a window:
        self.offsetSTFT = (self.cqtkernel.first_center)
        X,F,N = stft(data=x[self.offsetSTFT:],
                     window=self.cqtkernel.linWindow,
                     hopsize=self.cqtkernel.atomHOP,
                     nfft=self.cqtkernel.linFTLen,
                     fs=self.cqtkernel.fs)
        # filling cellCQT, assuming same number of frames:
        self.cellCQT['linear'] = X[self.cqtkernel.Kmax:,
                                   :self.nframes[0] * self.cqtkernel.winNr]
        if self.perfRast:
            # self._spCQT gets big...
            self.linCellCQT2LinSpCQT()
            ##self._spCQT = np.vstack([
            ##    self._spCQT,
            ##    np.zeros([self.cqtkernel.linBins,
            ##              self._spCQT.shape[1]],
            ##             dtype=np.complex)])
            ##emptyHops = self.cqtkernel.first_center *1./self.cqtkernel.atomHOP
            ##drop = emptyHops * (2**(self.octaveNr-1) - 1) # same as octave=0
            ### filling _spCQT:
            ##self._spCQT[(self.cqtkernel.bins * self.octaveNr):,
            ##            :self._spCQT.shape[1]-int(drop)] = (
            ##    self.cellCQT['linear'][:,int(drop):]
            ##    )
    
    def invertTransform(self):
        """invert the desired transform
        """
        if not self.perfRast:
            y = self.invertFromCellCQT()
            y += self.invertLinearPart()
            return y
        else:
            return self.invertFromSpCQTRast()
    
    def invertFromSpCQTRast(self):
        """
        """
        y = super(MinQTransfo, self).invertFromSpCQTRast()
        y += self.invertLinearPart()
        return y
    
    def invertLinearPart(self):
        """This inverts the linear part of the hybrid transform
        
        NB: as for the computation of this part in transform, a windowed
        version
        of a plain FFT should do the same job, and faster.
        """
        Y = np.zeros([self.cqtkernel.linFTLen / 2 + 1,
                      self.cellCQT['linear'].shape[1]],
                     dtype=np.complex)
        Y[self.cqtkernel.Kmax:] = self.cellCQT['linear']
        y = istft(X=Y, window=self.cqtkernel.linWindow,
                  hopsize=self.cqtkernel.atomHOP,
                  nfft=self.cqtkernel.linFTLen)
        y = y[(self.prefixZeros-self.offsetSTFT):]
        y = y[:self.datalen_init]
        return y
    
    def _compute_frequencies(self):
        freqs = super(MinQTransfo, self)._compute_frequencies()
        linfreqs = (
            np.arange(
                self.cqtkernel.Kmax,
                self.cqtkernel.Kmax + self.cqtkernel.linBins,
                dtype=np.float64) *
            self.cqtkernel.fs /
            self.cqtkernel.linFTLen # self.cqtkernel.FFTLen
            # + self.fmax * 2 ** (1. / self.cqtkernel.bins)
            )
        return np.concatenate([freqs, linfreqs])
    
    def spCQT2CellCQT(self):
        """Reverting spCQT (matrix form of the transform) back into
        the (original?) cellCQT (one matrix per octave, one for the linear
        part of the transform). 
        """
        # populating the CQT part of cellCQT
        super(MinQTransfo, self).spCQT2CellCQT()
        
        # now computing the linear frequency part:
        emptyHops = self.cqtkernel.first_center * 1. / self.cqtkernel.atomHOP
        dropped = emptyHops * (2.**(self.octaveNr - 1) - 1)
        X = self._spCQT[int(self.cqtkernel.bins * self.octaveNr):
                        int(self.cqtkernel.bins * self.octaveNr
                            + self.cqtkernel.linBins)]
        X = (
            np.hstack([np.zeros([self.cqtkernel.linBins, dropped]),
                       X,])
            )
        self.cellCQT['linear'] = np.copy(X)
        
        # resizing to fit original size:
        self.cellCQT['linear'] = (
            np.ascontiguousarray(
                self.cellCQT['linear'][:,:(self.nframes[0]*
                                           self.cqtkernel.winNr)])
            )

    def cellCQT2spCQT(self):
        """converts the cellCQT into a spCQT (matrix form)
        """
        # computing the self._spCQT from the CQT cellCQT cells:
        super(MinQTransfo, self).cellCQT2spCQT()
        self.linCellCQT2LinSpCQT()
        
    def linCellCQT2LinSpCQT(self):
        """converts cellCQT['linear'] into the corresponding bins in
        self._spCQT
        """
        self._spCQT = np.vstack([
            self._spCQT,
            np.zeros([self.cqtkernel.linBins,
                      self._spCQT.shape[1]],
                     dtype=np.complex)])
        emptyHops = self.cqtkernel.first_center * 1. / self.cqtkernel.atomHOP
        drop = emptyHops * (2**(self.octaveNr-1) - 1) # same as octave=0
        # filling _spCQT:
        self._spCQT[(self.cqtkernel.bins * self.octaveNr):,
                    :self._spCQT.shape[1]-int(drop)] = (
            self.cellCQT['linear'][:,int(drop):]
            )
        
