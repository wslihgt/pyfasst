from ..tools.utils import *

def stft(data, window=sinebell(2048),
         hopsize=256.0, nfft=2048.0, fs=44100.0):
    """
    X, F, N = stft(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
        F                     :
            values of frequencies at each Fourier bins
        N                     :
            central time at the middle of each analysis
            window
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    lengthData = data.size
    
    # should be the number of frames by YAAFE:
    numberFrames = np.ceil(lengthData / np.double(hopsize)) + 2
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = (numberFrames-1) * hopsize + lengthWindow
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data
    data = np.concatenate((np.zeros(lengthWindow/2.0), data))
    
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros(newLengthData - data.size)))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2 + 1
    
    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    
    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = n*hopsize
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, np.int32(nfft));
        
    # frequency and time stamps:
    F = np.arange(numberFrequencies)/np.double(nfft)*fs
    N = np.arange(numberFrames)*hopsize/np.double(fs)
    
    return STFT, F, N

def istft(X, window=sinebell(2048),
          analysisWindow=None,
          hopsize=256.0, nfft=2048.0):
    """
    data = istft(X,window=sinebell(2048),hopsize=1024.0,nfft=2048.0,fs=44100)

    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.

    Inputs:
        X                     :
            STFT of the signal, to be \"inverted\"
        window=sinebell(2048) :
            synthesis window
            (should be the \"complementary\" window
            for the analysis window)
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)

    Outputs:
        data                  :
            time series corresponding to the given STFT
            the first half-window is removed, complying
            with the STFT computation given in the
            function stft
    
    """
    if analysisWindow is None:
        analysisWindow = window
    
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = hopsize*(numberFrames-1) + lengthWindow
    
    normalisationSeq = np.zeros(lengthData)
    
    data = np.zeros(lengthData)
    
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft))
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)
    
    data = data[(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.
    # ...added in the stft computation
    
    # normalising the liutkus way:
    data = data / normalisationSeq
    
    return data

def filter_stft(data, W, analysisWindow=None,
                synthWindow=sinebell(2048),
                hopsize=256.0, nfft=2048.0, fs=44100.0):
    """Sequentially compute Fourier transfo, filter and overlap-add
    
    W is the M x M x F x N filter for the data, which should be T x M
    data T x M (number of samples, number of channels)
    """
    ns, nc = data.shape
    if nc != W.shape[0]:
        print "data.shape", data.shape, "W.shape", W.shape
        raise AttributeError("W does not have the right number of channels")
        
    # window defines the size of the analysis windows
    if analysisWindow is None or len(analysisWindow) != len(synthWindow):
        analysisWindow = synthWindow
    
    lengthWindow = synthWindow.size
    
    lengthData = ns
    
    # should be the number of frames by YAAFE:
    numberFrames = np.ceil(lengthData / np.double(hopsize))
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = (numberFrames-1) * hopsize + lengthWindow
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data
    data = np.concatenate((np.zeros([lengthWindow/2.0, nc]), data))
    
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data,
                           np.zeros([newLengthData - data.shape[0], nc])
                           )
                          )
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2 + 1
    if numberFrequencies != W.shape[2]:
        raise AttributeError("W not the right size")
    
    # new data to be written:
    normalisationSeq = np.zeros(newLengthData)
    ndata = np.zeros([newLengthData, nc])
    
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        
        # Compute Fourier transforms
        ft = np.zeros([nc, numberFrequencies],dtype=complex)
        for c in range(nc):
            frameToProcess = analysisWindow * data[beginFrame:endFrame, c]
            ft[c] = np.fft.rfft(frameToProcess, np.int32(nfft))
        
        # filter with W
        filteredFt = np.zeros_like(ft)
        if W.ndim == 3:
            for c1 in range(nc):
                for c2 in range(nc):
                    filteredFt[c1] += W[c1,c2] * ft[c2]            
        elif W.ndim == 4:
            for c1 in range(nc):
                for c2 in range(nc):
                    filteredFt[c1] += W[c1,c2,:,n] * ft[c2]
        else:
            raise NotImplementedError("For W.ndim== 3 or 4"+str(W.ndim))
        
        # overlap add
        #print beginFrame,endFrame,synthWindow.shape,analysisWindow.shape#DEBUG
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            synthWindow * analysisWindow
            )
        for c in range(nc):
            frameTMP = np.fft.irfft(filteredFt[c],
                                    np.int32(nfft))
            frameTMP = frameTMP[:lengthWindow]
            ndata[beginFrame:endFrame,c] = (
                ndata[beginFrame:endFrame,c] + 
                synthWindow * frameTMP
                )
        
        del ft, filteredFt
    ndata = ndata[(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.
    # ...added in the stft computation
    
    # normalising the liutkus way:
    ndata = ndata / np.vstack(normalisationSeq)
        
    return ndata

def filter_conv_stft(data, W, analysisWindow=None,
                      synthWindow=sinebell(2048),
                      hopsize=256.0, nfft=2048.0, fs=44100.0,
                      verbose=0):
    """Sequentially compute Fourier transfo, filter and overlap-add

    INPUTS
    
     W
      M x F x N (or M x F) filter for the data, which should be single channel
      
     data
      T (number of samples, number of channels)

     ...

     
    """
    if (data.ndim==2 and data.shape[1]!=1) or (data.ndim>2):
        raise AttributeError("provided data should be single channel")
    
    if (data.ndim==2 and data.shape[1]==1):
        data = data.flatten()
        
    ns = data.size
    nchanout = W.shape[0]
    
    if verbose:
        print "data.shape", data.shape, "W.shape", W.shape
    
    # window defines the size of the analysis windows
    if analysisWindow is None or len(analysisWindow) != len(synthWindow):
        analysisWindow = synthWindow
    
    lengthWindow = synthWindow.size
    
    lengthData = ns
    
    # should be the number of frames by YAAFE:
    numberFrames = np.ceil(lengthData / np.double(hopsize))
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = (numberFrames-1) * hopsize + lengthWindow
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data
    data = np.concatenate((np.zeros(lengthWindow/2.0), data))
    
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data,
                           np.zeros(newLengthData - data.shape[0])
                           )
                          )
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2 + 1
    if numberFrequencies != W.shape[1]:
        raise AttributeError("W not the right size")
    
    # new data to be written:
    normalisationSeq = np.zeros(newLengthData)
    ndata = np.zeros([newLengthData, nchanout])
    
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        
        # Compute Fourier transforms
        frameToProcess = analysisWindow * data[beginFrame:endFrame]
        ft = np.fft.rfft(frameToProcess, np.int32(nfft))
        
        # filter with W
        if W.ndim==2:
            filteredFt = W * ft
        elif W.ndim==3:
            filteredFt = W[:,:,n] * ft
            if verbose>1: print "W[:,:,n]", W[:,:,n]
        else:
            raise ValueError(
                "The provided filter does not have the right shape.")
        
        # overlap add
        #print beginFrame, endFrame, synthWindow.shape, analysisWindow.shape#DEBUG
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            synthWindow * analysisWindow
            )
        for c in range(nchanout):
            frameTMP = np.fft.irfft(filteredFt[c],
                                    np.int32(nfft))
            frameTMP = frameTMP[:lengthWindow]
            ndata[beginFrame:endFrame,c] = (
                ndata[beginFrame:endFrame,c] + 
                synthWindow * frameTMP
                )
        
        del ft, filteredFt
    
    ndata = ndata[(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.
    # ...added in the stft computation
    # normalising the liutkus way:
    ndata = ndata / np.vstack(normalisationSeq)
    return ndata

############
# wrapper transformation classes:
###########
class STFT():
    """Object that implements the computation of Short-Term Fourier Transforms
    (STFT) and its inverse.
    
    **Inputs:**

    :param integer linFTLen:
        size of the Fourier transform
    :param double atomHopFactor:
        ratio of delay from frame to frame. 0.25 corresponds to a 25% "hop"
        ratio, or equivalently to 75% of overlap between succesive frames.
    :param function winFunc:
        analysis window function.
    :param integer fs:
        sampling rate of the processed signals
    :param synthWinFunc:

    :param kwargs:
    
    """
    transformname = 'stft'
    def __init__(self, linFTLen=2048, atomHopFactor=0.25,
                 winFunc=np.hanning, fs=44100, synthWinFunc=None,
                 **kwargs):
        fthop = int(linFTLen * atomHopFactor)
        self.ftlen = linFTLen
        self.freqbins = self.ftlen / 2 + 1
        self.atomHopFactor = atomHopFactor
        self.fthop = fthop
        if winFunc is None:
            winFunc = np.hanning
        self.winFunc = winFunc
        self.window = self.winFunc(self.ftlen)
        self.synthWinFunc = (synthWinFunc if synthWinFunc is not None
                             else self.winFunc)
        self.synthWindow = self.synthWinFunc(self.ftlen)
        self.fs = fs
    
    def computeTransform(self, data):
        self.transfo, self.freq_stamps, self.time_stamps = stft(
            data=data,
            window=self.window,
            hopsize=self.fthop,
            fs=self.fs, nfft=self.ftlen
            )
        self.datalen_init = data.size
        self.time_stamps *= self.fs # for some reason, time_stamps is in samples
    
    def invertTransform(self):
        return istft(
            X=self.transfo,
            window=self.synthWindow,
            analysisWindow=self.window,
            hopsize=self.fthop,
            nfft=self.ftlen
            )[:self.datalen_init]
