"""SeparateLeadStereo, with Time-Frequency choice

Provides a class (``SeparateLeadProcess``) within which several
processings can be run on an audio file, in order to extract the
lead instrument/main voice from a (stereophonic) audio mixture.

copyright (C) 2011 - 2013 Jean-Louis Durrieu
"""

# Importing necessary packages:
import numpy as np

try:
    import scipy.io.wavfile as wav
except ImportError:
    import scipy
    spver = scipy.__version__
    raise ImportError('Version of scipy is %s, ' %(spver) + \
                      'to read WAV files, one needs >= 0.7.0')
from .SIMM import SIMM as SIMM
import os
import warnings
# importing the cython version of tracking:
#from tracking import viterbiTrackingArray
from .tracking._tracking import viterbiTracking as viterbiTrackingArray
# the following import gets useful functions for this class:
from . import separateLeadFunctions as slf
import scipy.optimize
from ..tftransforms import tft # time-freq transforms

eps = 10 ** -9

knownTransfos = ['stft', 'hybridcqt', 'minqt',
                 'cqt', 'mqt'] # TODO: 'cqt', 'erb'?

class SeparateLeadProcess():
    """SeparateLeadProcess
    
    class which implements the source separation algorithm, separating the
    'lead' voice from the 'accompaniment'. It can deal automatically with the
    task (the 'lead' voice becomes the most energetic one), or can be manually
    told what the 'lead' is (through the melody line).
    
    **Attributes**
     dataType : dtype
        this is the input data type (usually the same as the audio encoding)
    
     displayEvolution : boolean
        display the evolution of the arrays (notably HF0)
    
     F, N : integer, integer
        F the number of frequency bins in the time-frequency representation
          (this is half the Fourier bins, + 1)
          
        N the number of analysis input frames
    
     files :
        dictionary containing the filenames of the output files for the
        separated signals, with the following keys (after initialization)
        
            'inputAudioFilename' : input filename
            
            'mus_output_file' : output filename for the estimated
            'accompaniment', appending '_acc.wav' to the radical.
                
            'outputDirSuffix' : the subfolder name to be appended to the path
            of the directory of the input file, the output files will be
            written in that subfolder
                
            'outputDir' : the full path of the output files directory
            
            'pathBaseName' : base name for the output files
            (full path + radical for all output files)
            
            'pitch_output_file' : output filename for the estimated melody line
            appending '_pitches.txt' to the radical.
                
            'voc_output_file' : output filename for the estimated 'lead
            instrument', appending '_voc.wav' to the radical.
        
        Additionally, the estimated 'accompaniment' and 'lead' with unvoiced
        parts estimation are written to the corresponding filename without
        these unvoiced parts, to which '_VUIMM.wav' is appended.
        
     imageCanvas : instance from MplCanvas or MplCanvas3Axes
        canvas used to draw the image of HF0
        
     scaleData : double
        maximum value of the input data array.
        With scipy.io.wavfile, the data array type is integer, and does not
        fit well with the algorithm, so we need this scaleData parameter to
        navigate back and forth between the double and integer representation.
        
     scopeAllowedHF0 : double
        scope of allowed F0s around the estimated/given melody line 
    
     stftParams : dictionary with the parameters for the time-frequency
     representation (Short-Time Fourier Transform - STFT), with the keys:
            
            'hopsize' : the step, in number of samples, between analysis
            frames for the STFT
            
            'NFT' : the number of Fourier bins on which the Fourier transforms
            are computed.
            
            'windowSizeInSamples' : analysis frame length, in samples
    
     SIMMParams : dictionary with the parameters of the SIMM model
     (Smoothed Instantaneous Mixture Model [DRDF2010]_), with following keys:
            
            'alphaL', 'alphaR' : double
                stereo model, panoramic parameters for the lead part
                
            'betaL', 'betaR' : (R,) ndarray
                stereo model, panoramic parameters for each of the component of
                the accompaniment part.
                
            
            'chirpPerF0' : integer
                number of F0s between two 'stable' F0s, modelled
                as chirps.
                
            'F0Table' : (NF0,) ndarray
                frequency in Hz for each of the F0s appearing in WF0
            
            'HF0' : (NF0*chirpPerF0, N) ndarray, *estimated*
                amplitude array corresponding to the different F0s (this is
                what you want if you want the visualisation representation of
                the pitch saliances).
                
            'HF00' : (NF0*chirpPerF0, N) ndarray, *estimated*
                amplitude array HF0, after being zeroed everywhere outside
                the given scope from the estimated melody
                
            'HGAMMA' : (P, K) ndarray, *estimated*
                amplitude array corresponding to the different smooth shapes,
                decomposition of the filters on the smooth shapes in WGAMMA
                
            'HM' : (R, N) ndarray, *estimated*
                amplitude array corresponding to the decomposition of the
                accompaniment on the spectral shapes in WM
                
            'HPHI' : (K, N) ndarray, *estimated*
                amplitude array corresponding to the decomposition of the
                filter part on the filter spectral shapes in WPHI, defined
                as np.dot(WGAMMA, HGAMMA)
            
            'K' : integer
                number of filters for the filter part decomposition

            'maxF0' : double
                the highest F0 candidate

            'minF0' : double
                the lowest F0 candidate

            'NF0' : integer
                number of F0s in total

            'niter' : integer
                number of iterations for the estimation algorithm

            'P' : integer
                number of smooth spectral shapes for the filter part (in WGAMMA)

            'R' : integer
                number of spectral shapes for the accompaniment part (in WM)

            'stepNotes' : integer
                number of F0s between two semitones
            
            'WF0' : (F, NF0*chirpPerF0) ndarray, *fixed*
                'dictionary' of harmonic spectral shapes for the F0 candidates
                generated thanks to the KLGLOTT88 model [DRDF2010]

            'WGAMMA' : (F, P) ndarray, *fixed*
                'dictionary' of smooth spectral shapes for the filter part

            'WM' : (F, R) ndarray, *estimated*
                array of spectral shapes that are directly *estimated* on the
                signal
                
     verbose : boolean
        if True, the program writes some information about what is happening
    
     wavCanvas : instance from MplCanvas or MplCanvas3Axes
        the canvas that is going to be used to draw the input audio waveform
        
    
     XL, XR : (F, N) ndarray
        resp. left and right channel STFT arrays
    
    **Methods**
    
     Constructor : reads the input audio file, computes the STFT,
        generates the different dictionaries (for the source part,
        harmonic patterns WF0, and for the filter part, smooth
        patterns WGAMMA).
        
     automaticMelodyAndSeparation :
        launches sequence of methods to estimate the parameters, estimate the
        melody, then re-estimate the parameters and at last separate the
        lead from the rest, considering the lead is the most energetic source
        of the mixture (with some continuity regularity)
        
     estimSIMMParams :
        estimates the parameters of the SIMM, i.e. HF0, HPHI, HGAMMA, HM and WM
        
     estimStereoSIMMParams :
        estimates the parameters of the stereo version of the SIMM,
        i.e. same parameters as estimSIMMParams, with the alphas and betas 

     estimStereoSUIMMParams :
        same as above, but first adds 'noise' components to the source part

     initiateHF0WithIndexBestPath :
        computes the initial HF0, before the estimation, given the melody line
        (estimated or not)

     runViterbi :
        estimates the melody line from HF0, the energies of each F0 candidates

     setOutputFileNames :
        triggered when the text fields are changed, changing the output
        filenames

     writeSeparatedSignals :
        computing and writing the adaptive Wiener filtered separated files

     :py:func:`writeSeparatedSignalsWithUnvoice` :
        computing and writing the adaptive Wiener filtered separated files,
        unvoiced parts.
    
    **References**
    
    This is a class that encapsulates our work on source separation,
    published as:
    
    .. [DDR2011] J.-L. Durrieu, B. David and G. Richard,
       A Musically Motivated Mid-Level Representation
       For Pitch Estimation And Musical Audio Source Separation,
       IEEE Journal of Selected Topics on Signal Processing,
       October 2011, Vol. 5 (6), pp. 1180 - 1191.
        
    and
        
    .. [DRDF2010] J.-L. Durrieu, G. Richard, B. David and C. F\'evotte,
       Source/Filter Model for Main Melody Extraction
       From Polyphonic Audio Signals,
       IEEE Transactions on Audio, Speech and Language Processing,
       special issue on Signal Models and Representations of Musical
       and Environmental Sounds, March 2010, vol. 18 (3), pp. 564 -- 575.
       
    As of 3/1/2012, available at http://www.durrieu.ch/research
    
    """
    # files : dict containing filename to I/O
    # stftParams : dict containing the parameters for the STFT
    files = {}
    stftParams = {}
    SIMMParams = {}
    
    def __init__(self, inputAudioFilename,
                 windowSize=0.0464, hopsize=None, NFT=None, nbIter=10,
                 numCompAccomp=40,
                 minF0=39, maxF0=2000, stepNotes=16,
                 chirpPerF0=1,
                 K_numFilters=4,
                 P_numAtomFilters=30,
                 imageCanvas=None, wavCanvas=None,
                 progressBar=None,
                 verbose=True,
                 outputDirSuffix='/',
                 minF0search=None,
                 maxF0search=None,
                 tfrepresentation='stft',
                 cqtfmax=4000,
                 cqtfmin=50,
                 cqtbins=48,
                 cqtWinFunc=slf.minqt.sqrt_blackmanharris,
                 cqtAtomHopFactor=0.25,
                 initHF00='random',
                 freeMemory=True):
        """During init, process is initiated, STFTs are computed,
        and the parameters are stored.
        
        **Parameters**
        
         inputAudioFilename : string
            filename of the input audio file
         windowSize : double, optional
            analysis frame ('windows') size, in s. By default, 0.0464s
         nbIter : integer, optional
            number of iterations for the estimation algorithm. By default, 10
         numCompAccomp : integer, optional
            number of components for the accompaniment, default = 40
         minF0 : double/integer, optional
            lowest F0 candidate (in Hz), default=60Hz
         maxF0 : double/integer, optional
            highest F0 candidate (in Hz), default=2000Hz
         stepNotes : integer, optional
            number of F0 candidates in one semitone, default=16 F0s/semitone
         K_numFilters : integer, optional
            number of filter spectral shapes, default=4
         P_numAtomFilters : integer, optional
            number of atomic filter smooth spectral shapes, default=30
         imageCanvas : MplCanvas/MplCanvas3Axes, optional
            an instance of the MplCanvas/MplCanvas3Axes, giving access to the
            axes where to draw the HF0 image. By default=None
         wavCanvas : MplCanvas/MplCanvas3Axes, optional
            an instance of the MplCanvas/MplCanvas3Axes, giving access to the
            axes to draw the waveform of the input signal.
         progressBar : boolean, optional ???
            ???
         verbose : boolean, optional
            Whether to write out or not information about the evolution of the
            algorithm. By default=False.
         outputDirSuffix : string, optional
            the subfolder name (to be appended to the full path to the audio
            signal), where the output files are going to be written. By default
            ='/'
        
        """
        # discarding upper case letters from the input stri
        tfrepresentation = tfrepresentation.lower()
        
        if tfrepresentation not in knownTransfos:
            raise AttributeError("The desired Time-Freq representation "+
                                 tfrepresentation+" is not a recognized one.\n"+
                                 "Please choose from "+str(knownTransfos))
        
        self.tfrepresentation = tfrepresentation
        # representation specific parameters:
        if self.tfrepresentation in knownTransfos:
            self.stftParams['cqtfmin'] = cqtfmin
            self.stftParams['cqtfmax'] = cqtfmax
            self.stftParams['cqtbins'] = cqtbins
            self.stftParams['cqtWinFunc'] = cqtWinFunc
            self.stftParams['cqtAtomHopFactor'] = cqtAtomHopFactor
        
        self.files['inputAudioFilename'] = str(inputAudioFilename)
        self.imageCanvas = imageCanvas
        self.wavCanvas = wavCanvas
        self.displayEvolution = True
        self.verbose=verbose
        if self.imageCanvas is None:
            self.displayEvolution = False
        
        if inputAudioFilename[-4:] != ".wav":
            raise ValueError("File not WAV file? Only WAV format support, "+\
                             "for now...")
        
        self.files['outputDirSuffix']  = outputDirSuffix
        self.files['outputDir'] = str('/').join(\
            self.files['inputAudioFilename'].split('/')[:-1])+\
            '/'+self.files['outputDirSuffix'] +'/'
        if os.path.isdir(self.files['outputDir']):
            print "Output directory already existing - "+\
                  "NB: overwriting files in:"
            print self.files['outputDir']
        else:
            print "Creating output directory"
            print self.files['outputDir']
            os.mkdir(self.files['outputDir'])
        
        self.files['pathBaseName'] = self.files['outputDir'] + \
                                     self.files['inputAudioFilename'\
                                                ].split('/')[-1][:-4]
        self.files['mus_output_file'] = str(self.files['pathBaseName']+\
                                            '_acc.wav')
        self.files['voc_output_file'] = str(self.files['pathBaseName']+\
                                            '_lead.wav')
        self.files['pitch_output_file'] = str(self.files['pathBaseName']+\
                                              '_pitches.txt')
        
        print "Writing the different following output files:"
        print "    separated lead          in", \
              self.files['voc_output_file'] 
        print "    separated accompaniment in", \
              self.files['mus_output_file'] 
        print "    separated lead + unvoc  in", \
              self.files['voc_output_file'][:-4] + '_VUIMM.wav'
        print "    separated acc  - unvoc  in", \
              self.files['mus_output_file'][:-4] + '_VUIMM.wav'
        print "    estimated pitches       in", \
              self.files['pitch_output_file'] 
        
        # read the WAV file and store the STFT
        self.fs, data = wav.read(self.files['inputAudioFilename'])
        # for some bad format wav files, data is a str?
        # cf. files from beat/tempo evaluation campaign of MIREX
        ## print self.fs, data
        self.scaleData = 1.2 * np.abs(data).max() # to rescale the data.
        self.dataType = data.dtype
        data = np.double(data) / self.scaleData # makes data vary from -1 to 1
        if data.shape[0] == data.size: # data is multi-channel
            print "The audio file is not stereo. Making stereo out of mono."
            print "(You could also try the older separateLead.py...)"
            data = np.vstack([data, data]).T
            self.numberChannels = 1
        if data.shape[1] != 2:
            print "The data is multichannel, but not stereo... \n"
            print "Unfortunately this program does not scale well. Data is \n"
            print "reduced to its 2 first channels.\n"
            data = data[:,0:2]
            self.numberChannels = data.shape[1]
        
        # parameters for the STFT:
        self.stftParams['windowSizeInSamples'] = \
                 slf.nextpow2(np.round(windowSize * self.fs))
        if hopsize is None:
            self.stftParams['hopsize'] = (
                self.stftParams['windowSizeInSamples'] / 8.)
        else:
            self.stftParams['hopsize'] = np.double(hopsize)
        if NFT is None:
            self.stftParams['NFT'] = self.stftParams['windowSizeInSamples']
        else:
            self.stftParams['NFT'] = NFT

        # offsets are the number of samples added to the beginning of data
        # during the TF representation computation:
        # TODO: make this less of a hack?
        self.stftParams['offsets'] = {
            'stft': self.stftParams['windowSizeInSamples'] / 2,
            'minqt': 0,
            'mqt': 0,
            'hybridcqt': 0,
            'cqt': 0,}
        
        self.SIMMParams['niter'] = nbIter
        self.SIMMParams['R'] = numCompAccomp
        
        ##self.XR, F, N = slf.stft(data[:,0], fs=self.fs,
        ##                hopsize=self.stftParams['hopsize'] ,
        ##                window=slf.sinebell(\
        ##                       self.stftParams['windowSizeInSamples']),
        ##                nfft=self.stftParams['NFT'] )
        ##self.XL, F, N = slf.stft(data[:,1], fs=self.fs,
        ##                hopsize=self.stftParams['hopsize'] ,
        ##                window=slf.sinebell(\
        ##                       self.stftParams['windowSizeInSamples']),
        ##                nfft=self.stftParams['NFT'] )
        
        # non need to store this.
        ## self.SXR = np.abs(self.XR) ** 2
        ## self.SXL = np.abs(self.XL) ** 2
        
        # drawing the waveform to wavCanvas:
        if not(self.wavCanvas is None):
            if self.wavCanvas==self.imageCanvas:
                self.wavCanvas.ax2.clear()
                self.wavCanvas.ax2.plot(np.arange(data.shape[0]) / \
                                        np.double(self.stftParams['hopsize']),\
                                        data)
                #self.wavCanvas.ax2.plot(np.arange(data.shape[0]) / \
                #                       np.double(self.fs), \
                #                       data)
                self.wavCanvas.ax2.axis('tight')
                self.wavCanvas.draw()
            else:
                self.wavCanvas.ax.clear()
                self.wavCanvas.ax.plot(np.arange(data.shape[0]) / \
                                       np.double(self.fs), \
                                       data)
                self.wavCanvas.ax.axis('tight')
                self.wavCanvas.draw()
        
        del data
        
        # TODO: also process these as options:
        self.SIMMParams['minF0'] = minF0
        self.SIMMParams['maxF0'] = maxF0
        
        self.F = self.stftParams['NFT']/2 + 1
        # self.F, self.N = self.XR.shape
        # this is the number of F0s within one semitone
        self.SIMMParams['stepNotes'] = stepNotes
        # number of spectral shapes for the filter part
        self.SIMMParams['K'] = K_numFilters
        # number of elements in dictionary of smooth filters
        self.SIMMParams['P'] = P_numAtomFilters
        # number of chirped spectral shapes between each F0
        # this feature should be further studied before
        # we find a good way of doing that.
        self.SIMMParams['chirpPerF0'] = chirpPerF0
        self.scopeAllowedHF0 = 4.0 / 1.0
        
        # Create the harmonic combs, for each F0 between minF0 and maxF0:
        self.SIMMParams['initHF00'] = initHF00
        self.computeWF0()
        
        # for debug:
        if False: #DEBUG
            self.imageCanvas.ax.imshow(np.log(np.abs(self.XR)),
                                       aspect='auto',origin='lower')
            self.imageCanvas.draw()
            raise KeyboardInterrupt("Check these matrices !")
        
        if False: #DEBUG
            from IPython.Shell import IPShellEmbed
            
            ipshell = IPShellEmbed()
            
            ipshell()
            plt.figure()
            plt.imshow(np.log(self.SIMMParams['WF0']),
                                       aspect='auto',
                                       origin='lower',)
            plt.figure()
            plt.imshow(np.log(np.abs(self.XR)),aspect='auto',origin='lower')
        # Create the dictionary of smooth filters, for the filter part of
        # the lead isntrument:
        self.SIMMParams['WGAMMA'] = \
             slf.generateHannBasis(numberFrequencyBins=self.F,
                                   sizeOfFourier=self.stftParams['NFT'],
                                   Fs=self.fs,
                                   frequencyScale='linear', 
                                   numberOfBasis=self.SIMMParams['P'],
                                   overlap=.75)
        
        self.trackingParams = {}
        self.trackingParams['minF0search'] = self.SIMMParams['minF0']
        self.trackingParams['maxF0search'] = self.SIMMParams['maxF0']
        if minF0search is not None:
            self.trackingParams['minF0search'] = minF0search
        if maxF0search is not None:
            self.trackingParams['maxF0search'] = maxF0search
        
        print "Some parameter settings:"
        print "    Size of analysis windows: ", \
              self.stftParams['windowSizeInSamples'] 
        print "    Hopsize: ", self.stftParams['hopsize'] 
        print "    Size of Fourier transforms: ", self.stftParams['NFT'] 
        print "    Number of iterations to be done: ",self.SIMMParams['niter']  
        print "    Number of elements in WM: ", self.SIMMParams['R']
        
        self.freeMemory = freeMemory
    
    def setOutputFileNames(self, outputDirSuffix):
        """
        If already loaded a wav file, at this point, we can redefine
        where we want the output files to be written.
        
        Could be used, for instance, between the first estimation or the
        Viterbi smooth estimation of the melody, and the re-estimation
        of the parameters.
        
        """
        print "Redefining the Output Filenames !"
        
        self.files['outputDirSuffix']  = outputDirSuffix
        self.files['outputDir'] = str('/').join(\
            self.files['inputAudioFilename'].split('/')[:-1])+\
            '/'+self.files['outputDirSuffix'] +'/'
        if os.path.isdir(self.files['outputDir']):
            print "Output directory already existing - "+\
                  "NB: overwriting files in:"
            print self.files['outputDir']
        else:
            print "Creating output directory"
            print self.files['outputDir']
            os.mkdir(self.files['outputDir'])
        
        self.files['pathBaseName'] = self.files['outputDir'] + \
                                     self.files[\
                                      'inputAudioFilename'].split('/')[-1][:-4]
        self.files['mus_output_file'] = str(self.files['pathBaseName']+\
                                            '_acc.wav')
        self.files['voc_output_file'] = str(self.files['pathBaseName']+\
                                            '_lead.wav')
        self.files['pitch_output_file'] = str(self.files['pathBaseName']+\
                                              '_pitches.txt')
        
        print "Writing the different following output files:"
        print "    separated lead          in", \
              self.files['voc_output_file'] 
        print "    separated accompaniment in", \
              self.files['mus_output_file'] 
        print "    separated lead + unvoc  in", \
              self.files['voc_output_file'][:-4] + '_VUIMM.wav'
        print "    separated acc  - unvoc  in", \
              self.files['mus_output_file'][:-4] + '_VUIMM.wav'
        print "    estimated pitches       in", \
              self.files['pitch_output_file']
    
    def computeWF0(self):
        """Computes the frequency basis for the source part of SIMM,
        if tfrepresentation is a CQT, it also computes the cqt/hybridcqt
        transform object. 
        
        """
        if self.tfrepresentation == 'stftold':
            self.SIMMParams['F0Table'], WF0 = (
                slf.generate_WF0_chirped(
                    minF0=self.SIMMParams['minF0'],
                    maxF0=self.SIMMParams['maxF0'],
                    Fs=self.fs,
                    Nfft=self.stftParams['NFT'],
                    stepNotes=self.SIMMParams['stepNotes'],
                    lengthWindow=
                    self.stftParams['windowSizeInSamples'],
                    Ot=0.5, # 20130130 used to be 0.25
                    perF0=self.SIMMParams['chirpPerF0'],
                    depthChirpInSemiTone=.15,
                    loadWF0=True,
                    analysisWindow='sinebell'))
            self.SIMMParams['WF0'] = WF0[:self.F, :] # ensure same size as SX
            # number of harmonic combs
            self.SIMMParams['NF0'] = self.SIMMParams['F0Table'].size 
            # Normalization:
            # by max or by sum?
            self.SIMMParams['WF0'] = (
                self.SIMMParams['WF0'] /
                np.sum(self.SIMMParams['WF0'], axis=0))
        elif self.tfrepresentation in ['hybridcqt', 'minqt'] and False:
            if self.verbose:
                print "    Compute WF0, with MinQT transform"
                print "        - potentially (very) long -"
                if self.verbose>1:
                    print self.stftParams
            cqtwindowlength = np.ceil(
                self.fs /
                (self.stftParams['cqtfmin'] *
                 (2.**(1./self.stftParams['cqtbins']) - 1))
                )
            self.SIMMParams['F0Table'], WF0, self.mqt = (
                slf.generate_WF0_MinQT_chirped(
                    minF0=self.SIMMParams['minF0'],
                    maxF0=self.SIMMParams['maxF0'],
                    cqtfmax=self.stftParams['cqtfmax'],
                    cqtfmin=self.stftParams['cqtfmin'],
                    cqtbins=self.stftParams['cqtbins'],
                    Fs=self.fs,
                    Nfft=self.stftParams['NFT'],
                    stepNotes=self.SIMMParams['stepNotes'],
                    lengthWindow=cqtwindowlength,
                    # self.stftParams['windowSizeInSamples'],
                    Ot=0.5,
                    perF0=self.SIMMParams['chirpPerF0'],
                    depthChirpInSemiTone=.5,
                    loadWF0=True,
                    cqtWinFunc=self.stftParams['cqtWinFunc'],
                    atomHopFactor=self.stftParams['cqtAtomHopFactor'],
                    analysisWindow='sinebell',
                    verbose=self.verbose)
                )
            self.SIMMParams['WF0'] = WF0 / np.sum(WF0, axis=0)
            
            # number of harmonic combs
            self.SIMMParams['NF0'] = self.SIMMParams['F0Table'].size
            # self.F is different from the STFT one, resetting it:
            self.F = WF0.shape[0]
            
            self.stftParams['hopsize'] = self.mqt.cqtkernel.atomHOP
            self.stftParams['NFT'] = self.mqt.cqtkernel.linFTLen
            self.stftParams['windowSizeInSamples'] = (
                self.mqt.cqtkernel.linFTLen
                * (2 **(self.mqt.octaveNr-1))
                ) # 20130405T0355 DJL should guarantee better for chunk sizes
        else:
            self.mqt = tft.tftransforms[self.tfrepresentation](
                fmin=self.stftParams['cqtfmin'],
                fmax=self.stftParams['cqtfmax'],
                bins=self.stftParams['cqtbins'],
                fs=self.fs,
                linFTLen=self.stftParams['NFT'],
                atomHopFactor=self.stftParams['cqtAtomHopFactor'],
                winFunc=self.stftParams['cqtWinFunc'],
                perfRast=1,
                verbose=self.verbose
                )
            self.SIMMParams['F0Table'], WF0, self.mqt = (
                slf.generate_WF0_TR_chirped(
                    transform=self.mqt,
                    minF0=self.SIMMParams['minF0'],
                    maxF0=self.SIMMParams['maxF0'],
                    stepNotes=self.SIMMParams['stepNotes'],
                    Ot=0.5, perF0=self.SIMMParams['chirpPerF0'], 
                    depthChirpInSemiTone=0.5, loadWF0=True,
                    verbose=self.verbose))
            self.SIMMParams['WF0'] = WF0 / np.sum(WF0, axis=0)
            
            # number of harmonic combs
            self.SIMMParams['NF0'] = self.SIMMParams['F0Table'].size
            # self.F is different from the STFT one, resetting it:
            self.F = WF0.shape[0]
            
            if hasattr(self.mqt, 'cqtkernel'):
                # updating the following parameters if the transform
                # is a CQT type transform.
                self.stftParams['hopsize'] = self.mqt.cqtkernel.atomHOP
                self.stftParams['NFT'] = self.mqt.cqtkernel.FFTLen
                #self.stftParams['windowSizeInSamples'] = (
                #    self.mqt.cqtkernel.Nk_max
                #    * self.mqt.octaveNr) # to be ckeched...
                self.stftParams['windowSizeInSamples'] = (
                    self.mqt.cqtkernel.FFTLen
                    * (2**(self.mqt.octaveNr-1))
                    ) # 20130405T0355 DJL better maybe...
    
    def computeMonoX(self, start=0, stop=None):
        """Computes and return SX, the mono channel or mean over the
        channels of the power spectrum of the signal
        """
        fs, data = wav.read(self.files['inputAudioFilename'])
        data = np.double(data) / self.scaleData
        if len(data.shape)>1 and data.shape[1]>1:
            data = data.mean(axis=1)
            
        if self.tfrepresentation == 'stft':
            X, F, N = slf.stft(data, fs=self.fs,
                               hopsize=self.stftParams['hopsize'] ,
                               window=slf.sinebell(\
                               self.stftParams['windowSizeInSamples']),
                               nfft=self.stftParams['NFT'] ,
                               start=start, stop=stop)
            del data, F, N
            self.F, _ = X.shape
            # careful ! F and N are therefore for the whole signal!
            # X = X[:,start:stop]
            return np.maximum(np.abs(X)**2, 10 ** -8)
        elif self.tfrepresentation in knownTransfos:
            # start is in frames, same for stop, therefore have to convert
            # according to parameters of hybridcqt:
            start *= self.mqt.cqtkernel.atomHOP
            if stop is not None:
                # stop *= self.mqt.cqtkernel.atomHOP
                stop = (stop - 1) * self.mqt.cqtkernel.atomHOP 
                stop += self.stftParams['windowSizeInSamples'] #20130318
            else:
                stop = data.shape[0]
            data = data[start:stop]
            self.mqt.computeTransform(data=data)
            SX = np.maximum(
                np.abs(self.mqt.transfo)**2,
                10 ** -8)
            del self.mqt.transfo
            return SX
        
    def computeNFrames(self):
        """
        compute Nb Frames: 
        """
        if not hasattr(self, 'totFrames'):
            if self.tfrepresentation in knownTransfos:
                # NB for hybridcqt should be the same formula,
                # but the values are a bit different in nature.
                fs, data = wav.read(self.files['inputAudioFilename'])
                self.lengthData = data.shape[0]
                self.totFrames = (
                    np.int32(np.ceil((self.lengthData - 
                                      0) / # self.stftParams['windowSizeInSamples']) /
                                     self.stftParams['hopsize']
                                     + 1) + 1)# same number as in slf.stft
                    )
                self.N = self.totFrames
                
        return self.totFrames
    
    def computeStereoX(self, start=0, stop=None, ):
        """Compute the transform on each of the channels.
        
        TODO this function should be modified such that we only use the
        :py:class:`pyfasst.tftransforms.tft.TFTransform` framework. This
        could prove complicated though (especially for multiple chunk
        processing.). Current state (20130820): hack mainly focussed on STFT
        as a TF representation.
        """
        fs, data = wav.read(self.files['inputAudioFilename'])
        data = np.double(data) / self.scaleData
        if self.tfrepresentation == 'stft':
            starttime = start * self.stftParams['hopsize']
            if stop is not None:
                stoptime = stop * self.stftParams['hopsize']
            else:
                stoptime = data.shape[0]
            self.originalDataLen = stoptime - starttime
            
            if len(data.shape)>1:
                self.XR, F, N = slf.stft(
                    data[:, 0],#[starttime:stoptime,0],
                    fs=self.fs,
                    hopsize=self.stftParams['hopsize'] ,
                    window=slf.sinebell(
                        self.stftParams['windowSizeInSamples']),
                    nfft=self.stftParams['NFT'],
                    start=start, stop=stop )
                    # not very useful in practice
            else:
                self.XR, F, N = slf.stft(
                    data,#[starttime:stoptime],
                    fs=self.fs,
                    hopsize=self.stftParams['hopsize'] ,
                    window=slf.sinebell(
                        self.stftParams['windowSizeInSamples']),
                    nfft=self.stftParams['NFT'],
                    start=start, stop=stop )
            #self.XR = self.XR[:,start:stop]
            if len(data.shape)>1 and data.shape[1]>1:
                self.XL, F, N = slf.stft(
                    data[:,1], #[starttime:stoptime,1],
                    fs=self.fs,
                    hopsize=self.stftParams['hopsize'] ,
                    window=slf.sinebell(
                        self.stftParams['windowSizeInSamples']),
                    nfft=self.stftParams['NFT'],
                    start=start, stop=stop)
            else:
                self.XL = self.XR
            del F, N
        elif self.tfrepresentation in knownTransfos:
            start *= self.mqt.cqtkernel.atomHOP
            if stop is not None:
                # stop *= self.mqt.cqtkernel.atomHOP
                stop = (stop - 1) * self.mqt.cqtkernel.atomHOP 
                stop += self.stftParams['windowSizeInSamples'] #20130318
            else:
                stop = data.shape[0]
            # also works for multi channel data:
            data = data[start:stop]
            if len(data.shape)>1:
                self.mqt.computeTransform(data=data[:,0])
            else:
                self.mqt.computeTransform(data=data)
            self.XR = np.copy(self.mqt.transfo)
            del self.mqt.transfo
            if len(data.shape)>1 and data.shape[1]>1:
                self.mqt.computeTransform(data=data[:,1])
                self.XL = np.copy(self.mqt.transfo)
                del self.mqt.transfo
            else:
                # hybt.computeHybrid(data=data)
                self.XL = self.XR
        else:
            raise AttributeError(self.tfrepresentation
                                 + " not fully implemented.")
        
        #self.XL = self.XL[:,start:stop]
        del data
        self.F, _ = self.XR.shape
        
    def computeStereoSX(self, start=0, stop=None, ):
        fs, data = wav.read(self.files['inputAudioFilename'])
        data = np.double(data) / self.scaleData
        if self.tfrepresentation == 'stft':
            starttime = start * self.stftParams['hopsize']
            if stop is not None:
                stoptime = stop * self.stftParams['hopsize']
            else:
                stoptime = data.shape[0]
            self.originalDataLen = stoptime - starttime
            
            if len(data.shape)>1: # multichannel
                XR, F, N = slf.stft(
                    data[:,0], #[starttime:stoptime,0],
                    fs=self.fs,
                    hopsize=self.stftParams['hopsize'] ,
                    window=slf.sinebell(
                        self.stftParams['windowSizeInSamples']),
                    nfft=self.stftParams['NFT'] ,
                    start=start, stop=stop)
            else: # single channel
                XR, F, N = slf.stft(
                    data,#[starttime:stoptime],
                    fs=self.fs,
                    hopsize=self.stftParams['hopsize'] ,
                    window=slf.sinebell(
                        self.stftParams['windowSizeInSamples']),
                    nfft=self.stftParams['NFT'] ,
                    start=start, stop=stop)
            SXR = np.maximum(np.abs(XR)**2, 1e-8)
            del XR
            #XR = XR[:,start:stop]
            if len(data.shape)>1 and data.shape[1]>1:
                XL, F, N = slf.stft(
                    data[:,1],#[starttime:stoptime,1],
                    fs=self.fs,
                    hopsize=self.stftParams['hopsize'] ,
                    window=slf.sinebell(
                        self.stftParams['windowSizeInSamples']),
                    nfft=self.stftParams['NFT'] ,
                    start=start, stop=stop)
                SXL = np.maximum(np.abs(XL)**2, 1e-8)
                del XL, F, N
            else:
                SXL = SXR
        elif self.tfrepresentation in knownTransfos:
            start *= self.mqt.cqtkernel.atomHOP
            if stop is not None:
                stop = (stop - 1) * self.mqt.cqtkernel.atomHOP 
                stop += self.stftParams['windowSizeInSamples'] #20130318
            else: 
                stop = data.shape[0]
            # also works for multi channel data:
            data = data[start:stop]
            if len(data.shape)>1:
                self.mqt.computeTransform(data=data[:,0])
            else:
                self.mqt.computeTransform(data=data)
            SXR = np.maximum(np.abs(self.mqt.transfo)**2,10 ** -8)
            del self.mqt.transfo
            if len(data.shape)>1 and data.shape[1]>1:
                self.mqt.computeTransform(data=data[:,1])
                SXL = np.maximum(np.abs(self.mqt.transfo)**2,10 ** -8)
                del self.mqt.transfo
            else:
                # hybt.computeHybrid(data=data)
                SXL = SXR
        else:
            raise NotImplementedError("Transform %s not fully implemented"
                                      %self.tfrepresentation)
        
        #XL = XL[:,start:stop]
        del data
        self.F, _ = SXR.shape
        return SXR, SXL
    
    def estimSIMMParams(self, R=1):
        ## section to estimate the melody, on monophonic algo:
        SX = self.computeMonoX()
        # First round of parameter estimation:
        print "    Estimating IMM parameters, on mean of channels, with",R,\
              "\n    accompaniment components."
        HGAMMA, HPHI, HF0, HM, WM, recoError1 = SIMM.SIMM(
            # the data to be fitted to:
            SX,
            # the basis matrices for the spectral combs
            WF0=self.SIMMParams['WF0'],
            # and for the elementary filters:
            WGAMMA=self.SIMMParams['WGAMMA'],
            # number of desired filters, accompaniment spectra:
            numberOfFilters=self.SIMMParams['K'],
            numberOfAccompanimentSpectralShapes=R,#self.SIMMParams['R'],
            # putting only 2 elements in accompaniment for a start...
            # if any, initial amplitude matrices for 
            HGAMMA0=None, HPHI0=None,
            HF00=None,
            WM0=None, HM0=None,
            # Some more optional arguments, to control the "convergence"
            # of the algo
            numberOfIterations=self.SIMMParams['niter'],
            updateRulePower=1.,
            stepNotes=self.SIMMParams['stepNotes'], 
            lambdaHF0 = 0.0 / (1.0 * SX.max()), alphaHF0=0.9,
            verbose=self.verbose,
            displayEvolution=self.displayEvolution,
            imageCanvas=self.imageCanvas,
            F0Table=self.SIMMParams['F0Table'],
            chirpPerF0=self.SIMMParams['chirpPerF0'])
        
        self.SIMMParams['HGAMMA'] = HGAMMA
        self.SIMMParams['HPHI'] = HPHI
        self.SIMMParams['HF0'] = HF0
        self.SIMMParams['HM'] = HM
        self.SIMMParams['WM'] = WM
        del SX

    def estimHF0(self, R=1, maxFrames=1000):
        """
        estimating and storing only HF0 for the whole excerpt,
        with only 
        """
        ## section to estimate the melody, on monophonic algo:
        #SX = self.computeMonoX() # too heavy, try to guess before hand instead
        #totFrames = SX.shape[1]
        totFrames, nChunks, maxFrames = self.checkChunkSize(maxFrames)
        # First round of parameter estimation:
        print "    Estimating IMM parameters, on mean of channels, with",R,\
              "\n    accompaniment components."+\
              "    Nb of chunks: %d." %nChunks
        # del SX
        self.SIMMParams['HF0'] = np.zeros([self.SIMMParams['NF0'] * \
                                           self.SIMMParams['chirpPerF0'],
                                           totFrames])
        for n in range(nChunks):
            if self.verbose:
                print "Chunk nb", n+1, "out of", nChunks
            start = n*maxFrames
            stop = np.minimum((n+1)*maxFrames, totFrames)
            SX = self.computeMonoX(start=start, stop=stop)
            if self.SIMMParams['initHF00'] == 'nnls':
                # probably slower than running from random...
                HF00 = np.ones((self.SIMMParams['NF0']
                                * self.SIMMParams['chirpPerF0'],
                                stop-start))
                for framenb in range(stop-start):
                    if self.verbose>1:
                        print "frame", framenb
                    HF00[:,framenb], _ = scipy.optimize.nnls(
                        self.SIMMParams['WF0'],
                        SX[:,framenb])
                HF00 += eps
            else:
                HF00 = None
            HGAMMA, HPHI, HF0, HM, WM, recoError1 = SIMM.SIMM(
                # the data to be fitted to:
                SX,
                # the basis matrices for the spectral combs
                WF0=self.SIMMParams['WF0'],
                # and for the elementary filters:
                WGAMMA=self.SIMMParams['WGAMMA'],
                # number of desired filters, accompaniment spectra:
                numberOfFilters=self.SIMMParams['K'],
                numberOfAccompanimentSpectralShapes=R,#self.SIMMParams['R'],
                # putting only 2 elements in accompaniment for a start...
                # if any, initial amplitude matrices for 
                HGAMMA0=None, HPHI0=None,
                HF00=HF00, 
                WM0=None, HM0=None,
                # Some more optional arguments, to control the "convergence"
                # of the algo
                numberOfIterations=self.SIMMParams['niter'],
                updateRulePower=1.,
                stepNotes=self.SIMMParams['stepNotes'], 
                lambdaHF0 = 0.0 / (1.0 * SX.max()), alphaHF0=0.9,
                verbose=self.verbose,
                displayEvolution=self.displayEvolution,
                imageCanvas=self.imageCanvas,
                F0Table=self.SIMMParams['F0Table'],
                chirpPerF0=self.SIMMParams['chirpPerF0'])
            
            if self.tfrepresentation == 'stft':
                self.SIMMParams['HF0'][:,start:stop] = np.copy(HF0)
            elif self.tfrepresentation in knownTransfos:
                # the first frame of interest in the CQT representation,
                # for our purpose at least
                startincqt = np.sort(np.where(self.mqt.time_stamps>0)[0])[0]
                # and the last:
                stopincqt = (startincqt
                             + stop - start)
                self.SIMMParams['HF0'][:,start:stop] = (
                    np.copy(HF0[:,startincqt:stopincqt]))
            
            del SX
        
        F0Table=self.SIMMParams['F0Table']
        NF0 = self.SIMMParams['NF0'] * self.SIMMParams['chirpPerF0']
        db = SIMM.db
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
            imgYticklabels = np.int32(F0Table[np.array(imgYticks)/
                                              self.SIMMParams['chirpPerF0']
                                              ]).tolist()
            for k, v in notesFreqs.items():
                closestIndex = np.argmin(np.abs(F0Table-v))
                if np.abs(12*np.log2(F0Table[closestIndex])-\
                          12*np.log2(v)) < .25:
                    imgYticks.append(closestIndex)
                    imgYticklabels.append(k)
        if self.imageCanvas is not None:
            self.imageCanvas.ax.clear()
            self.imageCanvas.ax.imshow(db(self.SIMMParams['HF0']),
                                       origin='lower',
                                       cmap='jet',
                                       aspect='auto',
                                       interpolation='nearest')
            self.imageCanvas.ax.get_images()[0].set_clim(\
                np.amax(db(self.SIMMParams['HF0']))-100,\
                np.amax(db(self.SIMMParams['HF0'])))
            self.imageCanvas.ax.set_yticks(imgYticks)
            self.imageCanvas.ax.set_yticklabels(imgYticklabels)
            self.imageCanvas.draw()
            # self.imageCanvas.updateGeometry()
    
    def computeChroma(self, maxFrames=3000):
        """Compute the chroma matrix.
        """
        if hasattr(self, 'SIMMParams'):
            if 'HF0' not in self.SIMMParams.keys():
                self.estimHF0(maxFrames=maxFrames)
        else:
            raise AttributeError("The parameters for the SIMM are not"+\
                                 " well initialized")
        if not hasattr(self, 'N'):
            warnings.warn("Issues with the attributes, running again"+\
                          " the estimation.")
            self.estimHF0(maxFrames=maxFrames)
        
        self.chroma = np.zeros([12*self.SIMMParams['stepNotes'],
                                self.computeNFrames()])#self.N])
        for n in range(12*self.SIMMParams['stepNotes']):
            self.chroma[n] = \
                self.SIMMParams['HF0'][\
                n::(12*self.SIMMParams['stepNotes'])].mean(axis=0)
        self.chroma /= (self.chroma.sum(axis=0))
        
    def determineTuning(self):
        """Determine Tuning by checking the peaks corresponding
        to all possible patterns
        """
        if not hasattr(self, 'chroma'):
            self.computeChroma()
            
        chromaSummary = self.chroma.sum(axis=1)
        
        patterns = {}
        patterns['minorHarmoPattern'] = np.array([0,2,3,5,7,8,10])
        patterns['minorMelodPattern'] = np.array([0,2,3,5,7,9,11])
        patterns['majorPattern']      = np.array([0,2,4,5,7,9,11])
        patterns['andalusPattern']    = np.array([0,1,4,5,7,8,11])
        
        nbPattern = len(patterns.keys())
        nbTunings = self.SIMMParams['stepNotes']
        nbKey = 12
        scoresPerTuning = np.zeros([nbPattern, nbTunings*nbKey])
        for ntun in range(nbTunings):
            for nk in range(nbKey):
                for npatt, pattern in enumerate(patterns.keys()):
                    scoresPerTuning[npatt, ntun+nk*nbTunings] = \
                        chromaSummary[np.mod((patterns[pattern]+nk)*\
                                             nbTunings+\
                                             ntun, chromaSummary.size)].sum()
        bestTuning = np.argmax(scoresPerTuning)
        bestPattern = bestTuning / (nbTunings * nbKey)
        bestTuning = bestTuning - bestPattern * (nbTunings * nbKey)
        bestKey = bestTuning / nbTunings
        bestTuning = bestTuning - bestKey * nbTunings 
        return scoresPerTuning, bestTuning, \
               bestKey, patterns.keys()[bestPattern]
    
    def automaticMelodyAndSeparation(self):
        """Fully automated estimation of melody and separation of signals.
        """
        raise warnings.warn("This function does not work well with framed " + 
                            "estimation.")
        self.runViterbi()
        self.initiateHF0WithIndexBestPath()
        self.estimStereoSIMMParams()
        self.writeSeparatedSignals()
        self.estimStereoSUIMMParams()
        self.writeSeparatedSignalsWithUnvoice()
    
    def autoMelSepAndWrite(self, maxFrames=1000):
        """Fully automated estimation of melody and separation of signals.
        """
        self.estimHF0(maxFrames=maxFrames)
        self.runViterbi()
        self.initiateHF0WithIndexBestPath()
        self.estimStereoSIMMParamsWriteSeps(maxFrames=maxFrames)
    
    def runViterbi(self):
        if not('HF0' in self.SIMMParams.keys()):
            raise AttributeError("HF0 has probably not been estimated yet.")
        ##SX = self.computeMonoX() # useless here?
        self.computeNFrames() # just to be sure self.N is total nb of frames
        
        # Viterbi decoding to estimate the predominant fundamental
        # frequency line
        scale = 1.0
        NF0 = self.SIMMParams['NF0'] * self.SIMMParams['chirpPerF0']
        
        # only considering the desired range of 
        nmaxF0 = NF0
        nminF0 = 0
        # just so that it s easier to manipulate:
        minF0 = self.SIMMParams['minF0']
        maxF0 = self.SIMMParams['maxF0']
        minF0search = self.trackingParams['minF0search']
        maxF0search = self.trackingParams['maxF0search']
        if minF0search > minF0 and minF0search < maxF0:
            nminF0 = (
                np.where(self.SIMMParams['F0Table']>=minF0search)[0][0]
                * self.SIMMParams['chirpPerF0'])
        if maxF0search > minF0 and maxF0search < maxF0 and \
               maxF0search > minF0search:
            nmaxF0 = (
                (np.where(self.SIMMParams['F0Table']>=maxF0search)[0][0] + 1)
                * self.SIMMParams['chirpPerF0'])
            
        NF0 = nmaxF0 - nminF0
        print nminF0, nmaxF0 #DEBUG
        
        # filling the transitions probabilities
        transitions = np.exp(-np.floor(np.arange(0, NF0)/\
                                       self.SIMMParams['stepNotes']) * \
                             scale)
        cutoffnote = 2 * 5 * self.SIMMParams['stepNotes']
        cutoffnote = np.minimum(NF0, cutoffnote)
        transitions[cutoffnote:] = transitions[cutoffnote - 1]
        transitionMatrixF0 = np.zeros([NF0 + 1, NF0 + 1]) # toeplitz matrix
        b = np.arange(NF0)
        transitionMatrixF0[0:NF0, 0:NF0] = \
            transitions[\
                np.array(np.abs(np.outer(np.ones(NF0), b) \
                                - np.outer(b, np.ones(NF0))), dtype=int)]
        pf_0 = transitions[cutoffnote - 1] * 10 ** (-90)
        p0_0 = transitions[cutoffnote - 1] * 10 ** (-100)
        p0_f = transitions[cutoffnote - 1] * 10 ** (-80)
        transitionMatrixF0[0:NF0, NF0] = pf_0
        transitionMatrixF0[NF0, 0:NF0] = p0_f
        transitionMatrixF0[NF0, NF0] = p0_0
        
        sumTransitionMatrixF0 = np.sum(transitionMatrixF0, axis=1)
        transitionMatrixF0 = transitionMatrixF0 \
                             / np.outer(sumTransitionMatrixF0, \
                                        np.ones(NF0 + 1))
        
        priorProbabilities = 1 / (NF0 + 1.0) * np.ones([NF0 + 1])
        
        logHF0 = np.zeros([NF0 + 1, self.N])
        normHF0 = np.amax(self.SIMMParams['HF0'][nminF0:nmaxF0], axis=0)
        
        logHF0[0:NF0, :] = np.log(self.SIMMParams['HF0'][nminF0:nmaxF0])
        logHF0[0:NF0, normHF0==0] = np.amin(logHF0[logHF0>-np.Inf])
        logHF0[NF0, :] = np.maximum(np.amin(logHF0[logHF0>-np.Inf]),-100)
        # free all what s not needed anymore:
        del normHF0, transitions, b
        
        print "Running Viterbi algorithm to track the melody, " + \
              str(self.N) + " frames."
        indexBestPath = viterbiTrackingArray(NF0, self.N,\
            logHF0, np.log(priorProbabilities),
            np.log(transitionMatrixF0), verbose=False)
        indexBestPath += nminF0
        print "Viterbi algorithm done..."
        
        # drawing this as a line is actually a bit confusing, on the image
        #     TODO: think of a better representation (is contour good enough?)
        ##if self.displayEvolution and not(self.imageCanvas is None):
        ##    self.imageCanvas.ax.plot(indexBestPath, '-b')
        ##    self.imageCanvas.ax.axis('tight')
        ##    self.imageCanvas.draw()
        
        del logHF0
        
        # detection of silences:
        if 'HPHI' in self.SIMMParams and False: # in case not estimated
            # TODO: this is broken, when nchunks > 1
            #     needs a fix, maybe keeping relative energy as
            #     attribute, instead of computing it here.
            chirpPerF0 = self.SIMMParams['chirpPerF0']
            stepNotes = self.SIMMParams['stepNotes']
            HF00 = np.zeros([NF0 * chirpPerF0, self.N])
            scopeAllowedHF0 = self.scopeAllowedHF0# 4.0 / 1.0 # 2.0 / 1.0
            dim1index = np.array(\
                np.maximum(\
                    np.minimum(\
                        np.outer(chirpPerF0 * indexBestPath,
                                 np.ones(chirpPerF0 \
                                         * (2 \
                                            * np.floor(stepNotes / \
                                                       scopeAllowedHF0) \
                                            + 1))) \
                        + np.outer(np.ones(self.N),
                                   np.arange(-chirpPerF0 \
                                             * np.floor(stepNotes / \
                                                        scopeAllowedHF0),
                                             chirpPerF0 \
                                             * (np.floor(stepNotes / \
                                                         scopeAllowedHF0) \
                                                + 1))),
                        chirpPerF0 * NF0 - 1),
                    0),
                dtype=int).reshape(1, self.N * chirpPerF0 \
                                   * (2 * np.floor(stepNotes/scopeAllowedHF0)\
                                      + 1))
            dim2index = np.outer(np.arange(self.N),
                                 np.ones(chirpPerF0 \
                                         * (2 * np.floor(stepNotes \
                                                         /scopeAllowedHF0)+1),\
                                         dtype=int)\
                                 ).reshape(1, self.N * chirpPerF0 \
                                           * (2 * np.floor(stepNotes \
                                                           / scopeAllowedHF0) \
                                              + 1))
            HF00[dim1index, dim2index] = self.SIMMParams['HF0'][dim1index,
                                                                dim2index]
            
            HF00[:, indexBestPath == (NF0 - 1)] = 0.0
            HF00[:, indexBestPath == 0] = 0.0
            
            thres_energy = 0.000584
            SF0 = np.maximum(np.dot(self.SIMMParams['WF0'], HF00), eps)
            SPHI = np.maximum(np.dot(self.SIMMParams['WGAMMA'], \
                                     np.dot(self.SIMMParams['HGAMMA'],
                                            self.SIMMParams['HPHI'])), eps)
            SM = np.maximum(np.dot(self.SIMMParams['WM'], \
                                   self.SIMMParams['HM']), eps)
            hatSX = np.maximum(SPHI * SF0 + SM, eps)
            SX = self.computeMonoX()
            energyMel = np.sum((((SPHI * SF0)/hatSX)**2) * np.abs(SX),
                               axis=0)
            energyMelSorted = np.sort(energyMel)
            energyMelCumul = np.cumsum(energyMelSorted)
            energyMelCumulNorm = energyMelCumul / max(energyMelCumul[-1], eps)
            # normalized to the maximum of energy:
            # expressed in 0.01 times the percentage
            ind_999 = np.nonzero(energyMelCumulNorm>thres_energy)[0][0]
            if ind_999 is None:
                ind_999 = self.N
                
            melNotPresent = (energyMel <= energyMelCumulNorm[ind_999])
            indexBestPath[melNotPresent] = 0
        else:
            if self.verbose:
                print "    Not using energy threshold, since "+\
                      "parameters were deleted."
        
        freqMelody = self.SIMMParams['F0Table'][np.array(
            indexBestPath
            /self.SIMMParams['chirpPerF0'],
            dtype=int)]
        freqMelody[indexBestPath==0] = - freqMelody[indexBestPath==0]
        np.savetxt(self.files['pitch_output_file'],
                   np.array([np.arange(self.N) * \
                             self.stftParams['hopsize'] / np.double(self.fs),
                             freqMelody]).T)
        
        self.indexBestPath = indexBestPath
        self.freqMelody = freqMelody
    
    def initiateHF0WithIndexBestPath(self):
        # Second round of parameter estimation, with specific
        # initial HF00:
        NF0 = self.SIMMParams['NF0']
        chirpPerF0 = self.SIMMParams['chirpPerF0']
        stepNotes = self.SIMMParams['stepNotes']
        
        HF00 = np.zeros([NF0 * chirpPerF0, self.N])
        
        scopeAllowedHF0 = self.scopeAllowedHF0 # 2.0 / 1.0
        
        # indexes for HF00:
        # TODO: reprogram this with a 'where'?...
        dim1index = np.array(\
            np.maximum(\
            np.minimum(\
                np.outer(self.indexBestPath,# * chirpPerF0 #20130610 DJL???
                         np.ones(chirpPerF0 \
                                 * (2 \
                                    * np.floor(stepNotes / scopeAllowedHF0) \
                                    + 1))) \
                + np.outer(np.ones(self.N),
                           np.arange(-chirpPerF0 \
                                     * np.floor(stepNotes / scopeAllowedHF0),
                                     chirpPerF0 \
                                     * (np.floor(stepNotes / scopeAllowedHF0) \
                                        + 1))),
                chirpPerF0 * NF0 - 1),
            0),
            dtype=int)
        dim1index = dim1index[self.indexBestPath!=0,:]
        dim1index = dim1index.reshape(1,dim1index.size)
        
        dim2index = np.outer(np.arange(self.N),
                             np.ones(chirpPerF0 \
                                     * (2 * np.floor(stepNotes \
                                                     / scopeAllowedHF0) + 1), \
                                     dtype=int)\
                             )
        dim2index = dim2index[self.indexBestPath!=0,:]
        dim2index = dim2index.reshape(1,dim2index.size)
        
        HF00[dim1index, dim2index] = self.SIMMParams['HF0'].max()
        
        HF00[:, self.indexBestPath == (NF0 - 1)] = 0.0
        HF00[:, self.indexBestPath == 0] = 0.0
        
        self.SIMMParams['HF00'] = HF00
    
    def estimStereoSIMMParamsWriteSeps(self, maxFrames=1000):
        """Estimates the parameters little by little, by chunks,
        and sequentially writes the signals. In the end, concatenates all these
        separated signals into the desired output files
        """
        #SX = self.computeMonoX()
        totFrames, nChunks, maxFrames = self.checkChunkSize(maxFrames)
        # del SX
        
        # First round of parameter estimation:
        print "    Estimating IMM parameters, on stereo channels, with",\
              self.SIMMParams['R'],\
              "\n    accompaniment components."+\
              "    Nb of chunks: %d." %nChunks
        
        self.SIMMParams['HGAMMA'] = None
        for n in range(nChunks):
            if self.verbose:
                print "Chunk nb", n+1, "out of", nChunks
            start = n*maxFrames
            stop = np.minimum((n+1)*maxFrames, totFrames)
            # computing only the power spectra for each channel:
            #    - not storing the complex spectra -
            SXR, SXL = self.computeStereoSX(start=start, stop=stop)
            HF00 = np.zeros([self.SIMMParams['NF0']
                             * self.SIMMParams['chirpPerF0'],
                             SXR.shape[1]])
            if self.tfrepresentation == 'stft':
                startinHF00 = 0
                stopinHF00 = stop - start
            elif self.tfrepresentation in knownTransfos:
                startinHF00 = np.sort(np.where(self.mqt.time_stamps>0)[0])[0]
                stopinHF00 = startinHF00 + stop - start
            else:
                raise AttributeError(self.tfrepresentation
                                     + " not fully implemented.")
            HF00[:,startinHF00:stopinHF00] = (
                self.SIMMParams['HF00'][:,start:stop])
            alphaR, alphaL, HGAMMA, HPHI, HF0, \
                betaR, betaL, HM, WM, recoError2 = SIMM.Stereo_SIMM(
                # the data to be fitted to:
                SXR, SXL,
                # the basis matrices for the spectral combs
                WF0=self.SIMMParams['WF0'],
                # and for the elementary filters:
                WGAMMA=self.SIMMParams['WGAMMA'],
                # number of desired filters, accompaniment spectra:
                numberOfFilters=self.SIMMParams['K'],
                numberOfAccompanimentSpectralShapes=self.SIMMParams['R'], 
                # if any, initial amplitude matrices for
                HGAMMA0=self.SIMMParams['HGAMMA'],
                HPHI0=None,
                HF00=HF00,
                WM0=None, HM0=None,
                # Some more optional arguments, to control the "convergence"
                # of the algo
                numberOfIterations=self.SIMMParams['niter'],
                updateRulePower=1.0,
                stepNotes=self.SIMMParams['stepNotes'],
                lambdaHF0 = 0.0 / (1.0 * SXR.max()), alphaHF0=0.9,
                verbose=self.verbose, displayEvolution=False)
            
            self.SIMMParams['HGAMMA'] = HGAMMA
            self.SIMMParams['HPHI'] = HPHI
            self.SIMMParams['HF0'] = HF0
            self.SIMMParams['HM'] = HM
            self.SIMMParams['WM'] = WM
            self.SIMMParams['alphaR'] = alphaR
            self.SIMMParams['alphaL'] = alphaL
            self.SIMMParams['betaR'] = betaR
            self.SIMMParams['betaL'] = betaL
            
            # keeping the estimated HF0 in memory:
            self.SIMMParams['HF00'][:,start:stop] = (
                np.copy(HF0[:,startinHF00:stopinHF00]))
            
            
            del SXR, SXL, HF00
            
            # computing and storing the complex spectra
            self.computeStereoX(start=start, stop=stop)
            
            # writing the separated signals as output wavfile with suffix
            # equal to the chunk number
            self.writeSeparatedSignals(suffix='%05d.wav'%n)
            
            # freeing memory
            del self.XR, self.XL
            if self.freeMemory:
                del self.SIMMParams['HM'], self.SIMMParams['HF0']
                del self.SIMMParams['HPHI']
                del self.SIMMParams['alphaR'], self.SIMMParams['alphaL']
                del self.SIMMParams['betaR'], self.SIMMParams['betaL']
                                
        # Now concatenating the wav files
        self.overlapAddChunks(
            nChunks=nChunks,
            suffixIsSUIMM='.wav')
        
    def overlapAddChunks(self, nChunks,
                         suffixIsSUIMM='.wav'):
        # Now concatenating the wav files
        wlen = self.stftParams['windowSizeInSamples']
        offsetTF = self.stftParams['offsets'][self.tfrepresentation]
        # overlap add on the chunks:
        if self.tfrepresentation == 'stft':
            hopsize = self.stftParams['hopsize']
            overlapSamp = wlen - hopsize
            # for stft, the overlap is taken into account at computation
            # using rectangle synthesis function:
            overlapFunc = np.ones(overlapSamp)
        elif self.tfrepresentation in knownTransfos:
            hopsize = self.mqt.cqtkernel.atomHOP
            # for hybridcqt, have to compensate the overlap procedure:
            overlapSamp = wlen - hopsize
            # using sinebell ** 2 for overlapping function
            # (rectangle as analysis function for hybridcqt):
            overlapFunc = slf.sinebell(2 * overlapSamp)[overlapSamp:]**2
            if self.verbose>3:
                print "[DEBUG] check that window adds to 1:",
                print overlapFunc + overlapFunc[::-1]
        nuDataLen = (
            self.totFrames * hopsize
            + 2 * wlen)
        data = np.zeros([nuDataLen, 2], np.int16)
        
        cumulframe = 0
        ##data = []
        for n in range(nChunks):
            suffix='%05d%s'%(n, suffixIsSUIMM)
            fname = self.files['voc_output_file'][:-4] + suffix
            # data.append(wav.read(fname)[1])
            _, datatmp = wav.read(fname)
            datatype = type(datatmp[0][0])
            print datatype
            if n == 0 and nChunks!=1:
                # weighing by the overlapping function
                datatmp[-overlapSamp:,0] = datatype(
                    datatmp[-overlapSamp:,0]* overlapFunc)
                datatmp[-overlapSamp:,1] = datatype(
                    datatmp[-overlapSamp:,1]* overlapFunc)
                lendatatmp = (datatmp.shape[0] - offsetTF) # (datatmp.shape[0] - wlen/2)
                data[:lendatatmp, :] = np.copy(
                    datatmp[offsetTF:, :])# datatmp[wlen/2:, :])
                cumulframe = lendatatmp
            elif nChunks != 1:
                # weighing by the overlapping function
                if n!=nChunks-1:
                    datatmp[-overlapSamp:,0] = datatype(
                        datatmp[-overlapSamp:,0] * overlapFunc)
                    datatmp[-overlapSamp:,1] = datatype(
                        datatmp[-overlapSamp:,1]* overlapFunc)
                datatmp[:overlapSamp,0] = datatype(
                    datatmp[:overlapSamp,0] * overlapFunc[::-1])
                datatmp[:overlapSamp,1] = datatype(
                    datatmp[:overlapSamp,1] * overlapFunc[::-1])
                start = cumulframe - wlen + hopsize
                lendatatmp = datatmp.shape[0]
                stop = start + lendatatmp
                data[start:stop, :] += datatmp
                cumulframe = stop
            else: # n=0 and nChunks = 1:
                lendatatmp = datatmp.shape[0] - offsetTF
                data[:lendatatmp] = datatmp[offsetTF:, :]
            os.remove(fname)
        # data = np.vstack(data)
        wav.write(self.files['voc_output_file'][:-4] + suffixIsSUIMM,
                  self.fs,
                  data[:self.lengthData,:])
        
        data = np.zeros([nuDataLen, 2], np.int16)
        # overlap add on the chunks:
        cumulframe = 0
        ##data = []
        for n in range(nChunks):
            suffix='%05d%s'%(n, suffixIsSUIMM)
            fname = self.files['mus_output_file'][:-4] + suffix
            #data.append(wav.read(fname)[1])
            _, datatmp = wav.read(fname)
            datatype = type(datatmp[0][0])
            if n == 0 and nChunks!=1:
                # weighing by the overlapping function
                datatmp[-overlapSamp:,0] = datatype(
                    datatmp[-overlapSamp:,0]* overlapFunc)
                datatmp[-overlapSamp:,1] = datatype(
                    datatmp[-overlapSamp:,1]* overlapFunc)
                lendatatmp = (datatmp.shape[0] - offsetTF) # (datatmp.shape[0] - wlen/2)
                data[:lendatatmp, :] = np.copy(
                    datatmp[offsetTF:, :])# datatmp[wlen/2:, :])
                cumulframe = lendatatmp
            elif nChunks != 1:
                # weighing by the overlapping function
                if n!=nChunks-1:
                    datatmp[-overlapSamp:,0] = datatype(
                        datatmp[-overlapSamp:,0] * overlapFunc)
                    datatmp[-overlapSamp:,1] = datatype(
                        datatmp[-overlapSamp:,1] * overlapFunc)
                datatmp[:overlapSamp,0] = datatype(
                    datatmp[:overlapSamp,0] * overlapFunc[::-1])
                datatmp[:overlapSamp,1] = datatype(
                    datatmp[:overlapSamp,1] * overlapFunc[::-1])
                start = cumulframe - wlen + hopsize
                lendatatmp = datatmp.shape[0]
                stop = start + lendatatmp
                data[start:stop, :] += datatmp
                cumulframe = stop
            else: # n=0 and nChunks = 1:
                lendatatmp = datatmp.shape[0] - offsetTF
                data[:lendatatmp] = datatmp[offsetTF:, :]
            os.remove(fname)
        #data = np.vstack(data)
        wav.write(self.files['mus_output_file'][:-4] + suffixIsSUIMM,
                  self.fs,
                  data[:self.lengthData,:])
    
    def estimStereoSUIMMParamsWriteSeps(self, maxFrames=1000):
        """same as estimStereoSIMMParamsWriteSeps, but adds the unvoiced
        element in HF0
        """
        totFrames, nChunks, maxFrames = self.checkChunkSize(maxFrames)
        print "    Estimating IMM parameters, on stereo channels, with",\
              self.SIMMParams['R'],\
              "\n    accompaniment components."+\
              "    Nb of chunks: %d." %nChunks
        
        WUF0 = np.hstack([self.SIMMParams['WF0'],
                          np.ones([self.SIMMParams['WF0'].shape[0], 1])])
        self.SIMMParams['WUF0'] = WUF0
        for n in range(nChunks):
            if self.verbose:
                print "Chunk nb", n+1, "out of", nChunks
            start = n*maxFrames
            stop = np.minimum((n+1)*maxFrames, totFrames)
            SXR, SXL = self.computeStereoSX(start=start, stop=stop)
            HUF0 = np.zeros([self.SIMMParams['NF0']
                             * self.SIMMParams['chirpPerF0']
                             + 1,
                             SXR.shape[1]])
            if self.tfrepresentation == 'stft':
                startinHF00 = 0
                stopinHF00 = stop - start
            elif self.tfrepresentation in knownTransfos:
                startinHF00 = np.sort(np.where(self.mqt.time_stamps>0)[0])[0]
                stopinHF00 = startinHF00 + stop - start
            else:
                raise AttributeError(self.tfrepresentation
                                     + " not fully implemented.")
            HUF0[:-1,startinHF00:stopinHF00] = (
                self.SIMMParams['HF00'][:,start:stop])
            HUF0[-1] = 1
            alphaR, alphaL, HGAMMA, HPHI, HF0, \
                betaR, betaL, HM, WM, recoError3 = SIMM.Stereo_SIMM(
                # the data to be fitted to:
                SXR, SXL,
                # the basis matrices for the spectral combs
                WUF0,
                # and for the elementary filters:
                WGAMMA=self.SIMMParams['WGAMMA'],
                # number of desired filters, accompaniment spectra:
                numberOfFilters=self.SIMMParams['K'],
                numberOfAccompanimentSpectralShapes=self.SIMMParams['R'],
                # if any, initial amplitude matrices for
                HGAMMA0=self.SIMMParams['HGAMMA'],
                HPHI0=None,
                HF00=HUF0,
                WM0=None,#WM,
                HM0=None,#HM,
                # Some more optional arguments, to control the "convergence"
                # of the algo
                numberOfIterations=self.SIMMParams['niter'],
                updateRulePower=1.0,
                stepNotes=self.SIMMParams['stepNotes'], 
                lambdaHF0 = 0.0 / (1.0 * SXR.max()), alphaHF0=0.9,
                verbose=self.verbose, displayEvolution=False,
                updateHGAMMA=False)
        
            self.SIMMParams['HGAMMA'] = HGAMMA
            self.SIMMParams['HPHI'] = HPHI
            self.SIMMParams['HUF0'] = HF0
            self.SIMMParams['HM'] = HM
            self.SIMMParams['WM'] = WM
            self.SIMMParams['alphaR'] = alphaR
            self.SIMMParams['alphaL'] = alphaL
            self.SIMMParams['betaR'] = betaR
            self.SIMMParams['betaL'] = betaL
            
            del SXR, SXL, HUF0
            
            # computing and storing the complex spectra
            self.computeStereoX(start=start, stop=stop)
            
            # writing the separated signals as output wavfile with suffix
            # equal to the chunk number
            self.writeSeparatedSignals(suffix='%05d_VUIMM.wav'%n)
            
            # freeing memory
            del self.XR, self.XL
            del self.SIMMParams['HM'], self.SIMMParams['HUF0']
            del self.SIMMParams['HPHI']
            del self.SIMMParams['alphaR'], self.SIMMParams['alphaL']
            del self.SIMMParams['betaR'], self.SIMMParams['betaL']
        
        # Now concatenating the wav files
        self.overlapAddChunks(
            nChunks=nChunks,
            suffixIsSUIMM='_VUIMM.wav')
    
    def estimStereoSIMMParams(self):
        self.computeStereoX()
        SXR = np.abs(self.XR) ** 2
        SXL = np.abs(self.XL) ** 2
        alphaR, alphaL, HGAMMA, HPHI, HF0, \
            betaR, betaL, HM, WM, recoError2 = SIMM.Stereo_SIMM(
            # the data to be fitted to:
            SXR, SXL,
            # the basis matrices for the spectral combs
            WF0=self.SIMMParams['WF0'],
            # and for the elementary filters:
            WGAMMA=self.SIMMParams['WGAMMA'],
            # number of desired filters, accompaniment spectra:
            numberOfFilters=self.SIMMParams['K'],
            numberOfAccompanimentSpectralShapes=self.SIMMParams['R'], 
            # if any, initial amplitude matrices for
            HGAMMA0=None, HPHI0=None,
            HF00=self.SIMMParams['HF00'],
            WM0=None, HM0=None,
            # Some more optional arguments, to control the "convergence"
            # of the algo
            numberOfIterations=self.SIMMParams['niter'],
            updateRulePower=1.0,
            stepNotes=self.SIMMParams['stepNotes'],
            lambdaHF0 = 0.0 / (1.0 * SXR.max()), alphaHF0=0.9,
            verbose=self.verbose, displayEvolution=False)
        
        self.SIMMParams['HGAMMA'] = HGAMMA
        self.SIMMParams['HPHI'] = HPHI
        self.SIMMParams['HF0'] = HF0
        self.SIMMParams['HM'] = HM
        self.SIMMParams['WM'] = WM
        self.SIMMParams['alphaR'] = alphaR
        self.SIMMParams['alphaL'] = alphaL
        self.SIMMParams['betaR'] = betaR
        self.SIMMParams['betaL'] = betaL
        del SXR, SXL
    
    def estimStereoSUIMMParams(self):
        
        SXR = np.abs(self.XR) ** 2
        SXL = np.abs(self.XL) ** 2
        # adding the unvoiced part in the source basis:
        WUF0 = np.hstack([self.SIMMParams['WF0'],
                          np.ones([self.SIMMParams['WF0'].shape[0], 1])])
        HUF0 = np.vstack([self.SIMMParams['HF0'],
                          np.ones([1, self.SIMMParams['HF0'].shape[1]])])
        
        alphaR, alphaL, HGAMMA, HPHI, HF0, \
            betaR, betaL, HM, WM, recoError3 = SIMM.Stereo_SIMM(
            # the data to be fitted to:
            SXR, SXL,
            # the basis matrices for the spectral combs
            WUF0,
            # and for the elementary filters:
            WGAMMA=self.SIMMParams['WGAMMA'],
            # number of desired filters, accompaniment spectra:
            numberOfFilters=self.SIMMParams['K'],
            numberOfAccompanimentSpectralShapes=self.SIMMParams['R'],
            # if any, initial amplitude matrices for
            HGAMMA0=self.SIMMParams['HGAMMA'],
            HPHI0=self.SIMMParams['HPHI'],
            HF00=HUF0,
            WM0=None,#WM,
            HM0=None,#HM,
            # Some more optional arguments, to control the "convergence"
            # of the algo
            numberOfIterations=self.SIMMParams['niter'],
            updateRulePower=1.0,
            stepNotes=self.SIMMParams['stepNotes'], 
            lambdaHF0 = 0.0 / (1.0 * SXR.max()), alphaHF0=0.9,
            verbose=self.verbose, displayEvolution=False,
            updateHGAMMA=False)
        
        self.SIMMParams['HGAMMA'] = HGAMMA
        self.SIMMParams['HPHI'] = HPHI
        self.SIMMParams['HUF0'] = HF0
        self.SIMMParams['WUF0'] = WUF0
        self.SIMMParams['HM'] = HM
        self.SIMMParams['WM'] = WM
        self.SIMMParams['alphaR'] = alphaR
        self.SIMMParams['alphaL'] = alphaL
        self.SIMMParams['betaR'] = betaR
        self.SIMMParams['betaL'] = betaL
    
    def writeSeparatedSignals(self, suffix='.wav'):
        """Writes the separated signals to the files in self.files.
        If suffix contains 'VUIMM', then this method will take
        the WF0 and HF0 that contain the estimated unvoiced elements.
        """
        if 'VUIMM' in suffix:
            WF0    = self.SIMMParams['WUF0']
            HF0    = self.SIMMParams['HUF0']
        else:
            WF0    = self.SIMMParams['WF0']
            HF0    = self.SIMMParams['HF0']
        
        WGAMMA = self.SIMMParams['WGAMMA']
        HGAMMA = self.SIMMParams['HGAMMA']
        HPHI   = self.SIMMParams['HPHI']
        HM     = self.SIMMParams['HM']
        WM     = self.SIMMParams['WM']
        alphaR = self.SIMMParams['alphaR']
        alphaL = self.SIMMParams['alphaL']
        betaR  = self.SIMMParams['betaR']
        betaL  = self.SIMMParams['betaL']
        windowSizeInSamples = self.stftParams['windowSizeInSamples']
        
        SPHI   = np.dot(np.dot(WGAMMA, HGAMMA), HPHI)
        SF0 = np.dot(WF0, HF0)
        
        hatSXR = (alphaR**2) * SF0 * SPHI + np.dot(np.dot(WM, betaR**2),HM)
        hatSXL = (alphaL**2) * SF0 * SPHI + np.dot(np.dot(WM, betaL**2),HM)
        hatSXR = np.maximum(hatSXR, eps)
        hatSXL = np.maximum(hatSXL, eps)
        
        hatVR = (alphaR**2) * SPHI * SF0 / hatSXR * self.XR
        
        if self.tfrepresentation == 'stft':
            vestR = slf.istft(
                hatVR,
                hopsize=self.stftParams['hopsize'],
                nfft=self.stftParams['NFT'],
                window=slf.sinebell(windowSizeInSamples),
                originalDataLen=None,)#self.originalDataLen)#  / 4.0
        elif self.tfrepresentation in knownTransfos:
            self.mqt.transfo = hatVR
            vestR = self.mqt.invertTransform()
            del self.mqt.transfo
            
        
        hatVR = (alphaL**2) * SPHI * SF0 / hatSXL * self.XL
        
        del SPHI, SF0
        
        if self.tfrepresentation == 'stft':
            vestL = slf.istft(
                hatVR, 
                hopsize=self.stftParams['hopsize'],
                nfft=self.stftParams['NFT'],
                window=slf.sinebell(windowSizeInSamples),
                originalDataLen=None,)#self.originalDataLen)#  / 4.0
        elif self.tfrepresentation in knownTransfos:
            self.mqt.transfo = hatVR
            vestL = self.mqt.invertTransform()
            del self.mqt.transfo
        
        del hatVR
        
        vestR = np.array(np.round(vestR*self.scaleData), dtype=self.dataType)
        vestL = np.array(np.round(vestL*self.scaleData), dtype=self.dataType)
        
        wav.write(self.files['voc_output_file'][:-4] + suffix,
                  self.fs,
                  np.array([vestR,vestL]).T)
        
        del vestR, vestL
        
        hatMR = (np.dot(np.dot(WM,betaR ** 2), HM)) / hatSXR * self.XR
        
        if self.tfrepresentation == 'stft':
            mestR = slf.istft(
                hatMR,
                hopsize=self.stftParams['hopsize'],
                nfft=self.stftParams['NFT'],
                window=slf.sinebell(windowSizeInSamples),
                originalDataLen=None,)#self.originalDataLen) # / 4.0
        elif self.tfrepresentation in knownTransfos:
            self.mqt.transfo = hatMR
            mestR = self.mqt.invertTransform()
            del self.mqt.transfo
        
        hatMR = (np.dot(np.dot(WM,betaL ** 2), HM)) / hatSXL * self.XL
        
        if self.tfrepresentation == 'stft':
            mestL = slf.istft(
                hatMR, 
                hopsize=self.stftParams['hopsize'],
                nfft=self.stftParams['NFT'],
                window=slf.sinebell(windowSizeInSamples),
                originalDataLen=None,)#self.originalDataLen) # / 4.0
        elif self.tfrepresentation in knownTransfos:
            self.mqt.transfo = hatMR
            mestL = self.mqt.invertTransform()
            del self.mqt.transfo
        
        del hatMR
        
        mestR = np.array(np.round(mestR*self.scaleData), dtype=self.dataType)
        mestL = np.array(np.round(mestL*self.scaleData), dtype=self.dataType)
        wav.write(self.files['mus_output_file'][:-4] + suffix,
                  self.fs,
                  np.array([mestR,mestL]).T)
        
        del mestR, mestL
    
    def writeSeparatedSignalsWithUnvoice(self):
        """A wrapper to give a decent name to the function: simply
        calling self.writeSeparatedSignals with the
        '_VUIMM.wav' suffix.
        """
        self.writeSeparatedSignals(suffix='_VUIMM.wav')
    
    def checkChunkSize(self, maxFrames):
        """Computes the number of chunks of size maxFrames, and
        changes maxFrames in case it does not provide long enough
        chunks (especially the last chunk). 
        """
        totFrames = np.int32(self.computeNFrames())
        nChunks = totFrames / maxFrames + 1
        # checking size of last chunk, if "small", then making it
        # more even sized chunks
        if (totFrames-(nChunks-1)*maxFrames <
            self.stftParams['windowSizeInSamples'] /
            self.stftParams['hopsize'] ):
            print "Modifying the maxframes, such that chunks not too small"
            maxFrames = np.int(np.ceil(np.double(totFrames)/nChunks))
            nChunks = totFrames/maxFrames 
            print "The chunks are then maximum", maxFrames
            
        return totFrames, nChunks, maxFrames
