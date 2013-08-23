"""\
Description
-----------

FASST (Flexible Audio Source Separation Toolbox) class
    subclass it to obtain your own flavoured source separation model!

You can find more about the technique and how to use this module in the
provided documentation in `doc/` (`using the python package
<../description.html#using-the-python-package>`_)
    
Adapted from the Matlab toolbox available at:
http://bass-db.gforge.inria.fr/fasst/

Contact
-------
Jean-Louis Durrieu, EPFL-STI-IEL-LTS5
::

    jean DASH louis AT durrieu DOT ch

2012-2013
http://www.durrieu.ch

Copyright
---------
This software is distributed under the terms of the GNU Public 
(http://www.gnu.org/licenses/gpl.txt)

Reference
---------

"""

import numpy as np 
from numpy.testing import assert_array_almost_equal # FOR DEBUG/DEV
import warnings, os

import audioObject as ao
import demixTF as demix

import SeparateLeadStereo.SeparateLeadStereoTF as SLS

from sourcefilter.filter import generateHannBasis
from spatial.steering_vectors import gen_steer_vec_far_src_uniform_linear_array

import tftransforms.tft as tft # loads the possible transforms

import tools.signalTools as st
from tools.signalTools import inv_herm_mat_2d
from tools.nmf import NMF_decomp_init, NMF_decomposition

tftransforms = {
    'stftold': tft.TFTransform, # just making dummy, in FASST, not used
    'stft': tft.STFT,
    'mqt': tft.MinQTransfo,
    'minqt': tft.MinQTransfo,
    'nsgmqt': tft.NSGMinQT,
    'cqt': tft.CQTransfo}

eps = 1e-10
log_prior_small_cst = 1e-70
soundCelerity = 340. # m/s

########## Main classes for audio models ##########
class FASST(object):
    """**FASST**: Flexible Audio Source Separation Toolbox
    
    This is the superclass that implements the core functions for
    the framework for audio source separation as introduced in [Ozerov2012]_
    
        A. Ozerov, E. Vincent and F. Bimbot
        \"A General Flexible Framework for the Handling of Prior
        Information in Audio Source Separation,\" 
        IEEE Transactions on Audio, Speech and Signal Processing 20(4),
        pp. 1118-1133 (2012)                            
        Available: http://hal.inria.fr/hal-00626962/
    
    In order to use it, one should sub-class this class, and in particular
    define several elements that are assumed by the core functions for
    estimation and separation in this class, see below for a list.
    
    :param audio: the audio filename 
    :param transf:
        a string describing the desired Time-Frequency (TF) representation
    :param wlen: length of the analysis windows, mostly for STFT representation
    :param integer hopsize: the size of samples between two analysis frames
    :param integer iter_num: number of GEM iterations for parameter esitmation
    :param str sim_ann_opt: type of annealing strategy (`'ann'`. `'no_ann'`)
    :param list ann_PSD_lim:
        list of 2 elements, `ann_PSD_lim[0]` is the amount of added noise
        to the PSD at beginning, and `ann_PSD_lim[1]` is this amount at the
        end of the estimation.
    :param integer verbose:
        level of verbose: 0 for almost nothing, greater than 1 for debug
        messages
    :param double nmfUpdateCoeff:
        the exponent for the Non-Negative Matrix Factorization-type updates
        within the GEM iteration
    :param tffmin: minimum frequency for the TF representation 
    :param tffmax: maximal frequency
    :param tfWinFunc: window function (please provide a python function here)
    :param integer tfbpo:
        number of bins per octave, for Constant-Q-based representations
    :param double lambdaCorr:
        penalization term to control the correlation between the sources.
    
    Some important attributes of this class are:
    
    :var dict spat_comps:
        a dictionary containing the spatial component parameters and
        variables. In particular, for a given component numbered `spat_ind`,
        
    :var dict spec_comps:
        the spectral component parameters dictionary. 
    
    :var pyfasst.audioObject.AudioObject audioObject:
        the audio object that is to be processed.
        See :py:class:`pyfasst.audioObject.AudioObject` for details.
    
    :var sig_repr_params:
        Parameters for the computation of the signal TF representation. The
        keys in this dictionary are:
        
            `'transf'` - the type of TF representation

            `'wlen'` - the window length in samples

            `'fsize'` - the size of the Fourier transform (in samples)

            `'hopsize'` - the hop-size in samples between two consecutive frames

            `'tffmin'`, `'tffmax'`, `'tfbpo'`, `'tfWinFunc'`, `'hopfactor'` -
            variables related to specific TF representations.
            
    :var pyfasst.tftransforms.tft.TFTransform tft:
        The object that implements the TF transform.
        See :py:class:`pyfasst.tftransforms.tft.TFTransform`

    :var numpy.ndarray Cx:
        The computed transform, after :py:meth:`FASST.omp_transf_Cx` has been
        called. For memory management, as of 20130820, :py:attr:`FASST.Cx`,
        for a given frame and given frequency, is supposed to be Hermitian:
        only the upper diagonal is therefore kept. 
        
    For examples, see also:

    * :py:class:`MultiChanNMFInst_FASST`
    * :py:class:`MultiChanNMFConv`
    * :py:class:`MultiChanHMM`
    * :py:class:`multiChanSourceF0Filter`
    * :py:class:`multichanLead`
    
    """
    # for now only stft:
    implemented_transf = ['stft','stftold', 'mqt', 'minqt', 'cqt']
    implemented_annealing = ['ann', 'no_ann', ]
    
    def __init__(self,
                 audio,
                 transf='stft',
                 wlen=2048,
                 hopsize=512,
                 iter_num=50,
                 sim_ann_opt='ann',
                 ann_PSD_lim=[None, None],
                 verbose=0,
                 nmfUpdateCoeff=1.,
                 tffmin=25,
                 tffmax=18000,
                 tfWinFunc=None,
                 tfbpo=48,
                 lambdaCorr=0.):
        """**FASST**: Flexible Audio Source Separation Toolbox
        
        """
        
        self.verbose = verbose
        self.nmfUpdateCoeff = nmfUpdateCoeff
        
        if isinstance(audio, ao.AudioObject):
            self.audioObject = audio
        elif isinstance(audio, str) or isinstance(audio, unicode):
            self.audioObject = ao.AudioObject(filename=audio)
        else:
            raise AttributeError("The provided audio parameter is"+
                                 "not a supported format.")
        
        # parameters to compute the signal representation:
        self.sig_repr_params = {}
        self.sig_repr_params['transf'] = transf.lower()  # transformation type
        self.sig_repr_params['wlen'] = ao.nextpow2(wlen)      # window length
        self.sig_repr_params['fsize'] = ao.nextpow2(wlen) # Fourier length
        self.sig_repr_params['hopsize'] = hopsize
        self.sig_repr_params['tffmin'] = tffmin
        self.sig_repr_params['tffmax'] = tffmax
        self.sig_repr_params['tfbpo'] = tfbpo
        self.sig_repr_params['tfWinFunc'] = tfWinFunc
        self.sig_repr_params['hopfactor'] = (
            1. * hopsize / self.sig_repr_params['wlen'])
        if self.sig_repr_params['transf'] not in self.implemented_transf \
               or self.sig_repr_params['transf'] not in tftransforms:
            raise NotImplementedError(self.sig_repr_params['transf']
                                      + " not yet implemented.")
        elif self.sig_repr_params['transf'] != 'stftold':
            self.tft = tftransforms[self.sig_repr_params['transf']](
                fmin=tffmin,
                fmax=tffmax,
                bins=tfbpo,
                fs=self.audioObject.samplerate,
                perfRast=1,
                linFTLen=self.sig_repr_params['fsize'],
                atomHopFactor=self.sig_repr_params['hopfactor'],
                )
        elif self.sig_repr_params['transf'] == 'stftold':
            self.tft = tftransforms['stft'](
                fmin=tffmin,
                fmax=tffmax,
                bins=tfbpo,
                fs=self.audioObject.samplerate,
                perfRast=1,
                linFTLen=self.sig_repr_params['fsize'],
                atomHopFactor=self.sig_repr_params['hopfactor'],
                )
            
            
        # demix parameters:
        self.demixParams = {
            'tffmin': tffmin, 'tffmax': tffmax,
            'tfbpo': tfbpo,
            'tfrepresentation': transf.lower(), # 'stft', #transf.lower()
            'wlen': self.sig_repr_params['wlen'],
            'hopsize': self.sig_repr_params['wlen']/2,
            'neighbors': 20,
            'winFunc': tfWinFunc,
            }
        
        # noise parameters
        self.noise = {}
        self.noise['PSD'] = np.zeros(self.sig_repr_params['fsize']/2+1)
        self.noise['sim_ann_opt'] = sim_ann_opt
        self.noise['ann_PSD_lim'] = ann_PSD_lim
        
        self.spat_comps = {}
        self.spec_comps = {}
        
        self.iter_num = iter_num
        self.lambdaCorr = lambdaCorr
    
    def comp_transf_Cx(self):
        """Computes the signal representation, according
        to the provided signal representation flag, in
        :py:attr:`FASST.sig_repr_params['transf']`
        """
        if not hasattr(self.audioObject, '_data'):
            self.audioObject._read()
        
        if self.sig_repr_params['transf'] not in self.implemented_transf:
            raise ValueError(self.sig_repr_params['transf'] +
                             " not implemented - yet?")
        
        if self.verbose:
            print ("Computing the chosen signal representation:",
                   self.sig_repr_params['transf'] )
        
        nc = self.audioObject.channels
        Xchan = []
        for n in range(nc):
            if self.sig_repr_params['transf'] == 'stftold':
                X, freqs, times = ao.stft(
                    self.audioObject.data[:,n],
                    window=np.hanning(self.sig_repr_params['wlen']),
                    hopsize=self.sig_repr_params['hopsize'],
                    nfft=self.sig_repr_params['fsize'],
                    fs=self.audioObject.samplerate
                    )
            else:
                self.tft.computeTransform(self.audioObject.data[:,n],)
                X = self.tft.transfo
            Xchan.append(X)
            
        if self.verbose>1:
            print X.shape
        
        self.nbFreqsSigRepr, self.nbFramesSigRepr = X.shape
        ##assert self.nbFreqsSigRepr == self.tft.freqbins
        del X
        del self.audioObject.data
        
        if nc == 1:
            self.Cx = np.abs(Xchan[0])**2
        else:
            self.Cx = np.zeros([nc * (nc + 1) / 2,
                                self.nbFreqsSigRepr,
                                self.nbFramesSigRepr],
                               dtype=complex)
            for n1 in range(nc):
                for n2 in range(n1, nc):
                    # note : we keep only upper diagonal of Cx
                    # lower diagonal is conjugate of upper one.
                    n = n2 - n1 + np.sum(np.arange(nc, nc-n1, -1))
                    self.Cx[n] = Xchan[n1] * np.conj(Xchan[n2])
        
        if self.noise['ann_PSD_lim'][0] is None or \
               self.noise['ann_PSD_lim'][1] is None:
            mix_psd = 0
            # average power, for each frequency band, across the frames
            if nc == 1:
                mix_psd += np.mean(self.Cx, axis=1)
            else:
                for n1 in range(nc):
                    n = np.sum(np.arange(nc, nc-n1, -1)) # n2 = n1
                    mix_psd += np.mean(self.Cx[n], axis=1)
                    
            if self.verbose>1:
                print "mix_psd", mix_psd
            mix_psd /= nc
            if self.verbose>1:
                print "mix_psd/nc", mix_psd
            if self.noise['ann_PSD_lim'][0] is None:
                self.noise['ann_PSD_lim'][0] = np.real(mix_psd) / 100.
            if self.noise['ann_PSD_lim'][1] is None:
                self.noise['ann_PSD_lim'][1] = np.real(mix_psd) / 10000.
        if self.noise['sim_ann_opt'] in ('ann'):
            self.noise['PSD'] = self.noise['ann_PSD_lim'][0]
        
        # useless for the rest of computations:
        del Xchan
    
    def estim_param_a_post_model(self,):
        """Estimates the `a posteriori` model for the provided
        audio signal. In particular, this runs self.iter_num times
        the Generalized Expectation-Maximisation algorithm
        :py:meth:`FASST.GEM_iteration`, to
        update the various parameters of the model, so as to
        maximize the likelihood of the data given these parameters.
        
        From these parameters, the posterior expectation of the
        \"hidden\" or latent variables (here the spatial and spectral
        components) can be computed, leading to the estimation of the
        separated underlying sources.

        Consider using :py:meth:`FASST.separate_spat_comps` or
        :py:meth:`FASST.separate_spatial_filter_comp` to obtain the separated time
        series, once the parameters have been estimated.

        :returns:
            `logliks`: The log-likelihoods as computed after each GEM iteration.
        
        """
        
        logliks = np.ones(self.iter_num)
        
        # TODO: move this back in __init__, and remove from subclasses...
        if self.noise['sim_ann_opt'] in ['ann', ]:
            self.noise['PSD'] = self.noise['ann_PSD_lim'][0]
        elif self.noise['sim_ann_opt'] is 'no_ann':
            self.noise['PSD'] = self.noise['ann_PSD_lim'][1]
        else:
            warnings.warn("To add noise to the signal, provide the "+
                          "sim_ann_opt from any of 'ann', "+
                          "'no_ann' or 'ann_ns_inj' ")
            
        for i in range(self.iter_num):
            if self.verbose:
                print "Iteration", i+1, "on", self.iter_num
            # adding the noise psd if required:
            if self.noise['sim_ann_opt'] in ['ann', 'ann_ns_inj']:
                self.noise['PSD'] = (
                    (np.sqrt(self.noise['ann_PSD_lim'][0]) *
                     (self.iter_num - i) +
                     np.sqrt(self.noise['ann_PSD_lim'][1]) * i) /
                    self.iter_num) ** 2
                
            # running the GEM iteration:
            logliks[i] = self.GEM_iteration()
            if self.verbose:
                print "    log-likelihood:", logliks[i]
                if i>0:
                    print "        improvement:", logliks[i]-logliks[i-1]

        return logliks
    
    def GEM_iteration(self,):
        """GEM iteration: one iteration of the Generalized Expectation-
        Maximization algorithm to update the various parameters whose
        :py:attr:`FASST.spec_comp[spec_ind]['frdm_prior']` is set to ``'free'``.
        
        :returns:
            `loglik` (double): the log-likelihood of the data,
            given the updated parameters
        
        """
        if self.audioObject.channels==2:
            spat_comp_powers, mix_matrix, rank_part_ind = (
                self.retrieve_subsrc_params())
            
            # compute the sufficient statistics
            hat_Rxx, hat_Rxs, hat_Rss, hat_Ws, loglik = (
                self.compute_suff_stat(spat_comp_powers, mix_matrix))
            
            # update the mixing matrix
            self.update_mix_matrix(hat_Rxs, hat_Rss, mix_matrix, rank_part_ind)
            
            # from sub-sources to sources
            # (as given by the different spatial comps)
            #     had better have shape = [nbSpatComps,F,N]
            hat_W = np.zeros([len(rank_part_ind),
                              self.nbFreqsSigRepr,
                              self.nbFramesSigRepr])
            if self.verbose > 1:
                print "rank_part_in", rank_part_ind
            for w in range(len(rank_part_ind)):
                hat_W[w] = np.mean(hat_Ws[rank_part_ind[w]], axis=0)
                
            del spat_comp_powers, mix_matrix, rank_part_ind
            del hat_Rxx, hat_Rxs, hat_Rss, hat_Ws
        else:
            raise AttributeError("Nb channels "+str(self.audioObject.channels)+
                                 " not implemented yet")
        
        # update the spectral parameters
        self.update_spectral_components(hat_W)
        
        # normalize parameters
        self.renormalize_parameters()
        
        return loglik
    
    def comp_spat_comp_power(self, spat_comp_ind,
                             spec_comp_ind=[], factor_ind=[]):
        """Matlab FASST Toolbox help::
        
        % V = comp_spat_comp_power(mix_str, spat_comp_ind,                  
        %                          spec_comp_ind, factor_ind);            
        %
        % compute spatial component power
        %
        %
        % input
        % -----
        %
        % mix_str           : mixture structure
        % spat_comp_ind     : spatial component index
        % spec_comp_ind     : (opt) factor index (def = [], use all components)
        % factor_ind         : (opt) factor index (def = [], use all factors)
        % 
        %
        % output
        % ------
        %
        % V                 : (F x N) spatial component power

        :param integer spat_comp_ind:
            index of the spatial component 
        :param list spec_comp_ind:
            list of indices for the spectral components whose spatial component
            corresponds to the provided `spat_comp_ind`
        :param list factor_ind:
            list of indices of factors to be included. 

        Note: thanks to object-oriented programming, no need to provide the
        structure containing all the parameters, the instance has direct access
        to them.

        Note2: this may not completely work because the factor_ind should
        actually also depend on the index of the spectral component. TODO?
        """
        V = np.zeros([self.nbFreqsSigRepr, self.nbFramesSigRepr])
        if len(spec_comp_ind):
            spec_comp_ind_arr = spec_comp_ind
        else:
            spec_comp_ind_arr = self.spec_comps.keys()
        
        for k in spec_comp_ind_arr:
            if spat_comp_ind == self.spec_comps[k]['spat_comp_ind']:
                V_comp = np.ones([self.nbFreqsSigRepr, self.nbFramesSigRepr])
                if len(factor_ind):
                    factors_ind_arr = factor_ind
                else:
                    factors_ind_arr = self.spec_comps[k]['factor'].keys()
                    
                for f in factors_ind_arr:
                    factor = self.spec_comps[k]['factor'][f]
                    W = np.dot(factor['FB'], factor['FW'])
                    if len(factor['TB']):
                        H = np.dot(factor['TW'],factor['TB'])
                    else:
                        H = factor['TW']
                    V_comp *= np.dot(W, H)
                    del W
                    del H
                    
                V += V_comp
                del V_comp
                del factor
        
        return V
    
    def comp_spat_cmps_powers(self, spat_comp_ind,
                              spec_comp_ind=[], factor_ind=[]):
        """Compute the sum of the spectral powers corresponding to the
        spatial components as provided in the list `spat_comp_ind`
        
        NB: because this does not take into account the mixing process,
        the resulting power does not, in general, correspond to the
        the observed signal's parameterized spectral power.
        """
        V = 0
        for i in spat_comp_ind:
            V += self.comp_spat_comp_power(spat_comp_ind=i)
        return V
    
    def retrieve_subsrc_params(self,):
        """\
        Computes the various quantities necessary for the estimation of the
        main parameters:
        
        **Outputs**
        
        :returns:
        
         1. :py:attr:`spat_comp_powers` - 
            (`total_spat_rank` x `nbFreqsSigRepr` x `nbFramesSigRepr`) ndarray
            the spatial component power spectra. Note that total_spat_rank
            is the sum of all the spatial ranks for all the sources.
            
         2. :py:attr:`mix_matrix` -
            (`total_spat_rank` x `nchannels` x `nbFreqsSigRepr`)
            :py:class:`ndarray` the mixing matrices for each source
            
         3. :py:attr:`rank_part_ind` - 
            dictionary: each key is one source, and the values are the indices
            in `spat_comp_powers` and `mix_matrix` that correspond to that source.
            If the spatial rank of source `j` is 2, then its spectra will appear
            twice in `spat_comp_powers`, with mixing parameters (potentially
            different one from the other) appearing in two sub-matrices of
            :py:attr:`mix_matrix`.
            
        """
        K = len(self.spat_comps)
        rank_total = 0
        rank_part_ind = {}
        for j in range(K):
            # this is the ranks
            if self.spat_comps[j]['mix_type'] == 'inst':
                rank = self.spat_comps[j]['params'].shape[1]
            else:
                rank = self.spat_comps[j]['params'].shape[0]
            if self.verbose>1:
                print "    Rank of spatial source %d" %j +\
                      " is %d" %rank
            rank_part_ind[j] = (
                rank_total +
                np.arange(rank))
            rank_total += rank
        
        spat_comp_powers = np.zeros([rank_total,
                                     self.nbFreqsSigRepr,
                                     self.nbFramesSigRepr])
        
        mix_matrix = np.zeros([rank_total,
                               self.audioObject.channels,
                               self.nbFreqsSigRepr], dtype=complex)
        for j, spat_comp in self.spat_comps.items():
            spat_comp_j = self.comp_spat_comp_power(spat_comp_ind=j)
            for r in rank_part_ind[j]:
                spat_comp_powers[r] = spat_comp_j
            if spat_comp['mix_type'] == 'inst':
                for f in range(self.nbFreqsSigRepr):
                    #print rank_part_ind[j]
                    #print spat_comp['params'].shape
                    #print mix_matrix[rank_part_ind[j],:,f].shape
                    mix_matrix[rank_part_ind[j],:,f] = spat_comp['params'].T
            else:
                mix_matrix[rank_part_ind[j]] = spat_comp['params']
                
        return spat_comp_powers, mix_matrix, rank_part_ind
    
    def compute_suff_stat(self, spat_comp_powers, mix_matrix):
        """\
        Computes the sufficient statistics, used to update the parameters.
        
        **Inputs:**
        
        :param numpy.ndarray spat_comp_powers:
            (`total_spat_rank`x`nbFreqsSigRepr`x`nbFramesSigRepr`) `ndarray`.
            the estimated power spectra for the spatial components, as computed
            by :py:meth:`FASST.retrieve_subsrc_params`. 
        :param numpy.ndarray mix_matrix:
            (`total_spat_rank` x `nchannels` x `nbFreqsSigRepr`) `ndarray`.
            the mixing parameters, as a rank x n_channels x n_freqs `ndarray`.
            Computed from :py:meth:`FASST.retrieve_subsrc_params`
        
        **Outputs:**

        :returns:
            1. `hat_Rxx`
            2. `hat_Rxs`
            3. `hat_Rss`
            4. `hat_Ws`
            5. `loglik`
        
        """
        if self.audioObject.channels != 2:
            raise ValueError("Nb channels not supported:"+
                             str(self.audioObject.channels))
        
        if self.verbose: print "    Computing sufficient statistics"
        nbspatcomp = spat_comp_powers.shape[0]
        
        # CAUTION! non-initialized arrays !
        sigma_x_diag = np.empty([2,
                                 self.nbFreqsSigRepr,
                                 self.nbFramesSigRepr])
        #sigma_x_off = np.zeros([self.nbFreqsSigRepr,
        #                        self.nbFramesSigRepr], dtype=complex)
        
        # setting the first element with spat_comp 0:
        r = 0
        sigma_x_diag[0] = (
            np.vstack(np.abs(mix_matrix[r][0])**2) *
            spat_comp_powers[r]
            )
        sigma_x_diag[1] = (
            np.vstack(np.abs(mix_matrix[r][1])**2) *
            spat_comp_powers[r]
            )
        sigma_x_off = (
            np.vstack(mix_matrix[r][0] *
                      np.conj(mix_matrix[r][1])) *
            spat_comp_powers[r]
            )
        
        for n in range(2):
            sigma_x_diag[n] += np.vstack(self.noise['PSD'])
            # noise PSD should be of size nbFreqs
        
        for r in range(1, nbspatcomp):
            sigma_x_diag[0] += (
                np.vstack(np.abs(mix_matrix[r][0])**2) *
                spat_comp_powers[r]
                )
            sigma_x_diag[1] += (
                np.vstack(np.abs(mix_matrix[r][1])**2) *
                spat_comp_powers[r]
                )
            sigma_x_off += (
                np.vstack(mix_matrix[r][0] *
                          np.conj(mix_matrix[r][1])) *
                spat_comp_powers[r]
                )
            
        inv_sigma_x_diag, inv_sigma_x_off, det_sigma_x = (
            inv_herm_mat_2d(sigma_x_diag, sigma_x_off,
                            verbose=self.verbose))
        del sigma_x_diag, sigma_x_off
        
        # compute log likelihood
        loglik = - np.mean(np.log(det_sigma_x * np.pi) +
                           inv_sigma_x_diag[0] * self.Cx[0] +
                           inv_sigma_x_diag[1] * self.Cx[2] +
                           2. * np.real(inv_sigma_x_off * np.conj(self.Cx[1]))
                           )
        # compute expectations of Rss and Ws sufficient statistics
        Gs = np.empty((2, nbspatcomp,
                       self.nbFreqsSigRepr,
                       self.nbFramesSigRepr), dtype=np.complex) # {}
        # one for each channel (stereo, here)
        #Gs[0] = {}
        #Gs[1] = {}
        for r in range(nbspatcomp):
            Gs[0, r] = (
                (np.vstack(np.conj(mix_matrix[r][0])) * inv_sigma_x_diag[0] +
                 np.vstack(np.conj(mix_matrix[r][1])) *
                 np.conj(inv_sigma_x_off)) *
                spat_comp_powers[r]
                )
            
            Gs[1, r] = (
                (np.vstack(np.conj(mix_matrix[r][0])) * inv_sigma_x_off +
                 np.vstack(np.conj(mix_matrix[r][1])) * inv_sigma_x_diag[1]) *
                spat_comp_powers[r]
                )

        # the following quantities are assigned later, so
        # an empty allocation should do.
        hat_Rss = np.empty([self.nbFreqsSigRepr,
                            nbspatcomp,
                            nbspatcomp],
                           dtype=complex)
        hat_Ws = np.empty([nbspatcomp,
                           self.nbFreqsSigRepr,
                           self.nbFramesSigRepr])
        hatRssLoc1 = np.empty_like(self.Cx[0])
        hatRssLoc2 = np.empty_like(self.Cx[0])
        hatRssLoc3 = np.empty_like(self.Cx[0])
        for r1 in range(nbspatcomp):
            for r2 in range(nbspatcomp):
                # TODO: could probably factor a bit more the following formula:
                hatRssLoc1[:] = np.copy(self.Cx[0])
                hatRssLoc1 *= np.conj(Gs[0,r2])
                hatRssLoc1 += (np.conj(Gs[1,r2]) * self.Cx[1])
                hatRssLoc1 *= Gs[0,r1]
                
                hatRssLoc2[:] = np.copy(self.Cx[2])
                hatRssLoc2 *= np.conj(Gs[1,r2])
                hatRssLoc2 += (np.conj(Gs[0,r2] * self.Cx[1]))
                hatRssLoc2 *= Gs[1,r1]
                
                hatRssLoc3[:] = np.copy(Gs[0,r1])
                hatRssLoc3 *= np.vstack(mix_matrix[r2,0])
                hatRssLoc3 += (Gs[1,r1] * np.vstack(mix_matrix[r2,1]))
                hatRssLoc3 *= spat_comp_powers[r2]
                
                hatRssLoc1 += hatRssLoc2
                hatRssLoc1 -= hatRssLoc3
                
                #hatRssLoc = (Gs[0][r1] * np.conj(Gs[0][r2]) * self.Cx[0] +
                #             Gs[1][r1] * np.conj(Gs[1][r2]) * self.Cx[2] +
                #             Gs[0][r1] * np.conj(Gs[1][r2]) * self.Cx[1] +
                #             Gs[1][r1]*np.conj(Gs[0][r2])*np.conj(self.Cx[1])-
                #             (Gs[0][r1] * np.vstack(mix_matrix[r2][0]) +
                #              Gs[1][r1] * np.vstack(mix_matrix[r2][1]))
                #             * spat_comp_powers[r2]
                #             )
                if r1 == r2:
                    hatRssLoc1 += spat_comp_powers[r1]
                    hat_Ws[r1] = np.abs(np.real(hatRssLoc1))
                    
                hat_Rss[:,r1,r2] = np.mean(hatRssLoc1, axis=1)
                
        # To assure hermitian symmetry:
        for f in range(self.nbFreqsSigRepr):
            if self.verbose>10: # DEBUG
                assert_array_almost_equal(
                    hat_Rss[f],
                    (hat_Rss[f] + np.conj(hat_Rss[f]).T) / 2.)
                
            hat_Rss[f] = (hat_Rss[f] + np.conj(hat_Rss[f]).T) / 2.
            
        # Expectations of Rxs sufficient statistics
        hat_Rxs = np.empty([self.nbFreqsSigRepr,
                            2,
                            nbspatcomp],
                           dtype=complex)
        for r in range(nbspatcomp):
            hat_Rxs[:,0,r] = (
                np.mean(np.conj(Gs[0][r]) * self.Cx[0] +
                        np.conj(Gs[1][r]) * self.Cx[1], axis=1)
                )
            hat_Rxs[:,1,r] = (
                np.mean(np.conj(Gs[0][r]) * np.conj(self.Cx[1]) +
                        np.conj(Gs[1][r]) * self.Cx[2], axis=1)
                )
        
        del Gs
        
        # at last Rxx sufficient statistics:
        hat_Rxx = np.mean(self.Cx, axis=-1)
        # recommendation, use logarithm:
        # hat_Rxx[]
        
        return hat_Rxx, hat_Rxs, hat_Rss, hat_Ws, loglik
    
    def update_mix_matrix(self,hat_Rxs, hat_Rss, mix_matrix, rank_part_ind):
        """Update the mixing parameters, according to the current estimated
        spectral component parameters.
        
        :param hat_Rxs:
            (`nbFreqsSigRepr` x 2 x `nbspatcomp`) `ndarray`.
            Estimated intercorrelation between the observation and the
            sources, for each frequency bin of the TF representation.
        :param hat_Rss:
            (`nbFreqsSigRepr` x `nbspatcomp` x `nbspatcomp`) `ndarray`.
            Estimated auto-correlation matrix for each frequency bin, 
        :param mix_matrix:
            (`total_spat_rank` x `nchannels` x `nbFreqsSigRepr`) `ndarray`.
            the mixing parameters, as a rank x n_channels x n_freqs `ndarray`.
        :param rank_part_ind:
            a dictionary giving, for each spectral component, which spatial
            component should be used.
        
        The input parameters should be computed by :py:meth:`compute_suff_stat`
        and :py:meth:`retrieve_subsrc_params`, done automatically in
        :py:meth:`GEM_iteration`.
        
        """
        # deriving which components have which updating rule:
        upd_inst_ind = []
        upd_inst_other_ind = []
        upd_conv_ind = []
        upd_conv_other_ind = []
        for j, spat_comp_j in self.spat_comps.items():
            if spat_comp_j['frdm_prior'] == 'free' and \
                   spat_comp_j['mix_type'] == 'inst':
                upd_inst_ind.extend(rank_part_ind[j])
            else:
                upd_inst_other_ind.extend(rank_part_ind[j])
            
            if spat_comp_j['frdm_prior'] == 'free' and \
                   spat_comp_j['mix_type'] == 'conv':
                upd_conv_ind.extend(rank_part_ind[j])
            else:
                upd_conv_other_ind.extend(rank_part_ind[j])
        
        # update linear instantaneous coefficients:
        K_inst = len(upd_inst_ind)
        
        if len(upd_inst_ind):
            if self.verbose:
                print "    Updating mixing matrix, instantaneous sources"
            #hat_Rxs_bis = np.zeros([self.nbFreqsSigRepr,
            #                        2,
            #                        K_inst])
            hat_Rxs_bis = hat_Rxs[:,:,upd_inst_ind]
            if len(upd_inst_other_ind):
                for f in range(self.nbFreqsSigRepr):
                    hat_Rxs_bis[f] -= (
                        np.dot(mix_matrix[upd_inst_other_ind,:,f].T,
                               hat_Rss[f][np.vstack(upd_inst_other_ind),
                                                        upd_inst_ind]))
                    # hat_Rss[f][upd_inst_other_ind][:,upd_inst_ind])
            hat_Rxs_bis = np.real(np.mean(hat_Rxs_bis, axis=0))
            rm_hat_Rss = np.real(np.mean(hat_Rss[:,np.vstack(upd_inst_ind),
                                                 upd_inst_ind], axis=0))
            
            # in ozerov's code:
            ##mix_matrix_inst = np.dot(hat_Rxs_bis, np.linalg.inv(rm_hat_Rss))
            mix_matrix_inst = np.linalg.solve(rm_hat_Rss.T, hat_Rxs_bis.T)
            #                                   sym_pos=True).T
            if self.verbose>1:
                print "mix_matrix", mix_matrix
                print "mix_matrix_inst", mix_matrix_inst
                print "mix_matrix_inst.shape",mix_matrix_inst.shape
                print mix_matrix.shape, \
                      mix_matrix[upd_inst_ind].shape
            for f in range(self.nbFreqsSigRepr):
                mix_matrix[upd_inst_ind,:,f] = mix_matrix_inst
                
            del mix_matrix_inst
            
        # update convolutive coefficients: 
        if len(upd_conv_ind):
            if self.verbose:
                print "    Updating mixing matrix, convolutive sources"
            hat_Rxs_bis = hat_Rxs[:,:,upd_conv_ind]
            if len(upd_conv_other_ind):
                for f in range(self.nbFreqsSigRepr):
                    hat_Rxs_bis[f] -= (
                        np.dot(mix_matrix[upd_conv_other_ind,:,f].T,
                               hat_Rss[f][np.vstack(upd_conv_other_ind),
                                          upd_conv_ind]))
            for f in range(self.nbFreqsSigRepr):
                try:
                    mix_matrix[upd_conv_ind,:,f] = (
                        np.linalg.solve(hat_Rss[f].T, hat_Rxs_bis[f].T))
                except np.linalg.linalg.LinAlgError:
                    print "hat_Rss[f]:", hat_Rss[f]
                    print "hat_Rxs_bis[f]:", hat_Rxs_bis[f]
                    raise np.linalg.LinAlgError('Singular Matrix')
                except:
                    raise # re-raise the exception if that was not linalgerror...
                    
            ## smoothing
            ##for n in upd_conv_ind:
            ##    for nc in range(self.audioObject.channels):
            ##        smoothAbsMix = (
            ##            st.medianFilter(np.abs(mix_matrix[n,nc,:]),
            ##                            length=self.nbFreqsSigRepr/200)
            ##            )
            ##        mix_matrix[n,nc,:] = (
            ##            smoothAbsMix *
            ##            np.exp(1j * np.angle(mix_matrix[n,nc,:]))
            ##            )
            
        # update the matrix in the component parameters:
        for k, spat_comp_k in self.spat_comps.items():
            if spat_comp_k['frdm_prior'] == 'free':
                if spat_comp_k['mix_type'] == 'inst':
                    spat_comp_k['params'] = (
                        np.mean(mix_matrix[rank_part_ind[k]], axis=2)).T
                else:
                    spat_comp_k['params'] = (
                        mix_matrix[rank_part_ind[k]])
                    
        # mix_matrix should have changed outside this method... TBC
        # should we normalize here?
        ##self.renormalize_parameters()
        
    def separate_spatial_filter_comp(self,
                                     dir_results=None,
                                     suffix=None):
        """separate_spatial_filter_comp
        
        Separates the sources using only the estimated spatial
        filter (i.e. the mixing parameters in self.spat_comps[j]['params'])
        
        In particular, we consider here the corresponding MVDR filter,
        as exposed in [Maazaoui2011]_.
        
        per channel, the filter steering vector, source p:
        
        .. math::
        
            b(f,p) = \\frac{R_{aa,f}^{-1} a(f,p)}{a^{H}(f,p) R_{aa,f}^{-1} a(f,p)}
            
        with
        
        .. math::
        
            R_{aa,f} = \\sum_q a(f,q) a^{H}(f,q)
            
        It corresponds also to the given model in FASST, assuming that all the
        spectral powers are equal across all sources. Here, by computing the Wiener
        Gain WG to get the images, we actually have
        
        .. math::
        
            b(f,p) a(f,p)^H
            
        and the denominator therefore is the trace of the \"numerator\".
        
        .. [Maazaoui2011] Maazaoui, M.; Grenier, Y. and Abed-Meraim, K.
           Blind Source Separation for Robot Audition using
           Fixed Beamforming with HRTFs, 
           in proc. of INTERSPEECH, 2011.
        
        """
        # grouping the indices by spatial component
        spec_comp_ind = {}
        for spat_ind in range(len(self.spat_comps)):
            spec_comp_ind[spat_ind] = []
        for spec_ind, spec_comp in self.spec_comps.items():
            # add the spec comp index to the corresponding spatial comp:
            spec_comp_ind[spec_comp['spat_comp_ind']].append(spec_ind)
            
        # copying from separate_spec_comps -  could modify that one later...
        if dir_results is None:
            dir_results = (
                '/'.join(
                self.audioObject.filename.split('/')[:-1])
                )
            if self.verbose:
                print "Writing to same directory as input file: " + dir_results
        
        nc = self.audioObject.channels
        if nc != 2:
            raise NotImplementedError()
        
        nbSources = len(spec_comp_ind)
        sigma_comps_diag = np.zeros([nbSources, 2,
                                    self.nbFreqsSigRepr,
                                    self.nbFramesSigRepr])
        sigma_comps_off = np.zeros([nbSources,
                                   self.nbFreqsSigRepr,
                                   self.nbFramesSigRepr], dtype=np.complex)
        
        # computing individual spatial variance
        R_diag0 = np.zeros([nbSources, self.nbFreqsSigRepr])
        R_diag1 = np.zeros([nbSources, self.nbFreqsSigRepr])
        R_off  = np.zeros([nbSources, self.nbFreqsSigRepr], dtype=np.complex)
        
        for n in range(nbSources):
            if self.spat_comps[n]['mix_type'] == 'inst':
                raise NotImplementedError('Mixing params not convolutive...')
                mix_coefficients = self.spat_comps[n]['params'].T
                # mix_coefficients.shape should be (rank, nchannels)
            elif self.spat_comps[n]['mix_type'] == 'conv':
                mix_coefficients = self.spat_comps[n]['params']
                # mix_coefficients.shape should be (rank, nchannels, freq)
            
            # R_diag = np.zeros(2, self.nbFreqsSigRepr)
            R_diag0[n] = np.atleast_1d(
                (np.abs(mix_coefficients[:, 0])**2).sum(axis=0))
            R_diag1[n] = np.atleast_1d(
                        (np.abs(mix_coefficients[:, 1])**2).sum(axis=0))
            # element at (1,2): 
            R_off[n] = np.atleast_1d((
                mix_coefficients[:, 0] *
                np.conj(mix_coefficients[:, 1])).sum(axis=0))
            
        Raa_00 = np.mean(R_diag0, axis=0)
        Raa_11 = np.mean(R_diag1, axis=0)
        Raa_01 = np.mean(R_off, axis=0)
        inv_Raa_diag, inv_Raa_off, det_mat = inv_herm_mat_2d(
            [Raa_00, Raa_11],
            Raa_01, verbose=self.verbose)
        
        if not hasattr(self, 'files'):
            self.files = {}
            
        self.files['spatial'] = []
        
        fileroot = self.audioObject.filename.split('/')[-1][:-4]
        for n in range(nbSources):
            WG = self.compute_Wiener_gain_2d(
                    [R_diag0[n], R_diag1[n]],
                    R_off[n],
                    inv_Raa_diag,
                    inv_Raa_off,
                timeInvariant=True)
            normalization = np.real(WG[0,0] + WG[1,1])
            WG /= [[normalization]]
            if self.sig_repr_params['transf'] is 'stftold':
                # compute the stft/istft
                ndata = ao.filter_stft(
                    self.audioObject.data, WG, analysisWindow=None,
                    synthWindow=np.hanning(self.sig_repr_params['wlen']),
                    hopsize=self.sig_repr_params['hopsize'],
                    nfft=self.sig_repr_params['fsize'],
                    fs=self.audioObject.samplerate)
            else:
                #raise NotImplementedError("TODO")
                X = []
                for chan1 in range(nc):
                    self.tft.computeTransform(
                        self.audioObject.data[:,chan1])
                    X.append(self.tft.transfo)
                ndata = []
                if WG.ndim == 3:
                    for chan1 in range(nc):
                        self.tft.transfo = np.zeros([self.nbFreqsSigRepr,
                                                     self.nbFramesSigRepr],
                                                    dtype=np.complex)
                        for chan2 in range(nc):
                            self.tft.transfo += (
                                np.vstack(WG[chan1, chan2])
                                * X[chan2])
                        ndata.append(self.tft.invertTransform())
                        del self.tft.transfo
                elif WG.ndim == 4:
                    for chan1 in range(nc):
                        self.tft.transfo = np.zeros([self.nbFreqsSigRepr,
                                                     self.nbFramesSigRepr],
                                                    dtype=np.complex)
                        for chan2 in range(nc):
                            self.tft.transfo += (
                                WG[chan1, chan2]
                                * X[chan2])
                        ndata.append(self.tft.invertTransform())
                        del self.tft.transfo
                        
                ndata = np.array(ndata).T
                
            _suffix = '_spatial'
            if suffix is not None and n in suffix:
                _suffix += '_' + suffix[n]
            outAudioName = (
                dir_results + '/' + fileroot + '_' + str(n) + 
                '-' + str(nbSources) + _suffix + '.wav')
            self.files['spatial'].append(outAudioName)
            outAudioObj = ao.AudioObject(filename=outAudioName,
                                         mode='w')
            outAudioObj._data = np.int16(
                ndata[:self.audioObject.nframes,:] *
                self.audioObject._maxdata)#(2**15))
            outAudioObj._maxdata = 1
            outAudioObj._encoding = 'pcm16'
            outAudioObj.samplerate = self.audioObject.samplerate
            outAudioObj._write()
        
    def separate_spat_comps(self,
                            dir_results=None,
                            suffix=None):
        """separate_spat_comps
        
        This separates the sources for each spatial component.

        :param dir_results:
            provide the (existing) folder where to write the results
        :param dict suffix:
            a dictionary containing the labels for each source. If None,
            then no suffix is appended to the file names and the files
            are simply numbered `XXX_nbComps`. 
        
        """
        spec_comp_ind = {}
        for spat_ind in range(len(self.spat_comps)):
            spec_comp_ind[spat_ind] = []
        for spec_ind, spec_comp in self.spec_comps.items():
            spec_comp_ind[spec_comp['spat_comp_ind']].append(spec_ind)
            
        self.separate_comps(dir_results=dir_results,
                            spec_comp_ind=spec_comp_ind,
                            suffix=suffix)
    
    def separate_comps(self,
                       dir_results=None,
                       spec_comp_ind=None,
                       suffix=None):
        """separate_comps
        
        Separate the sources as defined by the spectral
        components provided in spec_comp_ind.
        
        This function differs from separate_spat_comps in the way
        that it does not assume the sources are defined by their spatial
        positions.

        :param dir_results:
            provide the (existing) folder where to write the results
        :param dict spec_comp_ind:
            a dictionary telling which spectral component to include in
            which separated source. If None, the default is to assume
            each spectral component is one source.
            Note that this is different to the behaviour of
            :py:meth:`separate_spat_comps` which assumes that each
            *spatial* component corresponds to one source. 
        :param dict suffix:
            a dictionary containing the labels for each source. If None,
            then no suffix is appended to the file names and the files
            are simply numbered `XXX_nbComps`. 
        
        Note: Trying to bring into one method
        ozerov's separate_spec_comps and separate_spat_comps
        """
        if dir_results is None:
            dir_results = (
                '/'.join(
                self.audioObject.filename.split('/')[:-1])
                )
            if self.verbose:
                print "Writing to same directory as input file: " + dir_results
        
        nc = self.audioObject.channels
        if nc != 2:
            raise NotImplementedError()
        
        if spec_comp_ind is None:
            spec_comp_ind = {}
            for spec_ind in range(len(self.spec_comps)):
                spec_comp_ind[spec_ind] = [spec_ind,]
        
        nbSources = len(spec_comp_ind)
        sigma_comps_diag = np.zeros([nbSources, 2,
                                     self.nbFreqsSigRepr,
                                     self.nbFramesSigRepr])
        sigma_comps_off = np.zeros([nbSources,
                                    self.nbFreqsSigRepr,
                                    self.nbFramesSigRepr], dtype=np.complex)
        
        # computing individual source variance
        for n in range(nbSources):
            if self.verbose>1: print "    source",n+1,"out of",nbSources
            spat_comp_ind = np.unique(
                [self.spec_comps[spec_ind]['spat_comp_ind']
                 for spec_ind in spec_comp_ind[n]]
                )
            if self.verbose>1: print "        spat_comp_ind", spat_comp_ind
            for spat_ind in spat_comp_ind:
                if self.verbose>1:
                    print "        spatial comp",spat_ind+1, \
                          "out of", (spat_comp_ind)
                sigma_c_diag, sigma_c_off = (
                    self.compute_sigma_comp_2d(spat_ind, spec_comp_ind[n])
                    )
                sigma_comps_diag[n] += sigma_c_diag
                sigma_comps_off[n] += sigma_c_off
                del sigma_c_diag, sigma_c_off
        # deriving inverse of mix covariance:
        inv_sigma_x_diag, inv_sigma_x_off = self.compute_inv_sigma_mix_2d(
            sigma_comps_diag,
            sigma_comps_off)
        
        if not hasattr(self, "files"):
            self.files = {}
        self.files['spat_comp'] = []
        
        if True: # self # IF TRANSFO is STFT !!!... 20130507 corrected now?
            fileroot = self.audioObject.filename.split('/')[-1][:-4]
            for n in range(nbSources):
                # get the Wiener filters:
                WG = self.compute_Wiener_gain_2d(
                    sigma_comps_diag[n],
                    sigma_comps_off[n],
                    inv_sigma_x_diag,
                    inv_sigma_x_off)
                # compute the stft/istft
                if self.sig_repr_params['transf'] == 'stftold':
                    ndata = ao.filter_stft(
                        self.audioObject.data, WG, analysisWindow=None,
                        synthWindow=np.hanning(self.sig_repr_params['wlen']),
                        hopsize=self.sig_repr_params['hopsize'],
                        nfft=self.sig_repr_params['fsize'],
                        fs=self.audioObject.samplerate)
                else:
                    X = []
                    for chan1 in range(nc):
                        self.tft.computeTransform(
                            self.audioObject.data[:,chan1])
                        X.append(self.tft.transfo)
                    ndata = []
                    if WG.ndim == 3:
                        for chan1 in range(nc):
                            self.tft.transfo = np.zeros([self.nbFreqsSigRepr,
                                                         self.nbFramesSigRepr],
                                                        dtype=np.complex)
                            for chan2 in range(nc):
                                self.tft.transfo += (
                                    np.vstack(WG[chan1, chan2])
                                    * X[chan2])
                            ndata.append(self.tft.invertTransform())
                            del self.tft.transfo
                    elif WG.ndim == 4:
                        for chan1 in range(nc):
                            self.tft.transfo = np.zeros([self.nbFreqsSigRepr,
                                                         self.nbFramesSigRepr],
                                                        dtype=np.complex)
                            for chan2 in range(nc):
                                self.tft.transfo += (
                                    WG[chan1, chan2]
                                    * X[chan2])
                            ndata.append(self.tft.invertTransform())
                            del self.tft.transfo
                        
                    ndata = np.array(ndata).T
                _suffix = ''
                if suffix is not None and n in suffix:
                    _suffix = '_' + suffix[n]
                outAudioName = \
                    dir_results + '/' + fileroot + '_' + str(n) + \
                    '-' + str(nbSources) + _suffix + '.wav'
                self.files['spat_comp'].append(outAudioName)
                outAudioObj = ao.AudioObject(filename=outAudioName,
                                             mode='w')
                outAudioObj._data = np.int16(
                    ndata[:self.audioObject.nframes,:] *
                    self.audioObject._maxdata)#(2**15))
                outAudioObj._maxdata = 1
                outAudioObj._encoding = 'pcm16'
                outAudioObj.samplerate = self.audioObject.samplerate
                outAudioObj._write()
        ## TODO: else for the other transforms
        ##       should work all the same, but with cqt, not very good
        ## means to cut signals and paste them back together...
        
    def mvdr_2d(self,
                theta,
                distanceInterMic=.3,
                ):
        """mvdr_2d(self,
        theta, # in radians
        distanceInterMic=.3, # in meters
        )
        
        MVDR minimum variance distortion-less response spatial
        filter, for a given angle theta and given distance between the mics.
        
        self.Cx is supposed to provide the necessary covariance matrix, for
        the \"Capon\" filter.
        """
        Cx = np.copy(self.Cx)
        Cx[0][:,:] = np.vstack(Cx[0].mean(axis=1))
        Cx[1][:,:] = np.vstack(Cx[1].mean(axis=1))
        Cx[2][:,:] = np.vstack(Cx[2].mean(axis=1))
        if self.verbose>1:
            print Cx
        
        inv_Cx_diag, inv_Cx_off, det_Cx = inv_herm_mat_2d(
            [Cx[0], Cx[2]],
            Cx[1],
            verbose=self.verbose)
        freqs = (
            np.arange(self.nbFreqsSigRepr) * 1. /
            self.sig_repr_params['fsize'] * self.audioObject.samplerate
            )
        
        filt = gen_steer_vec_far_src_uniform_linear_array(
                   freqs,
                   nchannels=self.audioObject.channels,
                   theta=theta,
                   distanceInterMic=distanceInterMic)
        
        W = np.zeros([self.audioObject.channels, # nc x nc x F x N
                      self.audioObject.channels,
                      self.nbFreqsSigRepr,
                      self.nbFramesSigRepr], dtype=np.complex)
        
        den = (
            np.vstack(np.abs(filt[0])**2) * inv_Cx_diag[0] + 
            np.vstack(np.abs(filt[1])**2) * inv_Cx_diag[1] +
            2 * np.real(
                np.vstack(np.conj(filt[0]) * filt[1]) * inv_Cx_off)
            )
        W[1,1] = (
            np.vstack(filt[1] * np.conj(filt[0])) * inv_Cx_off
            )
        W[0,0] = (
            np.conj(W[1,1]) +
            np.vstack(np.abs(filt[0])**2) * inv_Cx_diag[0] 
            )
        W[1,1] += (
            np.vstack(np.abs(filt[1])**2) * inv_Cx_diag[1] 
            )
        W[0,1] = (
            np.vstack(np.abs(filt[0])**2) * inv_Cx_off +
            np.vstack(filt[0] * np.conj(filt[1])) * np.conj(inv_Cx_diag[1]) 
            )
        W[1,0] = (
            np.vstack(np.abs(filt[1])**2) * np.conj(inv_Cx_off) +
            np.vstack(filt[1] * np.conj(filt[0])) * np.conj(inv_Cx_diag[0])
            )
        #if self.verbose>1:
        #    print W
        # should check that self.sig_repr_params['transf'] == 'stft'
        return ao.filter_stft(
            self.audioObject.data,
            W,
            analysisWindow=np.hanning(self.sig_repr_params['wlen']),
            synthWindow=np.hanning(self.sig_repr_params['wlen']),
            hopsize=self.sig_repr_params['hopsize'],
            nfft=self.sig_repr_params['fsize'],
            fs=self.audioObject.samplerate)
    
    def gcc_phat_tdoa_2d(self):
        """Using the cross-spectrum in self.Cx[1] to estimate the time
        difference of arrival detection function (the Generalized
        Cross-Correlation GCC), with the phase transform (GCC-PHAT) weighing
        function for the cross-spectrum.
        """
        return np.fft.irfft(self.Cx[1]/np.abs(self.Cx[1]),
                            n=self.sig_repr_params['fsize'],
                            axis=0)
    
    
    def compute_sigma_comp_2d(self, spat_ind, spec_comp_ind):
        """only for stereo case self.audioObject.channels==2
        """
        
        spat_comp_power = self.comp_spat_comp_power(
            spat_comp_ind=spat_ind,
            spec_comp_ind=spec_comp_ind)
        
        # getting the mixing coefficients for corresponding
        # spatial source, depending on mix_type
        if self.spat_comps[spat_ind]['mix_type'] == 'inst':
            mix_coefficients = self.spat_comps[spat_ind]['params'].T
            # mix_coefficients.shape should be (rank, nchannels)
        elif self.spat_comps[spat_ind]['mix_type'] == 'conv':
            mix_coefficients = self.spat_comps[spat_ind]['params']
            # mix_coefficients.shape should be (rank, nchannels, freq)
        
        # R_diag = np.zeros(2, self.nbFreqsSigRepr)
        R_diag0 = np.atleast_1d(
            (np.abs(mix_coefficients[:, 0])**2).sum(axis=0))
        R_diag1 = np.atleast_1d(
            (np.abs(mix_coefficients[:, 1])**2).sum(axis=0))
        # element at (1,2): 
        R_off = np.atleast_1d((
            mix_coefficients[:, 0] *
            np.conj(mix_coefficients[:, 1])).sum(axis=0))
        
        sigma_comp_diag = np.zeros([2,
                                    self.nbFreqsSigRepr,
                                    self.nbFramesSigRepr])
        if self.verbose>1:
            print R_diag0, "R_diag0.shape", R_diag0.shape
            print R_diag1, "R_diag1.shape", R_diag1.shape
            print R_off, "R_off.shape", R_off.shape
        
        sigma_comp_diag[0] = (
            np.vstack(R_diag0) *
            spat_comp_power)
        sigma_comp_diag[1] = (
            np.vstack(R_diag1) *
            spat_comp_power)
        
        sigma_comp_off = (
            np.vstack(R_off) * spat_comp_power)
        
        return sigma_comp_diag, sigma_comp_off
    
    def compute_inv_sigma_mix_2d(self,
                                 sigma_comps_diag,
                                 sigma_comps_off):
        """only for nb channels = 2
        
        sigma_comps_diag ncomp x nchan x nfreq x nframes
        
        """
        sigma_x_diag = sigma_comps_diag.sum(axis=0)
        sigma_x_off = sigma_comps_off.sum(axis=0)
        for n in range(2):
            sigma_x_diag[n] += np.vstack(self.noise['PSD'])
            # noise PSD should be of size nbFreqs
        
        inv_sigma_x_diag, inv_sigma_x_off, _ = (
            inv_herm_mat_2d(sigma_x_diag, sigma_x_off,
                            verbose=self.verbose))
        
        del sigma_x_diag, sigma_x_off
        
        return inv_sigma_x_diag, inv_sigma_x_off
    
    def compute_Wiener_gain_2d(self,
                               sigma_comp_diag,
                               sigma_comp_off,
                               inv_sigma_mix_diag,
                               inv_sigma_mix_off,
                               timeInvariant=False):
        """
        Matlab FASST Toolbox help::
        
            % WG = comp_WG_spat_comps(mix_str);
            %
            % compute Wiener gains for spatial components
            %
            %
            % input
            % -----
            %
            % mix_str           : input mix structure
            % 
            %
            % output
            % ------
            %
            % WG                : Wiener gains [M x M x F x N x K_spat]
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Flexible Audio Source Separation Toolbox (FASST), Version 1.0
            %
            % Copyright 2011 Alexey Ozerov, Emmanuel Vincent and Frederic Bimbot
            % (alexey.ozerov -at- inria.fr, emmanuel.vincent -at- inria.fr,
            %  frederic.bimbot -at- irisa.fr)     
            %
            % This software is distributed under the terms of the GNU Public 
            % License version 3 (http://www.gnu.org/licenses/gpl.txt)
            %
            % If you use this code please cite this research report
            %
            % A. Ozerov, E. Vincent and F. Bimbot
            % \"A General Flexible Framework for the Handling of Prior
            % Information in Audio Source Separation,\" 
            % IEEE Transactions on Audio, Speech and Signal Processing 20(4),
            % pp. 1118-1133 (2012).
            % Available: http://hal.inria.fr/hal-00626962/
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        """
        # Here, WG is given by the product:
        #     np.dot(sigma_comp
        if timeInvariant:
            WG = np.zeros([2, 2,
                           self.nbFreqsSigRepr,],
                          dtype=complex)# stands for Wiener Gains
        else:
            WG = np.zeros([2, 2,
                           self.nbFreqsSigRepr,
                           self.nbFramesSigRepr],
                          dtype=complex)# stands for Wiener Gains
        WG[0,0] = sigma_comp_off * np.conj(inv_sigma_mix_off)
        WG[1,1] = np.conj(WG[0,0])
        WG[0,0] += sigma_comp_diag[0] * inv_sigma_mix_diag[0]
        WG[1,1] += sigma_comp_diag[1] * inv_sigma_mix_diag[1]
        
        WG[0,1] = (
            sigma_comp_diag[0] * inv_sigma_mix_off +
            sigma_comp_off * inv_sigma_mix_diag[1]
            )
        WG[1,0] = (
            np.conj(sigma_comp_off) * inv_sigma_mix_diag[0] +
            sigma_comp_diag[1] * np.conj(inv_sigma_mix_off)
            )
        
        return WG
    
    def update_spectral_components(self, hat_W):
        """Update the spectral components,
        with `hat_W` as the expected value of power
        (and computed from )
        """
        if self.verbose:
            print "    Update the spectral components"
        omega = self.nmfUpdateCoeff
        nbspeccomp = len(self.spec_comps)
        
        for spec_comp_ind, spec_comp in self.spec_comps.items():
            nbfactors = len(spec_comp['factor'])
            spat_comp_ind = spec_comp['spat_comp_ind']
            
            # DEBUG
            if self.lambdaCorr > 0: # min inter-src correlation approach
                # this is the sum of all the spatial component powers
                spat_comp_powers = np.maximum(self.comp_spat_cmps_powers(
                    self.spat_comps.keys()), eps)
                ### we need the squared of that matrix too:
                ##spat_comp_powers_sqd = spat_comp_powers ** 2
                # the initial spatial comp. power of the current comp:
                spat_comp_power = (
                    np.maximum(
                        self.comp_spat_comp_power(
                            spat_comp_ind,
                            #spec_comp_ind=[spec_comp_ind],
                            ),
                        eps)
                    )
                # ... and removing from the other powers - for correlation
                # control:
                spat_comp_pow_minus = spat_comp_powers - spat_comp_power
                
                if np.all(spat_comp_pow_minus >=0): # DEBUG
                    warnings.warn(
                        "Not all spat_comp_pow_minus, "+
                        "%d negative values!" %np.sum(spat_comp_pow_minus >=0))
                    spat_comp_pow_minus = np.maximum(spat_comp_pow_minus, eps)
                
            for fact_ind, factor in spec_comp['factor'].items():
                # update FB - freq basis
                other_fact_ind_arr = range(nbfactors)
                other_fact_ind_arr.remove(fact_ind)
                other_fact_power = (
                    np.maximum(
                        self.comp_spat_comp_power(
                            spat_comp_ind=spat_comp_ind,
                            spec_comp_ind=[spec_comp_ind],
                            factor_ind=other_fact_ind_arr),
                        eps)
                    )
                if factor['FB_frdm_prior'] == 'free':
                    if self.verbose>1:
                        print "    Updating frequency basis %d-%d" %(
                            spec_comp_ind, fact_ind)
                    spat_comp_power = (
                        np.maximum(
                            self.comp_spat_comp_power(
                                spat_comp_ind,
                                #spec_comp_ind=[spec_comp_ind]
                                ),
                            eps)
                        )
                    #comp_num = hat_W[spat_comp_ind] / spat_comp_power**(2)
                    #comp_den = 1 / spat_comp_power
                    
                    if len(factor['TB']):
                        H = np.dot(factor['TW'], factor['TB'])
                    else:
                        H = factor['TW']
                        
                    FW_H = np.dot(factor['FW'], H).T
                    
                    # denominator + correlation penalization
                    if self.lambdaCorr > 0:
                        corrPen = (
                            self.lambdaCorr
                            * spat_comp_pow_minus #np.maximum(spat_comp_powers,
                            #           eps)
                            / np.maximum(spat_comp_powers**2, eps)
                            )
                    else:
                        corrPen = 0.
                    
                    comp_den = (
                        np.dot(other_fact_power * 
                               (1. / spat_comp_power +
                                corrPen),
                               FW_H))
                    # numerator
                    if self.lambdaCorr > 0:
                        corrPen *= 2 *(
                            spat_comp_power
                            / spat_comp_powers
                            )
                    comp_num = (
                        np.dot((hat_W[spat_comp_ind]
                                / (spat_comp_power**2)
                                # np.maximum(spat_comp_power**(2), eps)
                                + corrPen)
                               * other_fact_power,
                               FW_H))
                    
                    factor['FB'] *= (
                        comp_num / np.maximum(comp_den, eps)) ** omega
                    del comp_num, comp_den, spat_comp_power, H, FW_H
                    
                # update FW - freq weight
                if factor['FW_frdm_prior'] == 'free':
                    if self.verbose>1:
                        print "    Updating frequency weights %d-%d" %(
                            spec_comp_ind, fact_ind)
                    spat_comp_power = (
                        np.maximum(
                            self.comp_spat_comp_power(
                                spat_comp_ind,
                                spec_comp_ind=[spec_comp_ind]),
                            eps)
                        )
                    
                    if len(factor['TB']):
                        H = np.dot(factor['TW'], factor['TB'])
                    else:
                        H = factor['TW']
                        
                    # denominator + correlation penalization
                    if self.lambdaCorr > 0:
                        corrPen = (
                            self.lambdaCorr
                            * np.maximum(spat_comp_pow_minus,#-spat_comp_power,
                                         eps)
                            / np.maximum(spat_comp_powers**2, eps)
                            )
                    else:
                        corrPen = 0.
                    comp_den = (
                        np.dot(factor['FB'].T,
                               np.dot(other_fact_power * 
                                      (1. / spat_comp_power +
                                       corrPen),
                                      #other_fact_power /
                                      #spat_comp_power,
                                      H.T))
                        )
                    
                    # numerator
                    if self.lambdaCorr > 0:# 
                        corrPen *= 2 *(
                            spat_comp_power
                            / spat_comp_powers
                            )
                    comp_num = (
                        np.dot(factor['FB'].T,
                               np.dot((hat_W[spat_comp_ind]
                                       / (spat_comp_power**2) #np.maximum(spat_comp_power**2,eps)
                                       + corrPen)
                                      * other_fact_power,
                                      H.T))
                        )
                    factor['FW'] *= (
                        comp_num / np.maximum(comp_den, eps)) ** omega
                    del comp_num, comp_den, spat_comp_power, H
                    
                # update TW - time weights
                if factor['TW_frdm_prior'] == 'free':
                    if factor['TW_constr'] == 'NMF':
                        if self.verbose>1:
                            print "    Updating time weights %d-%d" %(
                                spec_comp_ind, fact_ind)
                        spat_comp_power = (
                            np.maximum(
                                self.comp_spat_comp_power(
                                    spat_comp_ind,
                                    spec_comp_ind=[spec_comp_ind]),
                                eps)
                            )
                        
                        W = np.dot(factor['FB'], factor['FW'])
                        
                        # correlation penalization
                        if self.lambdaCorr > 0:
                            corrPen = (
                                self.lambdaCorr
                                * np.maximum(spat_comp_pow_minus,
                                             # - spat_comp_power,
                                             eps)
                                / np.maximum(spat_comp_powers**2, eps)
                                )
                            ##if self.verbose>2: # DEBUG
                            ##    # pedantic :
                            ##    print "correlation stuff",
                            ##    print corrPen.mean(), (1./spat_comp_power).mean()
                        else:
                            corrPen = 0.
                        
                        if len(factor['TB']):
                            # denominator
                            comp_den = (
                                np.dot(W.T,
                                       np.dot(other_fact_power * 
                                              (1. / spat_comp_power +
                                               corrPen),#other_fact_power / 
                                              #spat_comp_power,
                                              factor['TB'].T)
                                       )
                                )
                            # numerator
                            if self.lambdaCorr > 0:# corrPen > 0:
                                corrPen *= 2 *(
                                    spat_comp_power
                                    / spat_comp_powers
                                    )
                            comp_num = (
                                np.dot(W.T,
                                       np.dot((hat_W[spat_comp_ind] /
                                               (spat_comp_power**2) #np.maximum(spat_comp_power**2,
                                               #            eps)
                                               + corrPen)
                                              * other_fact_power,
                                              factor['TB'].T)
                                       )
                                )
                        else:
                            # denominator
                            comp_den = (
                                np.dot(W.T,
                                       other_fact_power * 
                                       (1. / spat_comp_power +
                                        corrPen), #other_fact_power / 
                                       #spat_comp_power
                                       )
                                )
                            # numerator
                            if self.lambdaCorr > 0:# corrPen > 0:
                                corrPen *= 2 *(
                                    spat_comp_power
                                    / spat_comp_powers
                                    )
                            ##if self.verbose>5: # DEBUG to discover origin of NaN
                            ##    print "corrPen", corrPen
                            ##    print "other_fact_power", other_fact_power
                            ##    print "hat_W", hat_W[spat_comp_ind]
                            ##    print "squared", np.maximum(spat_comp_power**2,eps)
                                
                            comp_num = (
                                np.dot(W.T,
                                       other_fact_power * (hat_W[spat_comp_ind] /
                                        (spat_comp_power**2)
                                        + corrPen)
                                       )
                                )
                            
                        ##if self.verbose > 8: #DEBUG
                        ##    print "comp_num", comp_num
                        ##    print "comp_den", comp_den
                        factor['TW'] *= (
                            comp_num / np.maximum(comp_den, eps)) ** omega
                        del comp_num, comp_den, spat_comp_power, W
                    elif factor['TW_constr'] in ('GMM', 'GSMM', 'HMM', 'SHMM'):
                        warnings.warn(
                            "The GMM/GSMM/HMM still needs to be adapted "+
                            "to take into account the different factors. ")
                        nbfaccomps = factor['TW'].shape[0]
                        if self.verbose>1:
                            print "    Updating time weights, "+\
                                  "discrete state-based constraints"
                        if len(factor['TB']):
                            errorMsg = "In this implementation, "+\
                                       "as in Ozerov's, non-trivial "+\
                                       "time blobs TB is incompatible with "+\
                                       "discrete state-based constraints for"+\
                                       " the time weights TW"
                            raise AttributeError(errorMsg)
                        
                        if not('TW_all' in factor):
                            factor['TW_all'] = (
                                np.outer(np.ones(nbfaccomps),
                                         np.max(factor['TW'], axis=0))
                                )
                            
                        if 'TW_DP_params' not in factor:
                            if factor['TW_constr'] in ('GMM', 'GSMM'):
                                # prior probabilities
                                factor['TW_DP_params'] = (
                                    np.ones(nbfaccomps) /
                                    np.double(nbfaccomps))
                            else:
                                # transition probabilities
                                factor['TW_DP_params'] = (
                                    np.ones([nbfaccomps, nbfaccomps]) /
                                    np.double(nbfaccomps))
                                
                        if factor['TW_constr'] in ('GMM', 'HMM') and \
                               (np.max(factor['TW_all'])>1 or \
                                np.min(factor['TW_all'])<1):
                            factor['FB'] *= np.mean(factor['TW_all'])
                            factor['TW_all'][:] = 1.
                            
                        if self.verbose:
                            print "    Computing the Itakura Saito distance"+\
                                  " matrix"
                        ISdivMatrix = np.zeros([nbfaccomps,
                                                self.nbFramesSigRepr])
                        for compnb in range(nbfaccomps):
                            factor['TW'][:] = 0
                            factor['TW'][compnb] = factor['TW_all'][compnb]
                            
                            if factor['TW_constr'] not in ('GMM', 'HMM'):
                                # re-estimating the weights for discrete
                                # state model with the constraint on the
                                # single state presence active.
                                # NB: for GMM and HMM, these weights are
                                #     assumed to be 1
                                spat_comp_power = (
                                    np.maximum(
                                        self.comp_spat_comp_power(
                                            spat_comp_ind,
                                            spec_comp_ind=[spec_comp_ind],),
                                        eps)
                                    )
                                 
                                # NMF like updating for estimating the weight
                                Wbasis = np.dot(factor['FB'],
                                                factor['FW'][:,compnb])
                                comp_num = (
                                    np.dot(Wbasis,
                                           hat_W[spat_comp_ind] /
                                           np.maximum(spat_comp_power**2, eps))
                                    )
                                comp_den = (
                                    np.dot(Wbasis,
                                           1 / spat_comp_power)
                                    )
                                
                                factor['TW'][compnb] *= (
                                    comp_num /
                                    np.maximum(comp_den, eps)
                                    ) ** omega
                                
                                factor['TW_all'][compnb]=factor['TW'][compnb]
                                
                                del comp_num, comp_den, spat_comp_power
                                
                            # ratio to compute IS divergence between expected
                            # variance hat_W and the spatial component
                            # with the discrete state restriction
                            spat_comp_power = (
                                np.maximum(
                                self.comp_spat_comp_power(spat_comp_ind),
                                eps)
                                )
                            
                            W_V_ratio = (
                                hat_W[spat_comp_ind] /
                                spat_comp_power)
                            
                            ISdivMatrix[compnb] = (
                                np.sum(W_V_ratio
                                       - np.log(np.maximum(W_V_ratio, eps))
                                       - 1,axis=0)
                                )
                            
                            del W_V_ratio, spat_comp_power
                        
                        # decode the state sequence that minimizes the
                        # track in the IS div matrix, with best
                        # trade-off with the provided TW_DP_params
                        # (temporal constraints)
                        if self.verbose:
                            print "    Decoding the state sequence"
                        if factor['TW_constr'] in ('GMM', 'GSMM'):
                            active_state_seq = (
                                np.argmin(
                                    ISdivMatrix -
                                    np.vstack(
                                        np.log(factor['TW_DP_params'] + eps)),
                                    axis=0)
                                )
                            del ISdivMatrix
                        elif factor['TW_constr'] in ('HMM', 'SHMM'):
                            if self.verbose:
                                print "        Viterbi algorithm to "+\
                                      "determine the active state sequence"
                            accumulateVec = (
                                ISdivMatrix[:,0] -
                                np.log(1. / nbfaccomps)
                                )
                            antecedentMat = np.zeros([nbfaccomps,
                                                      self.nbFramesSigRepr],
                                                     dtype=np.int32)
                            for n in range(1, self.nbFramesSigRepr):
                                tmpMat = (
                                    np.vstack(accumulateVec) -
                                    np.log(factor['TW_DP_params'] + eps))
                                
                                antecedentMat[:,n] = (
                                    np.argmin(tmpMat, axis=0)
                                    )
                                accumulateVec += (
                                    tmpMat[antecedentMat[:,n],
                                           range(nbfaccomps)] + 
                                    ISdivMatrix[:,n]
                                    )
                                # to avoid overflow?
                                accumulateVec -= accumulateVec.min()
                            
                            del tmpMat
                            
                            active_state_seq = np.zeros(self.nbFramesSigRepr,
                                                        dtype=np.int32)
                            active_state_seq[-1] = np.argmin(accumulateVec)
                            for framenb in range(self.nbFramesSigRepr-1,0,-1):
                                active_state_seq[framenb-1] = (
                                    antecedentMat[active_state_seq[framenb],
                                                  framenb-1]
                                    )
                            
                        else:
                            raise NotImplementedError(
                                "No implementation for time constraint other "+
                                "than GMM, GSMM, HMM and SHMM")
                        
                        if self.verbose:
                            print "    Update Time Weights"
                            
                        factor['TW'][:] = 0.
                        for framenb in range(self.nbFramesSigRepr):
                            factor['TW'][active_state_seq[framenb],framenb] = (
                                factor['TW_all'][active_state_seq[framenb],
                                                 framenb]
                                )
                            
                        if factor['TW_DP_frdm_prior'] == 'free':
                            print "    Updating the transition probabilities"
                            if factor['TW_constr'] in ('GMM', 'GSMM'):
                                for compnb in range(nbfaccomps):
                                    factor['TW_DP_params'][compnb] = (
                                        np.sum(active_state_seq==compnb) * 1. /
                                        self.nbFramesSigRepr
                                        )
                            elif factor['TW_constr'] in ('HMM', 'SHMM'):
                                for prevstate in range(nbfaccomps):
                                    upd_den = np.sum(
                                        active_state_seq[:-1]==prevstate)
                                    if upd_den:
                                        for nextstate in range(nbfaccomps):
                                            upd_num = 1. * np.sum(
                                                (active_state_seq[:-1]==
                                                 prevstate) *
                                                (active_state_seq[1:]==
                                                 nextstate))
                                            factor['TW_DP_params'][prevstate,
                                                                   nextstate]=(
                                                upd_num / upd_den
                                                ) # TODO: check this part
                            else:
                                raise NotImplementedError(
                                    "Required time constraints not "+
                                    "implemented.")
                        
                # update TB = time basis
                if len(factor['TB']) and factor['TB_frdm_prior'] == 'free':
                    if self.verbose>1: print "    Updating Time basis"
                    spat_comp_power = (
                        np.maximum(
                            self.comp_spat_comp_power(
                                spat_comp_ind,
                                spec_comp_ind=[spec_comp_ind],),
                            eps)
                        )
                    W = (
                        np.dot(np.dot(factor['FB'], factor['FW']),
                               factor['TW'])
                        )
                    # denominator + correlation penalization
                    if self.lambdaCorr > 0:
                        corrPen = (
                            self.lambdaCorr
                            * np.maximum(spat_comp_pow_minus,# - spat_comp_power,
                                         eps)
                            / np.maximum(spat_comp_powers**2, eps)
                            )
                        ##if self.verbose>2:#DEBUG
                        ##    # pedantic :
                        ##    print corrPen.mean(), (1./spat_comp_power).mean()
                    else:
                        corrPen = 0.
                    comp_den = (
                        np.dot(W.T,
                               other_fact_power * 
                               (1. / spat_comp_power +
                                corrPen))
                        )
                    # numerator
                    if self.lambdaCorr > 0:# corrPen > 0:
                        corrPen *= 2 *(
                            spat_comp_power
                            / spat_comp_powers
                            )
                    comp_num = (
                        np.dot(W.T,
                               (hat_W[spat_comp_ind]
                                / np.maximum(spat_comp_power**2, eps)
                                + corrPen)
                               * other_fact_power)
                        )
                    factor['TB'] *= (
                        comp_num / np.maximum(comp_den, eps)) ** omega
                    del comp_num, comp_den, spat_comp_power, W
    
    def renormalize_parameters(self):
        """renormalize_parameters
        
        Re-normalize the components
        """
        if self.verbose>0:
            print "    re-normalizing components"
        pass
        if self.verbose>1:
            print "         normalizing spatial components..."
        # renormalize spatial components
        Kspat = len(self.spat_comps)
        spat_global_energy = np.zeros(Kspat)
        for spat_ind, spat_comp in self.spat_comps.items():
            spat_global_energy[spat_ind] = (
                np.mean (np.abs(spat_comp['params'])**2))
            spat_comp['params'] /= np.sqrt(spat_global_energy[spat_ind])
            
        if self.verbose>5:
            print "spat_global_energy", spat_global_energy
        
        # renormalize spectral components
        Kspec = len(self.spec_comps)
        for spec_ind, spec_comp in self.spec_comps.items():
            global_energy = spat_global_energy[spec_comp['spat_comp_ind']]
            
            nbfactors = len(spec_comp['factor'])
            
            for fact_ind, factor in spec_comp['factor'].items():
                factor['FB'] *= global_energy
                w = factor['FB'].max(axis=0)#.mean(axis=0)
                w[w==0] = 1.
                factor['FB'] /= w
                factor['FW'] *= np.vstack(w)
                
                if factor['TW_constr'] not in ('GMM', 'HMM'):
                    w = factor['FW'].mean(axis=0)
                    w[w==0] = 1.
                    factor['FW'] /= w
                    factor['TW'] *= np.vstack(w)
                    # Only testing this: in order to avoid
                    # big crash, if for one factor, everything in TW
                    # turns out to get 0, then "restart" it with random
                    if np.sum(factor['TW']) < eps:
                        factor['TW'] = np.random.randn(*factor['TW'].shape)**2
                        factor['TW'] *= 1e3 * eps # so it s not too small
                        if self.verbose:
                            print "    renorm: reinitialized TW for spec",
                            print spec_ind, "factor", fact_ind
                    if len(factor['TB']):
                        w = factor['TB'].mean(axis=1)
                        w[w==0] = 1.
                        factor['TB'] /= np.vstack(w)
                        factor['TW'] *= w
                        
                    global_energy = factor['TW'].mean()
                    if fact_ind < (nbfactors - 1):
                        factor['TW'] /= global_energy
                else:
                    raise NotImplementedError(
                        "Temporal discrete state mngmt not done yet. ")
            
    def setComponentParameter(self, newValue, spec_ind, fact_ind=0,
                              partLabel='FB', prior='free',
                              keepDimensions=True):
        """A helper function to set a
        :py:attr:`FASST.spec_comp[spec_ind]['factor'][fact_ind][partLabel]` to
        the given value.
        
        TODO 20130522 finish this function to make it general purpose...
        """
        ###### DEBUG #####
        print "NOT IMPLEMENTED YET, PLEASE SET THE COMPONENTS DIRECTLY"
        pass
        ###### DEBUG #####
        if keepDimenstions:
            if (newValue.shape !=
            self.spec_comp[spec_ind]['factor'][fact_ind][partLabel].shape):
                raise ValueError("the provided value does not have the correct"+
                                 " size:"+str(newValue.shape)+
                                 " instead of "+
                                 str(
                 self.spec_comp[spec_ind]['factor'][fact_ind][partLabel].shape))
            
        else:
            # nightmare of error checking for sizes... 
            if partLabel == 'FB':
                newShape = newValue.shape
                oldShape = self.spec_comp[
                    spec_ind]['factor'][fact_ind]['FB'].shape
                if newShape[0] != self.nbFreqsSigRepr:
                    raise ValueError("FB: cannot change dimension of "+
                                     "signal representation.")
                if newShape[1] != oldShape[1]:
                    if self.verbose:
                        print "    Changing the Freq Weights for FB:"
                        self.spec_comp[spec_ind]['factor'][fact_ind]['FB']
                self.spec_comp[spec_ind]['factor'][fact_ind]['FB'] = newValue
            elif partLabel == 'FW':
                pass
            elif partLabel == 'TW':
                pass
            elif partLabel == 'TB':
                pass
            else:
                raise ValueError("No such thing as "+
                                 partLabel+
                                 " in components!")
        self.spec_comp[spec_ind]['factor'][fact_ind][
            partLabel+'_frdm_prior'] = prior
                
    def initialize_all_spec_comps_with_NMF(self,
                                           sameInitAll=False,
                                           **kwargs):
        """Computes an NMF on the one-channel mix (averaging diagonal
        of self.Cx, which are the power spectra of the corresponding
        channel)

        .. math::
        
            C_x \\approx W H
        
        then, for all spec_comp in self.spec_comps, we set::
        
            spec_comp['FB'] = W
            spec_comp['TW'] = H
            
        
        """
        if sameInitAll:
            # initialize the components with the same parameters
            return self.initialize_all_spec_comps_with_NMF_same(**kwargs)
        else:
            # initialize the components with individual params,
            # in particular, initializing the NMF with the available
            # components (but only with factor 0)
            return self.initialize_all_spec_comps_with_NMF_indiv(**kwargs)

    def initialize_all_spec_comps_with_NMF_indiv(self, niter=10,
                                                 updateFreqBasis=True,
                                                 updateTimeWeight=True,
                                                 **kwargs):
        """initialize the spectral components with an NMF decomposition,
        with individual decomposition of the monophonic signal TF
        representation.

        TODO make keepFBind and keepTWind, in order to provide
        finer control on which indices are updated. Also requires
        a modified NMF decomposition function.
        
        """
        # list of the sizes of the 0th factors, of all components
        nbSpecComps = [spec_comp['factor'][0]['FB'].shape[1]
                       for spec_comp in self.spec_comps.values()]
        totalNMFComps = np.sum(nbSpecComps)
        
        # initializing the NMF FreqBasis (FB) and TimeWeight (TW)
        # with the corresponding quantities in self.spec_comps:
        FBinit = np.zeros([self.nbFreqsSigRepr, totalNMFComps])
        TWinit = np.zeros([totalNMFComps, self.nbFramesSigRepr])
        
        for spec_ind, spec_comp in self.spec_comps.items():
            ind_start = np.sum(nbSpecComps[:spec_ind])
            ind_stop = ind_start + nbSpecComps[spec_ind]
            FBinit[:,ind_start:ind_stop] = (
                spec_comp['factor'][0]['FB'])
            TWinit[ind_start:ind_stop] = (
                spec_comp['factor'][0]['TW'])
        
        # computing the monaural signal representation
        #     summing the contributions over all the channels:
        nc = self.audioObject.channels
        Cx = np.copy(np.real(self.Cx[0]))
        for chan in range(1, nc):
            # stored an "efficient" way, so index "complicated":
            index = np.sum(np.arange(nc, nc-chan, -1))
            Cx += np.real(self.Cx[index])
            
        Cx /= np.double(nc)
        
        W, H = NMF_decomp_init(SX=Cx, nbComps=totalNMFComps,
                               niter=niter, verbose=self.verbose,
                               Winit=FBinit, Hinit=TWinit,
                               updateW=updateFreqBasis,
                               updateH=updateTimeWeight)
        
        # copy the result in the corresponding spec_comps:
        for spec_ind, spec_comp in self.spec_comps.items():
            ind_start = np.sum(nbSpecComps[:spec_ind])
            ind_stop = ind_start + nbSpecComps[spec_ind]
            if updateFreqBasis:
                spec_comp['factor'][0]['FB'] = (
                    np.maximum(
                        W[:,ind_start:ind_stop],
                        eps)
                    )
            if updateTimeWeight:
                spec_comp['factor'][0]['TW'] = ( 
                    np.maximum(H[ind_start:ind_stop],eps))
                
        self.renormalize_parameters()
    
    def initialize_all_spec_comps_with_NMF_same(self, niter=10,
                                                **kwargs):
        """
        Initialize all the components with the same amplitude and spectral
        matrices `W` and `H`.
        """
        if not np.all([len(spec_comp['factor'])==1
                       for spec_comp in self.spec_comps.values()]):
            raise NotImplementedError(
                "NMF init not implemented for multi factor models.")
        
        nbSpecComps = [spec_comp['factor'][0]['FB'].shape[1]
                       for spec_comp in self.spec_comps.values()]
        nbComps = np.max(nbSpecComps)
        
        nc = self.audioObject.channels
        # computing the signal representation
        Cx = np.copy(np.real(self.Cx[0]))
        for chan in range(1, nc):
            # stored an "efficient" way, so index "complicated":
            index = np.sum(np.arange(nc, nc-chan, -1))
            Cx += np.real(self.Cx[index])
            
        Cx /= np.double(nc)
        
        # computing NMF of Cx:
        W, H = NMF_decomposition(SX=Cx, verbose=self.verbose,
                                 nbComps=nbComps, niter=niter)
        
        # reordering so that most energy in first components
        Hsum = H.sum(axis=1)
        indexSort = np.argsort(Hsum)[::-1]
        W = W[:,indexSort]
        H = H[indexSort]
        
        for spec_comp in self.spec_comps.values():
            ncomp = spec_comp['factor'][0]['FB'].shape[1]
            spec_comp['factor'][0]['FB'][:] = W[:, :ncomp]
            spec_comp['factor'][0]['TW'][:] = H[:ncomp]
        
        self.renormalize_parameters()
    
    def initializeConvParams(self, initMethod='demix'):
        """setting the spatial parameters

        :param str initMethod:
            initialization method. Can be either of: 'demix', 'rand'.
            If 'demix', then the spatial parameters are initialized by the
            anechoic steering vector corresponding to the first directions
            estimated by the DEMIX algorithm [Arberet2010]_, using the
            algorithm implemented in :py:mod:`pyfasst.demixTF`.
        
        """
        nc = self.audioObject.channels
        for spat_ind, spat_comp in self.spat_comps.items():
            if spat_comp['mix_type'] != 'inst':
                warnings.warn("Spatial component %d "%spat_ind+
                              "already not instantaneous, overwriting...")
            
            # spat_comp['time_dep'] = 'indep'
            spat_comp['mix_type'] = 'conv'
            # spat_comp['frdm_prior'] = 'free'
        
        if initMethod == 'demix':
            maxclusters = max(40, 10 * len(self.spat_comps))
            neighbours = 15
            
            # default for demix to work best: #FIXME!!!
            wlen = self.demixParams['wlen']# 2048
            hopsize = self.demixParams['hopsize']#1024 
            
            demixInst = demix.DEMIX(
                audio=self.audioObject.filename,
                nsources=len(self.spat_comps), # spatial comps for demix
                wlen=wlen,
                hopsize=hopsize,
                neighbors=neighbours,
                verbose=self.verbose,
                maxclusters=maxclusters)
            
            demixInst.comp_pcafeatures()
            demixInst.comp_parameters()
            demixInst.init_subpts_set()
            demixInst.comp_clusters()
            demixInst.refine_clusters()
            
            # mixing parameters from DEMIX estimation:
            #     results in an nsrc x nfreqs x nc array
            A = demixInst.steeringVectorsFromCentroids()
            del demixInst
        elif 'rand' in initMethod:
            A = (
                np.random.randn(len(self.spat_comps),
                                self.nbFreqsSigRepr,
                                nc,)
                + 1j * np.random.randn(len(self.spat_comps),
                                       self.nbFreqsSigRepr,
                                       nc,)
                )
        else:
            raise ValueError("Init method not implemented.")
            
        # filling the spatial components:
        for nspat, (spat_ind, spat_comp) in enumerate(self.spat_comps.items()):
            spat_comp_param_inst = spat_comp['params']
            spat_comp['params'] = np.zeros([self.rank[nspat],
                                            nc,
                                            self.nbFreqsSigRepr],
                                           dtype=np.complex)
            for r in range(self.rank[nspat]):
                spat_comp['params'][r] = (
                    A[spat_ind].T
                    )
    
class MultiChanNMFInst_FASST(FASST):
    """\
    This class implements the Multi-channel Non-Negative Matrix Factorisation
    (NMF)
    
    **Inputs:**
    
    :param audio:
        as in :py:class:`FASST`, `audio` is the filename of the file to be
        processed or directly a :py:class:`pyfasst.audioObject.AudioObject`
    :param integer nbComps:
        the number of desired sources/components
    :param integer nbNMFComps:
        the number of NMF components for each of the components.
        TODO: allow to pass a list so that the user can control the number of
        elements source by source, individually
    :param spatial_rank:
        the spatial rank of all the components. If it's a `nbComps`-long list,
        then `spatial_rank[n]` will be the spatial rank for the `n`-th source.
    :type spatial_rank: integer or list
    
    **Example:**
    
    ::
    
        >>> import pyfasst.audioModel as am
        >>> filename = 'data/tamy.wav'
        >>> # initialize the model
        >>> model = am.MultiChanNMFInst_FASST(
                audio=filename,
                nbComps=2, nbNMFComps=32, spatial_rank=1,
                verbose=1, iter_num=50)
        >>> # estimate the parameters
        >>> log_lik = model.estim_param_a_post_model()
        >>> # separate the sources using these parameters
        >>> model.separate_spat_comps(dir_results='data/')
    
    """
    def __init__(self, audio,
                 nbComps=3, nbNMFComps=4,
                 spatial_rank=2,
                 **kwargs):
        super(MultiChanNMFInst_FASST, self).__init__(audio=audio, **kwargs)
        self.comp_transf_Cx()
        
        self.nbComps = nbComps
        self.nbNMFComps = nbNMFComps
        self.rank = np.atleast_1d(spatial_rank)
        if self.rank.size < self.nbComps:
            self.rank = [self.rank[0],] * self.nbComps
        
        self._initialize_structures()
    
    def _initialize_structures(self): #, nbComps, nbNMFComps, spatial_rank):
        """Initializes the structures: spatial components (instantaneous) and
        spectral components (1 factor, with NMF simple structure).
        """
        nc = self.audioObject.channels
        
        self.spat_comps = {}
        self.spec_comps = {}
        for j in range(self.nbComps):
            # initialize the spatial component
            self.spat_comps[j] = {}
            self.spat_comps[j]['time_dep'] = 'indep'
            self.spat_comps[j]['mix_type'] = 'inst'
            self.spat_comps[j]['frdm_prior'] = 'free'
            self.spat_comps[j]['params'] = np.random.randn(nc, self.rank[j])
            if nc == 2: # spreading the sources evenly for init on stereo
                self.spat_comps[j]['params'] = (
                    np.array([np.sin((j+1) * np.pi / (2.*(self.nbComps + 1))) +
                              np.random.randn(self.rank[j])*np.sqrt(0.01),
                              np.cos((j+1) * np.pi / (2.*(self.nbComps + 1))) +
                              np.random.randn(self.rank[j])*np.sqrt(0.01)]))
            
            # initialize single factor spectral component
            self.spec_comps[j] = {}
            self.spec_comps[j]['spat_comp_ind'] = j
            self.spec_comps[j]['factor'] = {}
            self.spec_comps[j]['factor'][0] = {}
            self.spec_comps[j]['factor'][0]['FB'] = (
                0.75 * np.abs(np.random.randn(self.nbFreqsSigRepr,
                                              self.nbNMFComps)) +
                0.25)
            self.spec_comps[j]['factor'][0]['FW'] = (
                np.eye(self.nbNMFComps))
            self.spec_comps[j]['factor'][0]['TW'] = (
                0.75 * np.abs(np.random.randn(self.nbNMFComps,
                                              self.nbFramesSigRepr)) +
                0.25)
            self.spec_comps[j]['factor'][0]['TB'] = []
            self.spec_comps[j]['factor'][0]['FB_frdm_prior'] = 'free'
            self.spec_comps[j]['factor'][0]['FW_frdm_prior'] = 'fixed'
            self.spec_comps[j]['factor'][0]['TW_frdm_prior'] = 'free'
            self.spec_comps[j]['factor'][0]['TB_frdm_prior'] = []
            self.spec_comps[j]['factor'][0]['TW_constr'] = 'NMF'
            
        self.renormalize_parameters()
        
    def setSpecCompFB(self, compNb, FB, FB_frdm_prior='fixed'):
        """\
        sets the spectral component's frequency basis.

        :param integer compNb:
            the component to be initialized
        :param numpy.ndarray FB:
            the initial array to put in
            :py:attr:`spec_comp[compNb]['factor'][0]['FB']`
        :param str FB_frdm_prior:
            either 'fixed' or 'free'. 
        
        """
        speccomp = self.spec_comps[compNb]['factor'][0]
        if self.nbFreqsSigRepr != FB.shape[0]:
            raise AttributeError("Size of provided FB is not consistent"+
                                 " with inner attributes")
        speccomp['FB'] = np.copy(FB)
        ncomp = FB.shape[1]
        
        speccomp['FW'] = np.eye(ncomp)
        speccomp['TW'] = (
                0.75 * np.abs(np.random.randn(ncomp,
                                              self.nbFramesSigRepr)) +
                0.25)
        speccomp['FB_frdm_prior'] = FB_frdm_prior

class MultiChanNMFConv(MultiChanNMFInst_FASST):
    """\
    Takes the multichannel NMF instantaneous class, and makes it
    convolutive!
    
    Simply adds a method :py:meth:`makeItConvolutive` in order to transform
    instantaneous mixing parameters into convolutive ones.

    **Example:**
    
    ::
  
        >>> import pyfasst.audioModel as am
        >>> filename = 'data/tamy.wav'
        >>> # initialize the model
        >>> model = am.MultiChanNMFConv(
                audio=filename,
                nbComps=2, nbNMFComps=32, spatial_rank=1,
                verbose=1, iter_num=50)
        >>> # to be more flexible, the user _has to_ make the parameters
        >>> # convolutive by hand. This way, she can also start to estimate
        >>> # parameters in an instantaneous setting, as an initialization, 
        >>> # and only after "upgrade" to a convolutive setting:
        >>> model.makeItConvolutive()
        >>> # estimate the parameters
        >>> log_lik = model.estim_param_a_post_model()
        >>> # separate the sources using these parameters
        >>> model.separate_spat_comps(dir_results='data/')

    The following example shows the results for a more synthetic example
    (synthetis anechoic mixture of the voice and the guitar, with a delay of 0
    for the voice and 10 samples from the left to the right channel
    for the guitar)::
    
        >>> import pyfasst.audioModel as am
        >>> filename = 'data/dev1__tamy-que_pena_tanto_faz___thetas-0.79,0.79_delays-10.00,0.00.wav'
        >>> # initialize the model
        >>> model = am.MultiChanNMFConv(
                audio=filename,
                nbComps=2, nbNMFComps=32, spatial_rank=1,
                verbose=1, iter_num=200)
        >>> # to be more flexible, the user _has to_ make the parameters
        >>> # convolutive by hand. This way, she can also start to estimate
        >>> # parameters in an instantaneous setting, as an initialization, 
        >>> # and only after "upgrade" to a convolutive setting:
        >>> model.makeItConvolutive()
        >>> # we can initialize these parameters with the DEMIX algorithm:
        >>> model.initializeConvParams(initMethod='demix')
        >>> # and estimate the parameters:
        >>> log_lik = model.estim_param_a_post_model()
        >>> # separate the sources using these parameters
        >>> model.separate_spat_comps(dir_results='data/')
    
    """
    def __init__(self, audio,
                 nbComps=3, nbNMFComps=4,
                 spatial_rank=2,
                 **kwargs):
        super(MultiChanNMFConv, self).__init__(audio=audio,
                                               nbComps=nbComps,
                                               nbNMFComps=nbNMFComps,
                                               spatial_rank=spatial_rank,
                                               **kwargs)
        # self.makeItConvolutive()
        # DIY: upgrade to convolutive after a few instantaneous, maybe? 
        
    def makeItConvolutive(self):
        """If the spatial parameters are instantaneous, then it will be turned
        into a convolutive version of it. In this case, it duplicates the
        instantaneous parameter on all the frequencies and spatial rank.
        """
        nc = self.audioObject.channels
        for nspat, (spat_ind, spat_comp) in enumerate(self.spat_comps.items()):
            if spat_comp['mix_type'] != 'inst':
                warnings.warn("Spatial component %d "%spat_ind+
                              "already not instantaneous, skipping...")
            else:
                # spat_comp['time_dep'] = 'indep'
                spat_comp['mix_type'] = 'conv'
                # spat_comp['frdm_prior'] = 'free'
                spat_comp_param_inst = spat_comp['params']
                spat_comp['params'] = np.zeros([self.rank[nspat],
                                                nc,
                                                self.nbFreqsSigRepr],
                                               dtype=np.complex)
                for f in range(self.nbFreqsSigRepr):
                    spat_comp['params'][:,:,f] = spat_comp_param_inst.T

class MultiChanHMM(MultiChanNMFConv):
    """Conveniently adds methods to transform a :py:class:`MultiChanNMFConv`
    object such that the time structure is configured as a hidden Markov
    model (HMM) 
    """
    def __init__(self, audio,
                 nbComps=3, nbNMFComps=4,
                 spatial_rank=2,
                 **kwargs):
        super(MultiChanHMM, self).__init__(audio=audio,
                                               nbComps=nbComps,
                                               nbNMFComps=nbNMFComps,
                                               spatial_rank=spatial_rank,
                                               **kwargs)
        
    def makeItHMM(self):
        """
        Turns the required parameters into HMM time constraints
        """
        for spec_ind, spec_comp in self.spec_comps.items():
            for fac_ind, factor in spec_comp['factor'].items():
                factor['TW_constr'] = 'HMM'
                factor['TW_DP_frdm_prior'] = 'free'
    
    def makeItSHMM(self):
        """
        Turns the required parameters into SHMM time constraints
        """
        for spec_ind, spec_comp in self.spec_comps.items():
            for fac_ind, factor in spec_comp['factor'].items():
                nbfaccomps = factor['TW'].shape[0]
                factor['TW_constr'] = 'SHMM'
                factor['TW_DP_params'] = (
                    9 * np.eye( nbfaccomps)
                    )
                factor['TW_DP_params'] += 1.
                factor['TW_DP_params'] /= (
                    np.vstack(factor['TW_DP_params'].sum(axis=1)))
                factor['TW_DP_frdm_prior'] = 'fixed'
                # factor['TW_DP_frdm_prior'] = 'free'
        
class multiChanSourceF0Filter(FASST):
    """multi channel source/filter model
    nbcomps components, nbcomps-1 SF models, 1 residual component
    
    """
    def __init__(self, audio,
                 nbComps=3, 
                 nbNMFResComps=1, 
                 nbFilterComps=20, 
                 nbFilterWeigs=[4,], 
                 minF0=39, maxF0=2000, minF0search=80, maxF0search=800,
                 stepnoteF0=16, chirpPerF0=1, 
                 spatial_rank=1,
                 sparsity=None,
                 **kwargs):
        """
        **DESCRIPTION**
        __init__(self, audio,
                 nbComps=3, ## nb of components
                 nbNMFResComps=3, ## nb of residual components
                 nbFilterComps=20, ## nb of filter components
                 nbFilterWeigs=4, ## nb of filter components
                 minF0=80, maxF0=800, ## range for comb spectra
                 stepnoteF0=4, chirpPerF0=1, 
                 spatial_rank=1,
                 sparsity=None,
                 **kwargs)
        
        **ARGUMENTS**
        
        nbComps (int)
            The number of (spatial) components in FASST framework.
            
        nbNMFComps (int)
            The number of NMF components in each spatial component.
            
        sparsity (list of size 1 or nbComps)
            
        
        """
        super(multiChanSourceF0Filter, self).__init__(audio=audio, **kwargs)
        self.comp_transf_Cx()
        self.sourceParams = {'minF0': minF0,
                             'maxF0': maxF0,
                             'stepnoteF0': stepnoteF0,
                             'chirpPerF0': chirpPerF0,
                             'minF0search': minF0search,
                             'maxF0search': maxF0search,}
                             # __c quoi ca...__ 'chirpPerF02072': chirpPerF0}
        self.nbComps = nbComps
        self.nbNMFResComps = nbNMFResComps
        self.nbFilterComps = nbFilterComps
        if len(nbFilterWeigs) < self.nbComps - 1:
            self.nbFilterWeigs = [nbFilterWeigs[0],] * self.nbComps
        else:
            self.nbFilterWeigs = nbFilterWeigs
            
        # initialize the spatial_ranks, reformating here.
        # 20130611 TODO check that it does not break too much everywhere!
        self.spatial_rank = np.atleast_1d(spatial_rank)
        if self.spatial_rank.size < self.nbComps:
            self.spatial_rank = [self.spatial_rank[0],] * self.nbComps
        
        # the source dictionary is shared among all the components,
        # so storing it one for all:
        self.F0Table, WF0, trfoBis = (
            SLS.slf.generate_WF0_TR_chirped(
                transform=self.tft,
                minF0=minF0, maxF0=maxF0,
                stepNotes=stepnoteF0,
                Ot=0.5, perF0=chirpPerF0, 
                depthChirpInSemiTone=0.5, loadWF0=True,
                verbose=self.verbose,)
            )
        
        # removing patterns in low energy bins - setting to eps:
        for nwf0comp in range(WF0.shape[1]): 
            indLowEnergy = np.where(WF0[:,nwf0comp]<WF0[:,nwf0comp].max()*1e-4)
            WF0[indLowEnergy, nwf0comp] = eps
        self.sourceFreqComps = (
            np.ascontiguousarray(
            np.hstack([WF0[:self.nbFreqsSigRepr],
                       np.vstack(np.ones(self.nbFreqsSigRepr))]))
            )
        del WF0
        self.nbSourceComps = self.sourceFreqComps.shape[1]
        self.sourceFreqWeights = np.eye(self.nbSourceComps)
        # ... and the same for the filter part
        self.filterFreqComps = (
            generateHannBasis(
                numberFrequencyBins=self.nbFreqsSigRepr,
                sizeOfFourier=self.sig_repr_params['fsize'],
                Fs=self.audioObject.samplerate,
                frequencyScale='linear',
                numberOfBasis=self.nbFilterComps)
            )
        self.sparsity = sparsity
        self._initialize_structures()
    
    def _initialize_structures(self, seed=None):
        """initialize the structures for the model.
        """
        np.random.seed(seed) # essential for DEBUG
        self.rank = self.spatial_rank
        nc = self.audioObject.channels
        sparsity = self.sparsity
        
        self.spat_comps = {}
        self.spec_comps = {}
        for j in range(self.nbComps - 1):
            # initialize the spatial component
            self.spat_comps[j] = {}
            self.spat_comps[j]['time_dep'] = 'indep'
            self.spat_comps[j]['mix_type'] = 'inst'
            self.spat_comps[j]['frdm_prior'] = 'free'
            self.spat_comps[j]['params'] = np.random.randn(nc, self.rank[j])
            if nc == 2: # spreading the sources evenly for init on stereo
                self.spat_comps[j]['params'] = (
                    np.array([np.sin((j+1) * np.pi / (2.*(self.nbComps))) +
                              np.random.randn(self.rank[j])*np.sqrt(0.01),
                              np.cos((j+1) * np.pi / (2.*(self.nbComps))) +
                              np.random.randn(self.rank[j])*np.sqrt(0.01)]))
            
            # initialize source factor spectral component
            self.spec_comps[j] = {}
            self.spec_comps[j]['spat_comp_ind'] = j
            self.spec_comps[j]['factor'] = {}
            self.spec_comps[j]['factor'][0] = {}
            self.spec_comps[j]['factor'][0]['FB'] = self.sourceFreqComps
            self.spec_comps[j]['factor'][0]['FW'] = self.sourceFreqWeights
            self.spec_comps[j]['factor'][0]['TW'] = (
                0.75 * np.abs(np.random.randn(self.nbSourceComps,
                                              self.nbFramesSigRepr)) +
                0.25)
            self.spec_comps[j]['factor'][0]['TB'] = []
            self.spec_comps[j]['factor'][0]['FB_frdm_prior'] = 'fixed'
            self.spec_comps[j]['factor'][0]['FW_frdm_prior'] = 'fixed'
            self.spec_comps[j]['factor'][0]['TW_frdm_prior'] = 'free'
            self.spec_comps[j]['factor'][0]['TB_frdm_prior'] = []
            self.spec_comps[j]['factor'][0]['TW_constr'] = 'NMF'
            
            # initialize filter factor spectral components
            self.spec_comps[j]['factor'][1] = {}
            self.spec_comps[j]['factor'][1]['FB'] = self.filterFreqComps
            self.spec_comps[j]['factor'][1]['FW'] = (
                0.75 * np.abs(np.random.randn(self.nbFilterComps,
                                              self.nbFilterWeigs[j])) +
                0.25)
            self.spec_comps[j]['factor'][1]['TW'] = (
                0.75 * np.abs(np.random.randn(self.nbFilterWeigs[j],
                                              self.nbFramesSigRepr)) +
                0.25)
            self.spec_comps[j]['factor'][1]['TB'] = []
            self.spec_comps[j]['factor'][1]['FB_frdm_prior'] = 'fixed'
            self.spec_comps[j]['factor'][1]['FW_frdm_prior'] = 'free'
            self.spec_comps[j]['factor'][1]['TW_frdm_prior'] = 'free'
            self.spec_comps[j]['factor'][1]['TB_frdm_prior'] = []
            self.spec_comps[j]['factor'][1]['TW_constr'] = 'NMF'
            
        # residual component:
        self.resSpatialRank = self.rank[-1]#2
        j = self.nbComps - 1
        # initialize the spatial component
        self.spat_comps[j] = {}
        self.spat_comps[j]['time_dep'] = 'indep'
        self.spat_comps[j]['mix_type'] = 'inst'
        self.spat_comps[j]['frdm_prior'] = 'free'
        self.spat_comps[j]['params'] = np.random.randn(nc, self.resSpatialRank)
        # 20120920 trying no initialization for residual:
        ##if nc == 2: # spreading the sources evenly for init on stereo
        ##    self.spat_comps[j]['params'] = (
        ##        np.array([np.sin((j+1) * np.pi / (2.*(self.nbComps + 1))) +
        ##                np.random.randn(self.resSpatialRank)*np.sqrt(0.01),
        ##                np.cos((j+1) * np.pi / (2.*(self.nbComps + 1))) +
        ##                np.random.randn(self.resSpatialRank)*np.sqrt(0.01)]))
            
        # initialize single factor spectral component
        self.spec_comps[j] = {}
        self.spec_comps[j]['spat_comp_ind'] = j
        self.spec_comps[j]['factor'] = {}
        self.spec_comps[j]['factor'][0] = {}
        self.spec_comps[j]['factor'][0]['FB'] = (
            0.75 * np.abs(np.random.randn(self.nbFreqsSigRepr,
                                          self.nbNMFResComps)) +
            0.25)
        self.spec_comps[j]['factor'][0]['FW'] = (
            np.eye(self.nbNMFResComps))
        self.spec_comps[j]['factor'][0]['TW'] = (
            0.75 * np.abs(np.random.randn(self.nbNMFResComps,
                                          self.nbFramesSigRepr)) +
            0.25)
        self.spec_comps[j]['factor'][0]['TB'] = []
        self.spec_comps[j]['factor'][0]['FB_frdm_prior'] = 'free'
        self.spec_comps[j]['factor'][0]['FW_frdm_prior'] = 'fixed'
        self.spec_comps[j]['factor'][0]['TW_frdm_prior'] = 'free'
        self.spec_comps[j]['factor'][0]['TB_frdm_prior'] = []
        self.spec_comps[j]['factor'][0]['TW_constr'] = 'NMF'
        
        if sparsity is None or len(sparsity) not in (1, self.nbComps):
            for j in range(self.nbComps):
                self.spec_comps[j]['sparsity'] = False
        elif len(sparsity) == self.nbComps:
            # sparsity induces a "sparse" activation of
            # self.spec_comps[j]['factor'][0]['TW'], that is,
            # the time weights for the source part.
            # This is implemented as in:
            # Durrieu, J.-L. & Thiran, J.-P.
            #    Sparse Non-Negative Decomposition Of Speech Power Spectra For
            #    Formant Tracking
            # in proc. of the IEEE International Conference on Acoustics,
            # Speech and Signal Processing, Pragues, Czech Republic, 2011.
            #
            # This means that at each GEM iteration, the TW coefficients
            # are further shrinked down to be concentrating around a
            # single component (a single F0 in SF model)
            for j in range(self.nbComps):
                self.spec_comps[j]['sparsity'] = sparsity[j]
        else:
            for j in range(self.nbComps):
                self.spec_comps[j]['sparsity'] = sparsity[0]
        
        self.renormalize_parameters()
        
    def initSpecCompsWithLabelAndFiles(self, instrus=[], instru2modelfile={},
                                       freqBasisAdaptive='fixed'):
        """Initialize the spectral components with the instrument labels as
        well as with the components stored in the provided dictionary in
        `instru2modelfile`
        
        `instrus` is a list with labels:
            `'SourceFilter'`:
                keep the intialized source filter model
            `'Free_<nb_comp>'`:
                initialize the model with an adaptable
                spectral component using `nb_comp` elements in the NMF
                frequency basis
                
            `<key_in_instru2modelfile>`:
                initialize with the :py:class:GSMM
                available and stored in the archive npz with filename
                `instru2modelfile[key_in_instru2modelfile]`
                
        NB: needs the gmm-gsmm module to be installed and in the pythonpath
        """
        instrumentNames = {}
        for n, i in enumerate(instrus):
            instrumentNames[n] = i
            if i == 'SourceFilter':
                self.spec_comps[n]['label'] = i
                print "    Source", n, "left as general Source-Filter model."
            elif 'Free' in i: # assumes Free_nbNMFComps
                nbNMFComps = int(i.split('_')[-1])
                print "    Source", n, "set as free NMF source."
                # initialize single factor spectral component
                self.spec_comps[n] = {}
                self.spec_comps[n]['label'] = i
                self.spec_comps[n]['spat_comp_ind'] = n
                self.spec_comps[n]['factor'] = {}
                self.spec_comps[n]['factor'][0] = {}
                self.spec_comps[n]['factor'][0]['FB'] = (
                    0.75 * np.abs(np.random.randn(self.nbFreqsSigRepr,
                                                  nbNMFComps)) +
                    0.25)
                self.spec_comps[n]['factor'][0]['FW'] = (
                    np.eye(nbNMFComps))
                self.spec_comps[n]['factor'][0]['TW'] = (
                    0.75 * np.abs(np.random.randn(nbNMFComps,
                                                  self.nbFramesSigRepr)) +
                    0.25)
                self.spec_comps[n]['factor'][0]['TB'] = []
                self.spec_comps[n]['factor'][0]['FB_frdm_prior'] = 'free'
                self.spec_comps[n]['factor'][0]['FW_frdm_prior'] = 'fixed'
                self.spec_comps[n]['factor'][0]['TW_frdm_prior'] = 'free'
                self.spec_comps[n]['factor'][0]['TB_frdm_prior'] = []
                self.spec_comps[n]['factor'][0]['TW_constr'] = 'NMF'
                # sparsity stuff
                sparsity = self.sparsity
                if sparsity is None or len(sparsity) not in (1, self.nbComps):
                    self.spec_comps[n]['sparsity'] = False
                elif len(sparsity) == self.nbComps:
                    self.spec_comps[n]['sparsity'] = sparsity[n]
                else:
                    self.spec_comps[n]['sparsity'] = sparsity[0]
            else: #if i != 'SourceFilter':
                print "    Source", n, "is", i
                modelfile = instru2modelfile[i]
                struc = np.load(modelfile)
                gsmm = struc['gsmm'].tolist()
                # Keeping only spectra that are not flat:
                decisionSpectra = np.any(np.diff(gsmm.sigw, axis=1)!=0, axis=1)
                # keeping only the spectra with enough weight:
                #     hard decision, remove all spectra with w == min(w)
                # decisionOnWeight = np.where(gsmm.w!=gsmm.w.min())[0]
                #     harder decision: remove all with w under a threshold:
                decisionOnWeight = (gsmm.w > gsmm.w.max()*1e-3)
                
                keepIndex = np.where(decisionSpectra+decisionOnWeight)[0]
                
                FB = np.ascontiguousarray(gsmm.sigw[keepIndex].T)
                #self.setSpecCompFB(compNb=n, FB=FB, FB_frdm_prior='fixed')
                self.setSpecCompFB(compNb=n, FB=FB,
                                   FB_frdm_prior=freqBasisAdaptive)
                self.spec_comps[n]['label'] = i
                struc.close()
        
        return instrumentNames
        
    def setSpecCompFB(self, compNb, FB, FB_frdm_prior='fixed',):
        """SetSpecCompFB
        
        sets the spectral component's frequency basis.
        
        """
        speccomp = self.spec_comps[compNb]['factor'][0]
        if self.nbFreqsSigRepr != FB.shape[0]:
            raise AttributeError("Size of provided FB is not consistent"+
                                 " with inner attributes")
        speccomp['FB'] = np.copy(FB)
        ncomp = FB.shape[1]
        
        speccomp['FW'] = np.eye(ncomp)
        speccomp['TW'] = (
                0.75 * np.abs(np.random.randn(ncomp,
                                              self.nbFramesSigRepr)) +
                0.25)
        speccomp['FB_frdm_prior'] = FB_frdm_prior
    
    def initializeFreeMats(self, niter=10):
        """initialize free matrices, with NMF decomposition
        """
        # we initialize the matrices with NMF decomposition using the
        # source matrix as basis W, the residual is left uninitialized
        nc = self.audioObject.channels
        # computing the signal representation
        Cx = np.copy(np.real(self.Cx[0]))
        for chan in range(1, nc):
            # stored an "efficient" way, so index "complicated":
            index = np.sum(np.arange(nc, nc-chan, -1))
            Cx += np.real(self.Cx[index])
        
        Cx /= np.double(nc)
        
        # computing NMF of Cx:
        W, H = NMF_decomp_init(SX=Cx,
                               Winit=np.dot(
                                   self.sourceFreqComps,
                                   self.spec_comps[0]['factor'][0]['FW']),
                               verbose=self.verbose,
                               nbComps=self.nbSourceComps,
                               niter=niter,
                               updateW=False, updateH=True,
                               )
        
        for ncomp in range(self.nbComps-1):
            spec_comp = self.spec_comps[ncomp]
            spec_comp['factor'][0]['TW'][:] = np.copy(H) / (self.nbComps-1)
            
        self.renormalize_parameters()
    
    def makeItConvolutive(self):
        """Takes the spatial parameters and sets them to a convolutive
        mixture, in case the parameter has not yet been changed to
        'conv' mode.
        """
        nc = self.audioObject.channels
        for nspat, (spat_ind, spat_comp) in enumerate(self.spat_comps.items()):
            if spat_comp['mix_type'] != 'inst':
                warnings.warn("Spatial component %d "%spat_ind+
                              "already not instantaneous, skipping...")
            else:
                # spat_comp['time_dep'] = 'indep'
                spat_comp['mix_type'] = 'conv'
                # spat_comp['frdm_prior'] = 'free'
                spat_comp_param_inst = spat_comp['params']
                spat_comp['params'] = np.zeros([self.rank[nspat],
                                                nc,
                                                self.nbFreqsSigRepr],
                                               dtype=np.complex)
                for f in range(self.nbFreqsSigRepr):
                    spat_comp['params'][:,:,f] = (
                        np.atleast_2d(spat_comp_param_inst.T))
    
    def estim_param_a_post_model(self,):
        """estim_param_a_post_model
        
        Estimation of model parameters, using the sparsity constraints.
        """
        
        logSigma0 = np.log(np.max([spec['factor'][0]['TW'].shape[0]
                                   for spec in self.spec_comps.values()])**2)
        logSigmaInf = np.log(9.0)
        
        logliks = np.ones(self.iter_num)
        
        if self.noise['sim_ann_opt'] in ['ann', ]:
            self.noise['PSD'] = self.noise['ann_PSD_lim'][0]
        elif self.noise['sim_ann_opt'] is 'no_ann':
            self.noise['PSD'] = self.noise['ann_PSD_lim'][1]
        else:
            warnings.warn("To add noise to the signal, provide the "+
                          "sim_ann_opt from any of 'ann', "+
                          "'no_ann' or 'ann_ns_inj' ")
            
        for i in range(self.iter_num):
            if self.verbose:
                print "Iteration", i+1, "on", self.iter_num
            # adding the noise psd if required:
            if self.noise['sim_ann_opt'] in ['ann', 'ann_ns_inj']:
                self.noise['PSD'] = (
                    (np.sqrt(self.noise['ann_PSD_lim'][0]) *
                     (self.iter_num - i) +
                     np.sqrt(self.noise['ann_PSD_lim'][1]) * i) /
                    self.iter_num) ** 2
                
            # running the GEM iteration:
            logliks[i] = self.GEM_iteration()
            if self.verbose:
                print "    log-likelihood:", logliks[i]
                if i>0:
                    print "        improvement:", logliks[i]-logliks[i-1]
                    
            # sparsity
            sigma = np.exp(logSigma0 +
                           (logSigmaInf - 
                            logSigma0) / 
                           max(self.iter_num - 1.0, 1.) * i)
            self.reweigh_sparsity_constraint(sigma)
            
        return logliks
    
    def reweigh_sparsity_constraint(self, sigma):
        """reweigh_sparsity_constraint
        """
        if self.verbose>1:
            print "reweigh_sparsity_constraint:"
            print "    sigma", sigma
        for j in range(self.nbComps):
            spec_comp = self.spec_comps[j]
            if spec_comp['sparsity'] and \
                   spec_comp['factor'][0]['TW'].shape[0]>2:
                TW = spec_comp['factor'][0]['TW']
                K = TW.shape[0]
                # barycenter from energy of factor 0 TW component
                muTW = (
                    np.dot(np.arange(K - 1) * 
                           (np.arange(K - 1, 0, -1))**2, 
                           TW[:-1,:]) / 
                    np.dot((np.arange(K - 1, 0, -1))**2,
                           np.maximum(TW[:-1,:], eps))
                    )
                # smoothing the sequence:
                muTW  = st.medianFilter(muTW, length=spec_comp['sparsity'])
                if self.verbose>1:
                    print "        muTW NaNs in comp %d:" %j,
                    print np.any(np.isnan(muTW))
                
                twmask = (
                    np.exp(- 0.5 *
                           ((np.vstack(np.arange(K)) - muTW)**2) /
                           sigma)
                    )
                twmask[-1] = twmask.max(axis=0)
                twmask[:,twmask[-1]>0] /= twmask[-1][twmask[-1]>0]
                TW *= twmask

class multichanLead(multiChanSourceF0Filter):
    """Multiple Channel Source Separation, with Lead/Accompaniment initial
    separation

    This instantiation of :class:`multiChanSourceF0Filter` provides convenient
    methods (:func:`multichanLead.runDecomp` for instance) to separate the
    lead instrument from the accompaniment, as in [Durrieu2011]_, and
    then use the obtained parameters/signals in order to initialize the more
    general source separation algorithm.

    Tentative plan for estimation:
        
        1) estimate the Lead/Accompaniment using SIMM

        2) estimate the spatial parameters for each of the separated signals

        3) plug the SIMM params and the spatial params into pyFASST, and

        4) re-estimate

        5) write the estimated signals and enjoy success!

    NB: as for now, the sole Lead/Accompaniment separation achieves better
    separation than the combination of all the possibilities, probably
    because of a more flexible framework for the former than for the latter.
    Some results have been published at the
    `SiSEC <http://sisec.wiki.irisa.fr>`_ 2013 evaluation campaign.
    
    """
    def __init__(self, *args, **kwargs):
        """multichanLead
        
        subclasses multiChanSourceF0Filter
        
        Provides additional methods to estimate the lead/accompaniment parameters
        meant to be used as initial parameters for one of the sources.
        
        """
        super(multichanLead, self).__init__(*args, **kwargs)
        # removing some data from the object, recomputing when needed:
        del self.Cx
        del self.spat_comps
        ##del self.spec_comps
        
    def runDecomp(self, instrus=[],
                  instru2modelfile={},
                  dir_results='tmp/', maxFrames=4000,
                  niter_nmf=20, niter_simm=30):
        """Running the scheme that should make me famous.
        """
        # checking the folder for results
        if not os.path.isdir(dir_results):
            os.mkdir(dir_results)
        
        # running some checks that the input is alright:
        for i in instrus:
            if not(i=='SourceFilter' or
                   i in instru2modelfile or
                   i.startswith("Free_")):
                raise ValueError('Instrument %s not known.' %i)
        
        # just running everything in __init__:
        # estimating the separated 
        self.estimSUIMM(maxFrames=maxFrames,
                        dir_results=dir_results,
                        simmIterNum=niter_simm)
        
        ##############
        # entering vacuum of nightmare of research trial and errors...
        # thus expect many undesirable commented lines...
        
        # putting everything in the right containers:
        self.comp_transf_Cx()
        self._initialize_structures()
        self.makeItConvolutive()
        
        # running DEMIX:
        ## 20130604 no need anymore, only for ALead:
        ALead, AAccp = self.demixOnSepSIMM(unvoiced=True)
        #   spatial components:
        #    accompaniment parameters:
        ## THE FOLLOWING SEEMS TO LEAD TO ISSUES and results not so good...
        ## 20130604 do this after initialize with NMF...
        ## for j in range(1, self.nbComps-1):
        ##     for r in range(self.rank):
        ##       ## the following assumes the instruments are sorted in the
        ##       ## right order, but we still need to think about that !
        ##       # self.spat_comps[j]['params'][r][:,:] = AAccp[j-1].T
        ##       # so for now, we just go for the sum of all the mixing params
        ##       self.spat_comps[j]['params'][r][:,:] = AAccp.sum(axis=0).T
        ## Trying randomized init:
        self.initializeConvParams(initMethod='rand')
        #    no modif for noise component...
        #    lead instrument spatial mat:
        for r in range(self.rank[0]):
            self.spat_comps[0]['params'][r][:,:] = ALead[0].T
        
        #   spectral components:
        
        ## Using the instrument models to initialize the matrices:
        # For convenience, we do this in a separate method:
        instrumentNames = self.initSpecCompsWithLabelAndFiles(
            instrus=instrus,
            instru2modelfile=instru2modelfile,
            freqBasisAdaptive='fixed')
        ## instrumentNames = {}
        ## for n, i in enumerate(instrus):
        ##     instrumentNames[n] = i
        ##     if i == 'SourceFilter':
        ##         print "    Source", n, "left as general Source-Filter model."
        ##     elif 'Free' in i: # assumes Free_nbNMFComps
        ##         nbNMFComps = int(i.split('_')[-1])
        ##         print "    Source", n, "set as free NMF source."
        ##         # initialize single factor spectral component
        ##         self.spec_comps[n] = {}
        ##         self.spec_comps[n]['spat_comp_ind'] = n
        ##         self.spec_comps[n]['factor'] = {}
        ##         self.spec_comps[n]['factor'][0] = {}
        ##         self.spec_comps[n]['factor'][0]['FB'] = (
        ##             0.75 * np.abs(np.random.randn(self.nbFreqsSigRepr,
        ##                                           nbNMFComps)) +
        ##             0.25)
        ##         self.spec_comps[n]['factor'][0]['FW'] = (
        ##             np.eye(nbNMFComps))
        ##         self.spec_comps[n]['factor'][0]['TW'] = (
        ##             0.75 * np.abs(np.random.randn(nbNMFComps,
        ##                                           self.nbFramesSigRepr)) +
        ##             0.25)
        ##         self.spec_comps[n]['factor'][0]['TB'] = []
        ##         self.spec_comps[n]['factor'][0]['FB_frdm_prior'] = 'free'
        ##         self.spec_comps[n]['factor'][0]['FW_frdm_prior'] = 'fixed'
        ##         self.spec_comps[n]['factor'][0]['TW_frdm_prior'] = 'free'
        ##         self.spec_comps[n]['factor'][0]['TB_frdm_prior'] = []
        ##         self.spec_comps[n]['factor'][0]['TW_constr'] = 'NMF'
        ##         # sparsity stuff
        ##         sparsity = self.sparsity
        ##         if sparsity is None or len(sparsity) not in (1, self.nbComps):
        ##             self.spec_comps[n]['sparsity'] = False
        ##         elif len(sparsity) == self.nbComps:
        ##             self.spec_comps[n]['sparsity'] = sparsity[n]
        ##         else:
        ##             self.spec_comps[n]['sparsity'] = sparsity[0]
        ##     else: #if i != 'SourceFilter':
        ##         print "    Source", n, "is", i
        ##         modelfile = instru2modelfile[i]
        ##         struc = np.load(modelfile)
        ##         gsmm = struc['gsmm'].tolist()
        ##         # Keeping only spectra that are not flat:
        ##         decisionSpectra = np.any(np.diff(gsmm.sigw, axis=1)!=0, axis=1)
        ##         # keeping only the spectra with enough weight:
        ##         #     hard decision, remove all spectra with w == min(w)
        ##         # decisionOnWeight = np.where(gsmm.w!=gsmm.w.min())[0]
        ##         #     harder decision: remove all with w under a threshold:
        ##         decisionOnWeight = (gsmm.w > gsmm.w.max()*1e-3)
        ##         
        ##         keepIndex = np.where(decisionSpectra+decisionOnWeight)[0]
        ##         
        ##         FB = np.ascontiguousarray(gsmm.sigw[keepIndex].T)
        ##         #self.setSpecCompFB(compNb=n, FB=FB, FB_frdm_prior='fixed')
        ##         self.setSpecCompFB(compNb=n, FB=FB, FB_frdm_prior='free')
        ##         struc.close()
        
        suffix = dict(instrumentNames)
        # suffix[len(suffix)] = ''
        if self.verbose>1:
            print 'suffix', suffix
        
        self.renormalize_parameters()
        
        # initialize parameters with NMF:
        # putting the HF0 from the SIMM model back in:
        #    lead instrument
        ##self.spec_comps[0]['factor'][0]['TW'][:-1] = (
        ##    self.simmModel.SIMMParams['HF00'])
        startincqt = np.sort(np.where(self.tft.time_stamps>=0)[0])[0]
        stopincqt = (
            startincqt + self.simmModel.SIMMParams['HF00'].shape[1])
        self.spec_comps[0][
            'factor'][0]['TW'][:-1, startincqt:stopincqt] = (
            self.simmModel.SIMMParams['HF00'])
        
        self.initialize_all_spec_comps_with_NMF(updateFreqBasis=False,
                                                niter=niter_nmf)
        # putting the HF0 from the SIMM model back in:
        #    lead instrument
        self.spec_comps[0][
            'factor'][0]['TW'][:-1, startincqt:stopincqt] = (
            self.simmModel.SIMMParams['HF00'])
        ##self.spec_comps[0]['factor'][0]['TW'][:-1] = (
        ##    self.simmModel.SIMMParams['HF00'])
        #        the following are too variable to be kept for now:
        #self.spec_comps[0]['factor'][1]['TW'][:-1] = (
        #    self.simmModel.SIMMParams['HPHI'])
        #self.spec_comps[0]['factor'][1]['FW'][:-1] = (
        #    self.simmModel.SIMMParams['HGAMMA'])
        #self.spec_comps[0]['factor'][1]['FB'][:-1] = (
        #    self.simmModel.SIMMParams['WGAMMA'])
        
        #    accompaniment: nothing for now.
        #    accompaniment: avoid or reduce effect of stuff in source 0, maybe:
        for j in range(1,self.nbComps-1):
            if instrumentNames[j] == 'SourceFilter':
                self.spec_comps[j]['factor'][0]['TW'][
                    :-1, startincqt:stopincqt] = 1.*(
                    self.simmModel.SIMMParams['HF00']==0)
        #    noise: nothing for now.
        
        ## 20130605T0104
        ##    Should we iterate a sequence of (estim_param_a_post_model, demix)
        ##    here? 
        
        # separate the files with these parameters:
        self.renormalize_parameters()
        self.separate_spat_comps(dir_results=dir_results,
                                 suffix=suffix)
        
        if self.verbose>1:
            print suffix
        
        # replace this with method:
        # run DEMIX on the separated files:
        ##estFiles = self.files['spat_comp']
        ##nbSources = len(self.spat_comps)
        ##if self.verbose>1:
        ##    print nbSources, "sources:", estFiles
        ##for nest, estfilename in enumerate(estFiles):
        ##    if self.verbose>1:
        ##        print estfilename
        ##    A = self.demixOnGivenFile(estfilename, nsources=1)
        ##    for r in range(self.rank[nest]):
        ##        self.spat_comps[nest]['params'][r][:,:] = (
        ##            A[0].T + 1e-3 * np.random.randn(*A[0].T.shape))
        ##
        ##self.renormalize_parameters()
        
        estFiles = self.initConvDemixOnSepSrc(suffix)
        
        self.separate_spatial_filter_comp(dir_results=dir_results,
                                          suffix=suffix)
        
        # Re-estimating all the parameters:
        logliks = self.estim_param_a_post_model()
        
        # Separate and Write them...
        if self.verbose:
            print "Writing files to", dir_results
            print self.files
        self.separate_spat_comps(dir_results=dir_results,
                                 suffix=suffix)
        return logliks
    
    def estimSIMM(self, maxFrames=4000, dir_results='tmp/', simmIterNum=30):
        """This method runs the SIMM estimation on the provided audio file.
        
        The lead source is assumed to be self.spec_comps[0]
        """
        
        ##numCompAccomp = (
        ##        np.sum([spec_comp['factor'][0]['FB'].shape[1]
        ##                for ncomp, spec_comp in self.spec_comps.items()])-
        ##        self.spec_comps[0]['factor'][0]['FB'].shape[1]
        ##        )
        numCompAccomp = 40 # TODO: check if this improves solo/acc separation
        if simmIterNum is None:
            simmIterNum = self.iter_num
        
        self.simmModel = SLS.SeparateLeadProcess(
            inputAudioFilename=self.audioObject.filename,
            stepNotes=self.sourceParams['stepnoteF0'],
            chirpPerF0=self.sourceParams['chirpPerF0'],
            nbIter=simmIterNum,
            windowSize=(
                self.sig_repr_params['wlen']/
                np.double(self.audioObject.samplerate)), # in seconds
            hopsize=self.sig_repr_params['hopsize'],
            NFT=self.sig_repr_params['fsize'],
            numCompAccomp=numCompAccomp,
            K_numFilters=self.nbFilterWeigs[0],
            P_numAtomFilters=self.nbFilterComps,
            #imageCanvas=canvas,
            minF0search=self.sourceParams['minF0search'],
            maxF0search=self.sourceParams['maxF0search'],
            minF0=self.sourceParams['minF0'],
            maxF0=self.sourceParams['maxF0'],
            verbose=self.verbose,
            tfrepresentation=self.sig_repr_params['transf'],
            cqtfmax=self.sig_repr_params['tffmax'],#4000,
            cqtfmin=self.sig_repr_params['tffmin'],#50,
            cqtbins=self.sig_repr_params['tfbpo'],#48,
            cqtWinFunc=self.sig_repr_params['tfWinFunc'],
            #slf.minqt.sqrt_blackmanharris,
            cqtAtomHopFactor=self.sig_repr_params['hopfactor'],#0.25,
            outputDirSuffix='tmp/', # dir_results,
            # this is not working, have to find a way
            initHF00='random',
            freeMemory=False)
        
        self.simmModel.autoMelSepAndWrite(maxFrames=maxFrames)
        
    def estimSUIMM(self, maxFrames=4000, **kwargs):
        """separates the audio signal into lead+accompaniment,
        including more noisy components for the lead than `self.estimSIMM`
        """
        if not hasattr(self, "simmModel"):
            self.estimSIMM(maxFrames=maxFrames, **kwargs)
            
        self.simmModel.estimStereoSUIMMParamsWriteSeps(maxFrames=maxFrames)
        
    
    def demixOnSepSIMM(self, unvoiced=True):
        """run DEMIX on the separated signals resulting from SIMM model
        """
        if not hasattr(self, 'simmModel'):
            self.estimSIMM()
            if unvoiced:
                self.estimSUIMM()
                
        if unvoiced:
            suffix = '_VUIMM'
        else:
            suffix = ''
        # DEMIX on lead instrument
        leadfilename = (
            self.simmModel.files['voc_output_file'][:-4] +
            suffix + '.wav')
        ALead = self.demixOnGivenFile(
            leadfilename,
            nsources=1)
        
        # DEMIX on accompaniment
        accpfilename = (
            self.simmModel.files['mus_output_file'][:-4] +
            suffix + '.wav')
        AAccp = self.demixOnGivenFile(
            accpfilename,
            nsources=self.nbComps-2)
        
        return ALead, AAccp
    
    def demixOnGivenFile(self, filename, nsources=1):
        '''running the DEMIX algorithm from :demix.DEMIX:
        
        '''
        maxclusters = 40
        neighbours = 15
        
        # default for demix to work best: #FIXME!!!
        #wlen = 2048
        #hopsize = 1024
        
        demixInst = demix.DEMIX(
            audio=filename,
            nsources=nsources, # spatial comps for demix
            #wlen=wlen,
            #hopsize=hopsize,
            #neighbors=neighbours,
            verbose=self.verbose,
            maxclusters=maxclusters,
            **self.demixParams)
            
        #demixInst.comp_pcafeatures()
        #demixInst.comp_parameters()
        #demixInst.init_subpts_set()
        demixInst.comp_clusters()
        demixInst.refine_clusters()
        
        # mixing parameters from DEMIX estimation:
        #     results in an nsrc x nfreqs x nc array
        A = demixInst.steeringVectorsFromCentroids()
        del demixInst
        if A.size == 0:
            warnMsg = "There are no clusters in demix, returning dummy matrix."
            warnings.warn(warnMsg)
            if self.verbose:
                print warnMsg
            return np.cos(0.25 * np.pi) * np.ones([nsources,
                                                   A.shape[1], A.shape[2]])
        return A
    
    def initConvDemixOnSepSrc(self, suffix):
        """initialize the convolutive parameters with DEMIX, running on each of
        the separated sources
        """
        if not hasattr(self, "files"):
            warnings.warn("The sources were not separated, compute them first"+
                          " with separate_spat_comps.")
            return None
        estFiles = self.files['spat_comp']
        nbSources = len(self.spat_comps)
        if self.verbose>1:
            print nbSources, "sources:", estFiles
        for nest, estfilename in enumerate(estFiles):
            if self.verbose>1:
                print estfilename
            A = self.demixOnGivenFile(estfilename, nsources=1)
            for r in range(self.rank[nest]):
                self.spat_comps[nest]['params'][r][:,:] = (
                    A[0].T + 1e-3 * np.random.randn(*A[0].T.shape))
        
        self.renormalize_parameters()
        
        return estFiles
    
