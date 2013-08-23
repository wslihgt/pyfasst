"""DEMIX Python/NumPy implementation

Description
-----------
DEMIX is an algorithm that counts the number of sources,
based on their spatial cues, and returns the estimated parameters,
which are related to the relative amplitudes between the channels,
as well as the relative time shifts. The full description is given
in [Arberet2010]_:

    Arberet, S.; Gribonval, R. & Bimbot, F.
    A Robust Method to Count and Locate Audio Sources in
    a Multichannel Underdetermined Mixture
    IEEE Transactions on Signal Processing, 2010, 58, 121 - 133

This implementation is based on the MATLAB Toolbox provided
by the authors of the above article.

Additionally, this implementation further allows time-frequency
representations other than the short-term Fourier transform (STFT). 

Copyright
---------
Jean-Louis Durrieu, EPFL-STI-IEL-LTS5

jean DASH louis AT durrieu DOT ch

2012-2013

Reference
---------

"""

import numpy as np
import audioObject as ao
import tftransforms.tft as tft
tfts = {'stft': tft.STFT,
        'cqt': tft.CQTransfo,
        'mqt': tft.MinQTransfo,
        'minqt': tft.MinQTransfo,
        }
# to get the PCA for 2D vectors:
import tools.signalTools as st
import warnings

eps = 1e-10

# global variables 
uVarBound = 2.33 # 2.33 # 1.7 #2.33
uDistBound = 2.33 # 2.33 # 1.7 # 2.33

def decibel(val):
    return 20. * np.log10(val)

# alias for above function:
db = decibel

def invdb(dbval):
    return 10. ** (dbval / 20.)

def get_indices_peak(sequence, ind_max, threshold=0.8):
    """returns the indices of the peak around ind_max,
    with values down to  ``threshold * sequence[ind_max]``
    """
    thres_value = sequence[ind_max] * threshold
    # considering sequence is circular, we put ind_max to the center,
    # and find the bounds:
    lenSeq = sequence.size
    ##midSeq = lenSeq / 2
    ##indices = np.remainder(np.arange(lenSeq) - midSeq + ind_max, lenSeq)
    ##
    #newseq = sequence[]
    indSup = ind_max
    indInf = ind_max
    while sequence[indSup]>thres_value:
        indSup += 1
        indSup = np.remainder(indSup, lenSeq)
    while sequence[indInf]>thres_value:
        indInf -= 1
        indInf = np.remainder(indInf, lenSeq)
        
    indices = np.zeros(lenSeq, dtype=bool)
    if indInf < indSup:
        indices[(indInf+1):indSup] = True
    else:
        indices[:indSup] = True
        indices[(indInf+1):] = True
    
    return indices

def confidenceFromVar(variance, neighbors):
    """Computes the confidence, in dB, for a given number of
    neighbours and a variance.
    
    """
    ## JLD: TODO check how this confidence is computed and how it can
    ##be interpreted.
    alpha = (
        (2 * variance * (neighbors - 1) + 1 +
         np.sqrt(1 + 4 * variance * (neighbors - 1)))
        / (2 * variance * (neighbors - 1))
        )
    return 20 * np.log10(alpha)

class DEMIX(object):
    """DEMIX algorithm, for 2 channels.
    """
    def __init__(self, audio,
                 nsources=2,
                 wlen=2048,
                 hopsize=1024,
                 neighbors=20,
                 verbose=0,
                 maxclusters=100,
                 tfrepresentation='stft',
                 tffmin=25, tffmax=18000,
                 tfbpo=48,
                 winFunc=tft.sqrt_blackmanharris):
        """init function: DEMIX for 2 channels
        """
        self.nsources = nsources
        if nsources is None:
            self.maxclusters = maxclusters
        else:
            self.maxclusters = np.maximum(maxclusters, self.nsources*10)
        
        self.verbose = verbose
        self.neighbors = neighbors
        if isinstance(audio, ao.AudioObject):
            self.audioObject = audio
        elif isinstance(audio, str):
            self.audioObject = ao.AudioObject(filename=audio)
        else:
            raise AttributeError("The provided audio parameter is"+
                                 "not a supported format.")
        
        self.sig_repr_params = {}
        self.sig_repr_params['wlen'] = wlen      # window length
        self.sig_repr_params['fsize'] = ao.nextpow2(wlen) # Fourier length
        self.sig_repr_params['hopsize'] = hopsize
        self.sig_repr_params['tfrepresentation'] = tfrepresentation.lower()
        self.sig_repr_params['hopfactor'] = (
            1. * hopsize / self.sig_repr_params['wlen'])
        self.sig_repr_params['tftkwargs'] = {
            'fmin': tffmin, 'fmax': tffmax, 'bins': tfbpo,
            'linFTLen': self.sig_repr_params['fsize'],
            'fs': self.audioObject.samplerate,
            'atomHopFactor': self.sig_repr_params['hopfactor'],
            'winFunc': winFunc
            }
        self.tft = tfts[self.sig_repr_params['tfrepresentation']](
            **self.sig_repr_params['tftkwargs'])
    
    def comp_clusters(self, threshold=0.8):
        """Computes the time-frequency clusters, along with their centroids,
        which contain the parameters of the mixing process - namely `theta`,
        which parameterizes the relative amplitude, and `delta`, which is
        homogeneous to a delay in samples between the two channels.
        """
        self.comp_sig_repr() # the STFT
        self.comp_pcafeatures() # PCA on the neighborhouds
        self.comp_parameters() # extract some parameters from the PCA features
        
        self.init_subpts_set() # initiate the mask of points to cluster to full
        
        # now find the clusters...
        self.clusterCentroids = []
        self.clusters = []
        nbRemainingPoints = self.subTFpointSet.sum()
        if self.verbose:
            print "Computing the clusters... ", nbRemainingPoints, "TF points"
        while nbRemainingPoints and len(self.clusters) < self.maxclusters:
            if self.verbose>4: #DEBUG: tends to be a bit cumbersome
                print "Remaining points:", nbRemainingPoints
            self.clusterCentroids.append(self.get_max_confidence_point())
            if self.verbose>1:
                ## attention: no delta in centroid yet...
                # print "    centroid:", self.clusterCentroids[-1]
                print "    belongs to previous cluster:",
                print np.any([ccc[self.clusterCentroids[-1].index_freq,
                                self.clusterCentroids[-1].index_time] for
                              ccc in self.clusters])
            ind_cluster_theta_pts, distBound = (
                self.getTFpointsNearTheta_OneScale(self.clusterCentroids[-1])
                )
            if self.verbose>1:
                print "comp_clusters:"
                print "    ind_cluster_theta_pts, theta",
                print ind_cluster_theta_pts.sum()
            
            ## already done in above getTFpointsNearTheta method: 
            # ind_cluster_theta_pts *= self.subTFpointSet
            maxDelta = self.identify_deltaT(
                ind_cluster_pts=ind_cluster_theta_pts,
                centroid=self.clusterCentroids[-1],
                threshold=threshold)
            
            if maxDelta is not None:
                self.clusterCentroids[-1].delta = maxDelta
                ind_cluster_theta_pts = (
                    self.getTFPointsNearDelta(
                    centroid=self.clusterCentroids[-1]))
            else:
                self.clusterCentroids[-1].delta = np.NaN
                # keeping only the centroid in that cluster, then:
                ind_cluster_theta_pts *= False
                ind_cluster_theta_pts[self.clusterCentroids[-1].index_freq,
                                self.clusterCentroids[-1].index_time] = True
            
            if self.verbose>1:
                print "comp_clusters:"
                print "    ind_cluster_theta_pts, delta",
                print ind_cluster_theta_pts.sum()
                print "    ***theta***:", self.clusterCentroids[-1].theta
                print "    ***delta***:", maxDelta
            self.clusters.append(ind_cluster_theta_pts)
            if self.verbose>1:
                print "comp_clusters:"
                print "    self.clusters[-1]", self.clusters[-1].sum()
            
            # removing the points that were just classified:
            self.subTFpointSet *= (-ind_cluster_theta_pts)
            
            nbRemainingPoints = self.subTFpointSet.sum()
            
    def refine_clusters(self):
        """Refining the clusters in order to verify that they are possible.
        Additionally, if self.nsources is defined, this method only keeps
        the required number. Otherwise, it is decided by choosing the most
        likely centroids.
        """
        self.reestimate_clusterCentroids()
        if self.nsources is not None:
            # keeping only the best clusters, according to self.nsources
            self.keepBestClusters()
        else:
            # estimate the active number of clusters:
            ##distanceArray = self.get_centroid_distance()
            ##clusterDistBounds = [cc.clusterDistBound() for
            ##                     cc in self.clusterCentroids]
            self.remove_weak_clusters(#distanceArray,
                                      #clusterDistBounds)
                )
            
        self.reestimate_clusterCentroids()
    
    def getTFpointsNearTheta_OneScale(self,
                                      centroid_tfpoint,
                                      ):# index_pts_to_classify # in self....
        """returns the TF points whose theta is close to that of the centroid,
        among the points considered in index_pts_to_classify
        
        TODO: make the function for different scales, as in matlab toolbox
        """
        dist = np.abs(self.thetaAll - centroid_tfpoint.theta)
        ## globally set:
        #uVarBound = 2.33
        #uDistBound = 2.33
        distBound = np.sqrt(
            self.estim_bound_var_theta(centroid_tfpoint.confidence,
                                       infOrSup='sup',
                                       u=uVarBound)
            ) * uDistBound
        
        # conversion to angle on the sphere (???)
        distBound = (
            1 / 2. *
            np.arccos(((distBound**2 - 2)**2) / 2. - 1)
            )
        
        return (dist<distBound) * self.subTFpointSet, distBound
    
    def identify_deltaT(self, ind_cluster_pts, centroid, threshold=0.8):
        """returns the delay maxDelta in samples that corresponds to the
        largest peak of the cluster defined by the provided cluster index
        """
        distBound = centroid.estimDeltaPhi()
        # max nb of samples in error interval:
        nbMaxSamples = 2**7
        # max zoom for oversampling 
        zoomMax = 2**4
        
        if distBound != np.Inf:
            zoom = 2 ** np.ceil(np.log2(nbMaxSamples / 2 * distBound))
            zoom = max(1, zoom)
            zoom = min(zoom, zoomMax)
            zoom = int(zoom)
            
        else:
            zoom = 1
            
        ind_candidates = centroid.getPeakCandidateIndices(zoom=zoom,
                                                          distBound=distBound)
        
        # Computing the detection function, with peaks at the correct delta:
        deltaDetFun = self.compute_temporal(ind_cluster_pts=ind_cluster_pts,
                                            zoom=zoom)
        # invalidating non candidate peaks:
        deltaDetFun[-ind_candidates] = 0 
        # % x-axis values:
        Nft = self.sig_repr_params['fsize']
        deltaAxis = (
            np.concatenate([np.arange(start=0, stop=Nft*zoom/2),
                            np.arange(start=-Nft*zoom/2, stop=0)])
            ) * 1. / zoom
        
        indMaxDelta = np.argmax(deltaDetFun)
        peakDelta = deltaDetFun[indMaxDelta]
        maxDelta = deltaAxis[indMaxDelta]
        indicePeak = get_indices_peak(sequence=deltaDetFun,
                                      ind_max=indMaxDelta,
                                      threshold=threshold)
        if not np.isinf(distBound) and \
               np.any(deltaDetFun[-indicePeak] > threshold*peakDelta):
            maxDelta = None
        
        return maxDelta
    
    def getTFPointsNearDelta(self, centroid):
        """returns a TF mask which is True if their corresponding value of
        `delta` is close enough to the delta from the `centroid`.
        """
        # the reduced frequencies:
        frequencies = self.freqs / self.audioObject.samplerate
        # at first used self.egv0, but actually, that's probably not
        # good, because of some sort of undeterminacies:
        # steering vectors for the TF plane:
        u0 = np.cos(self.thetaAll)
        u1 = np.sin(self.thetaAll) * np.exp(1j * self.phiAll)
        # conjugated v centroid : hence the + in the complex exp
        v_centroid = (
            np.vstack([np.ones(self.freqs.size) * np.cos(centroid.theta),
                       np.exp(1j * 2 * np.pi * centroid.delta * frequencies) *
                       np.sin(centroid.theta)])
            )
        dist = np.sqrt(2 *
                       (1 - np.abs(u0 * np.vstack(v_centroid[0]) +
                                   u1 * np.vstack(v_centroid[1]) ))
                       ##(1 - np.abs(self.egv0[0] * np.vstack(v_centroid[0]) +
                       ##            self.egv0[1] * np.vstack(v_centroid[1]) ))
                       # if euclidean distance, should be np.real, no ?
                       )
        distBound = self.estimDAOBound(confidence=centroid.confidence)
        # discarding points that have a higher confidence than centroid (ie
        # previously identified centroids)
        distBound[centroid.confidence<self.confidence] = 0
        ind_cluster_pts = (dist < distBound)
        ind_cluster_pts[centroid.index_freq, centroid.index_time] = True
        return ind_cluster_pts
    
    def compute_temporal(self, ind_cluster_pts, zoom):
        """This computes the inverse Fourier transform of the
        estimated Steering Vectors, weighed by their inverse variance
        
        The result is a detection function that provides peaks at the
        most likely delta - the delay in samples.
        """
        thetaVars = np.copy(self.thetaVarAll)
        thetaVars[thetaVars==0] = eps
        y = (1/thetaVars) * np.exp(1j * self.phiAll)
        # effectively removing points outside the set of cluster points:
        thetaVars[-ind_cluster_pts] = np.Inf
        y[-ind_cluster_pts] = 0
        
        # normalizing each channel
        normalisation = np.sum((1 / thetaVars), axis=1)
        normalisation[normalisation==0] = eps
        
        # the following may not work for representations other than stft:
        if self.sig_repr_params['tfrepresentation'] not in (
            'stft', 'stftold'):
            #raise NotImplementedError(
            #    "The required representation is borken in DEMIX.")
            
            # kernel to replace the Fourier Kernel (slower)
            K = np.vstack([
                np.exp(2j*np.pi*
                       np.arange(self.sig_repr_params['fsize'] * zoom)
                       / self.audioObject.samplerate * fp)
                for fp in self.tft.freq_stamps])
            deltaDetFun = np.abs(np.dot(y.sum(axis=1)/normalisation,
                                        K))
        else:
            deltaDetFun = np.fft.irfft(y.sum(axis=1) / normalisation,
                                       n=self.sig_repr_params['fsize'] * zoom)
        
        return deltaDetFun
    
    def comp_sig_repr(self):
        """Computes the signal representation, stft
        """
        if not hasattr(self.audioObject, '_data'):
            self.audioObject._read()
        
        if self.verbose:
            print ("Computing the chosen signal representation:")
        
        nc = self.audioObject.channels
        if nc != 2:
            raise ValueError("This implementation only deals "+
                             "with stereo audio!")
        
        self.sig_repr = {}
        
        # if more than 1min of signal, take 1 min in the middle
        # better way : sample data so as to take randomly in the signal
        lengthData = self.audioObject.data.shape[0]
        startData = 0
        endData = lengthData
        oneMinLenData = 60*self.audioObject.samplerate
        oneMinLenData *= 2 # or more than 1min?
        
        if lengthData>oneMinLenData:
            startData = (lengthData - oneMinLenData)/2
            endData = startData + oneMinLenData
            
        if self.sig_repr_params['tfrepresentation'] == 'stftold':
            self.sig_repr[0], freqs, times = ao.stft(
                self.audioObject.data[startData:endData,0],
                window=np.hanning(self.sig_repr_params['wlen']),
                hopsize=self.sig_repr_params['hopsize'],
                nfft=self.sig_repr_params['fsize'],
                fs=self.audioObject.samplerate
                )
            
            self.sig_repr[1], freqs, times = ao.stft(
                self.audioObject.data[startData:endData,1],
                window=np.hanning(self.sig_repr_params['wlen']),
                hopsize=self.sig_repr_params['hopsize'],
                nfft=self.sig_repr_params['fsize'],
                fs=self.audioObject.samplerate
                )
        else:
            self.tft.computeTransform(
                self.audioObject.data[startData:endData,0],)
            self.sig_repr[0] = self.tft.transfo
            self.tft.computeTransform(
                self.audioObject.data[startData:endData,1],)
            self.sig_repr[1] = self.tft.transfo
            freqs = self.tft.freq_stamps
                
        # keeping the frequencies, not computing them each time
        self.freqs = freqs
        
        del self.audioObject.data
        
    def comp_pcafeatures(self):
        """Compute the PCA features
        """
        if not(hasattr(self, 'X0')):
            self.comp_sig_repr()
        
        self.lbd0, self.lbd1, self.egv0, self.egv1 = st.prinComp2D(
            X0=self.sig_repr[0],
            X1=self.sig_repr[1],
            neighborNb=self.neighbors)
        
        self.confidence = 20. * np.log10(self.lbd0 /
                                         np.maximum(self.lbd1, eps))
        # issue with NaN ... :
        self.confidence[np.isnan(self.confidence)] = 0. # is that ok?...
        # 20130208 DJL adding a weighting scheme to lower low energy
        # coeff influence:
        start = self.neighbors / 2
        energy = np.mean(
            np.array(self.sig_repr.values()) ** 2,
            axis=0)[:,start:start+self.confidence.shape[1]]
        self.confidence[energy<energy.max()*1e-6] = 0.
        # no need for the signal representation anymore, for now:
        del self.sig_repr
        del self.lbd0
        # del self.egv0 # needed in comp_parameters
        del self.egv1
        del self.lbd1
        
    def comp_parameters(self):
        """comp_parameters
        """
        # removing first 2 bands, avoiding continuous components
        self.confidence[:2,:] = eps
        
        # storing the data points following the chosen parameterization of the
        # steering vectors:
        # 
        #    u_k = [cos(\theta_k) sin(\theta_k) exp(-j 2 \pi f \delta_k)]^T
        #
        # where \theta_k is an intensity parameter (IP),
        #       \delta_k is the time delay between the channels
        #       and k is the source index.
        #
        self.phiAll = np.angle(self.egv0[1] / self.egv0[0])
        self.thetaAll = np.arctan(np.abs(self.egv0[1]) / np.abs(self.egv0[0]))
        self.thetaVarAll = self.estim_var_theta(confidence=self.confidence)
        del self.egv0
        
    def estim_var_theta(self, confidence):
        maxLim_confidence = 3073.
        thetaVarAll = np.ones_like(confidence)
        indNotZeroConf = np.where(confidence>eps)
        if self.verbose>1 and False:
            print "estim_var_theta: indNotZeroConf", indNotZeroConf
        # The confusing equations are because the values are stored as dBs...
        thetaVarAll[indNotZeroConf[0], indNotZeroConf[1]] = (
            10 ** (confidence[indNotZeroConf[0], indNotZeroConf[1]] / 20.) /
            ((self.neighbors-1) *
             (10 ** (confidence[indNotZeroConf[0], indNotZeroConf[1]] / 20.)
              - 1) ** 2)
            )
        thetaVarAll[confidence>=maxLim_confidence] = 0
        thetaVarAll[confidence<=eps] = np.Inf
        return thetaVarAll
    
    def estim_bound_var_theta(self, confidence, infOrSup='sup', u=2.33):
        minConfidence = 0.
        sigma = np.sqrt(1600. / (self.neighbors - 1.))
        
        signInfSup = {'inf': 1, 'sup': -1}
        confidence += signInfSup[infOrSup] * u * sigma
        confidence = np.maximum(confidence, minConfidence)
        confidence = np.atleast_2d(confidence)
        if self.verbose>1 and False:
            print "confidence", confidence
        # the method below returns a 2D array, and therefore,
        # we only return the (only) element of it:
        if np.size(confidence)>1:
            return self.estim_var_theta(confidence)
        else:
            return self.estim_var_theta(confidence)[0,0]
    
    def estimDAOBound(self, confidence, confidenceVal=None):
        """computes the max distance between centroid and points
        """
        maxErrorTheta = self.estim_bound_var_theta(confidence=confidence,
                                                   u=uVarBound)
        if confidenceVal is not None:
            thetaVarAll = self.estim_bound_var_theta(
                confidence=confidenceVal,
                u=uVarBound)
        else:
            thetaVarAll = self.thetaVarAll
            
        if self.verbose>1 and False: # DEBUG, lots of stuff written!
            print "daobound", (np.sqrt(thetaVarAll) +
                               np.sqrt(maxErrorTheta)) * uDistBound
        return (np.sqrt(thetaVarAll) +
                np.sqrt(maxErrorTheta)) * uDistBound
    
    def init_subpts_set(self, ):
        # setting the "mask" for the
        try:
            self.subTFpointSet = np.ones(self.confidence.shape, dtype=bool)
        except AttributeError:
            raise AttributeError("no lbd0 in object!")
        
    def get_max_confidence_point(self, ):
        tmpConfidence = np.copy(self.confidence)
        # it should be a boolean array of the same size as self.confidence:
        tmpConfidence[(- self.subTFpointSet)] = 0
        index = np.unravel_index(tmpConfidence.argmax(), tmpConfidence.shape)
        del tmpConfidence
        return TFPoint(demixinstance=self,
                       index_freq=index[0],
                       index_time=index[1])
    
    def reestimate_clusterCentroids(self):
        """reestimate cluster centroids
        
        considering all the cluster masks, reestimate the centroids,
        discarding the clusters for which there was no well-defined delta.
        """
        if not(hasattr(self, 'clusterCentroids')):
            self.comp_clusters()
        
        # filtering out clusters for which we dont have a correct delay:
        ind_good_cluster = - np.isnan(
            [c.delta for c in self.clusterCentroids])
        if self.verbose>1: #DEBUG
            print "reestimate_clusterCentroids"
            print "    ind_good_cluster", ind_good_cluster.sum()
            print "    [c.delta for c in self.clusterCentroids]"
            print [c.delta for c in self.clusterCentroids]
            print "   cluster sizes:", [c.sum() for c in self.clusters]
        if self.verbose:
            print "reestimate_clusterCentroids"
            print "    constraining the clusters to the good ones"
        self.clusterCentroids = (
            [self.clusterCentroids[n] for n in np.where(ind_good_cluster)[0]])
        self.clusters = (
            [self.clusters[n] for n in np.where(ind_good_cluster)[0]])
        
        if self.verbose:
            print "  reestimate_clusterCentroids: Computing exclusive clusters"
        nbClusters = ind_good_cluster.sum()
        # filtering out the Time-Freq points that are in several clusters:
        self.create_exclusive_clusters()
        
        if self.verbose:
            print ("  reestimate_clusterCentroids: "+
                   "Computing thresholded clusters")
        # adaptive thresholding of clusters
        thresholdedClusters = self.adaptive_thresholding_clusters()
        
        if self.verbose:
            print "  reestimate_clusterCentroids: For each cluster, "
            print "                               reestimate the centroid"
        # estimation of centroids for each cluster
        for n, cluster in enumerate(thresholdedClusters):
            if self.verbose>1:
                print "    cluster", n+1, "of", len(thresholdedClusters)
            if cluster.sum() != 0:
                # only the confidences for the current cluster:
                confidences = self.confidence[cluster]
                thetas = self.thetaAll[cluster]
                variances = self.thetaVarAll[cluster]
                if self.verbose > 1 and False: # DEBUG
                    print "reestim_cluster: confidence", confidences
                varBounds = self.estim_bound_var_theta(confidence=confidences,
                                                       infOrSup='sup',
                                                       u=uVarBound)
                # normalisation coeff:
                varP = 1. / (
                    np.sum(1./variances)
                    )
                # weighted mean of thetas, weight is varP/variance
                theta = varP * np.sum(thetas / variances)
                # confidence of the estimation:
                varBound = 1. / np.sum(1./varBounds)
                clusterConfidence = confidenceFromVar(varBound, self.neighbors)
                
                # updating if the confidence of current estimation is better than
                # the original one:
                # NB even we use thresholdedClusters, the centroids should be
                #   the same as for self.clusters.
                if clusterConfidence > self.clusterCentroids[n].confidence:
                    self.clusterCentroids[n].theta = theta
                    self.clusterCentroids[n].confidence =  clusterConfidence
    
    def create_exclusive_clusters(self):
        """
        create_exclusive_clusters
        
        reconfigures the cluster indices in self.clusters such
        that all the Time-Freq points that appear in more than
        one cluster are dismissed from all computations
        """
        if not(hasattr(self, 'clusters')):
            self.comp_clusters()
        
        # JLD modifying this such that we keep the TF points in the cluster
        # with best confidence.
        if len(self.clusters):
            maskOnlyOneCluster = np.array(self.clusters[0], dtype=int)
        else:
            warnings.warn("create_exclusive_clusters: no clusters anymore!")
        for clustern, cluster in enumerate(self.clusters[1:]):
            maskOnlyOneCluster += self.clusters[clustern+1]
            cluster *= (maskOnlyOneCluster==1)
        
        # original program : remove all common tf points
        ##maskOnlyOneCluster = (np.sum(self.clusters, axis=0)==1)
        ##for c in self.clusters:
        ##    c *= maskOnlyOneCluster
        
        # JDL: adding a control to discard empty clusters
        #    strange thing: this case never appeared in DEMIX in matlab...
        self.remove_empty_clusters()
    
    def get_centroid_distance(self):
        """get_centroid_distance
        
        distance between the centroids
        """
        nbClusters = len(self.clusters)
        nbFreqs = self.sig_repr_params['fsize'] / 2 + 1
        # self.sig_repr[0].shape[0]#
        
        distanceArray = np.zeros([nbClusters, nbClusters])
        
        # frequencies vector:
        f = np.arange(0, nbFreqs) * 1. / self.sig_repr_params['fsize']
        for n in range(nbClusters):
            for m in range(n+1, nbClusters):
                theta = self.clusterCentroids[n].theta
                alpha = self.clusterCentroids[m].theta
                diffDelta = (self.clusterCentroids[n].delta -
                             self.clusterCentroids[m].delta)
                d2 = (2. * (1. -
                            np.mean(1. / np.sqrt(2.) *
                                    np.sqrt(1. +
                                            np.cos(2*alpha) * np.cos(2*theta) +
                                            np.sin(2*alpha) * np.sin(2*theta) *
                                            np.cos(2.*np.pi * f * diffDelta)),
                                    axis=0)
                            )
                      )
                distanceArray[n,m] = np.sqrt(d2)
        return distanceArray
    
    def adaptive_thresholding_clusters(self):
        """compute for each cluster in self.clusters a threshold depending
        on the other clusters, in order to keep only those points in cluster
        that are close to the actual centroid, but not close to centroids of
        other clusters.
        
        The returned clusters are the original clusters thresholded. 
        """
        distanceArray = self.get_centroid_distance()
        nbClusters = len(self.clusters)
        thresholdedClusters = []
        
        confidenceThreshold = np.zeros(nbClusters)
        distTocluster = np.ones(nbClusters) * 2
        
        if self.verbose:
            print "adaptive thresholding the clusters"
        
        for n in range(nbClusters):
            if self.verbose>1:
                print "    cluster", n, "of", nbClusters
                
            clustersMinusN = range(nbClusters)
            clustersMinusN.remove(n)
            for m in clustersMinusN:
                if self.verbose>2 and False: # DEBUG
                    print "[min(n,m), max(n,m)]", [min(n,m), max(n,m)]
                dist_n_m = np.copy(distanceArray[min(n,m), max(n,m)])
                if self.verbose>2 and False: # DEBUG
                    print "dist_n_m", dist_n_m
                # max erreurs sur l'estimation des centroids:
                dist_centroid_n = self.estimDAOBound(
                    confidence=self.clusterCentroids[n].confidence,
                    confidenceVal=np.Inf)
                
                dist_centroid_m = self.estimDAOBound(
                    confidence=self.clusterCentroids[m].confidence,
                    confidenceVal=np.Inf)
                if self.verbose>2 and False:  # DEBUG
                    print "  dist_centroid_n/m",dist_centroid_n,dist_centroid_m
                
                dist_n_m -= (dist_centroid_n + dist_centroid_m)
                dist_n_m = max(dist_n_m , 0.)
                distTocluster[n] = min(dist_n_m, distTocluster[n])
                if self.verbose>2 and False: # DEBUG
                    print "distTocluster[n]", distTocluster[n]
                    if distTocluster[n]==0:
                        print "    nul dist for", [min(n,m), max(n,m)]
            # estimation of threshold
            confidenceThreshold[n] = confidenceFromVar((distTocluster[n]/2)**2,
                                                       self.neighbors)
            
            # self.clusters[n] *= (self.confidence >= confidenceThreshold[n])
            thresholdedClusters.append(
                self.clusters[n] * (self.confidence >= confidenceThreshold[n]))
            
        # to avoid errors when considering empty clusters...
        self.remove_empty_clusters()
        return thresholdedClusters
    
    def keepBestClusters(self, nsources=None):
        if nsources is None:
            nsources = self.nsources
        if nsources is None:
            raise AttributeError('The nb of sources is not provided.')
        nclusters = len(self.clusters)
        if nclusters < nsources:
            warnings.warn("The number of clusters %d" %nclusters +
                          "is different from the required\nnumber of sources" +
                          " %d." %nsources)
            pass
        else:
            confidences = [cc.confidence for cc in self.clusterCentroids]
            sortedIndices = np.argsort(confidences)
            # indices of increasing values sorted: read upside down:
            self.clusterCentroids = (
                [self.clusterCentroids[n] for
                 n in sortedIndices[:-(nsources+1):-1]])
            self.clusters = (
                [self.clusters[n] for
                 n in sortedIndices[:-(nsources+1):-1]]
                )
    
    def remove_empty_clusters(self):
        """remove_empty_clusters
        
        DJL: this did never happen in DEMIX Matlab version, have to contact
        authors for explanations...
        """
        clustersizes = np.array([c.sum() for c in self.clusters])
        if self.verbose>1:
            print "remove_empty_clusters"
            print "    cluster sizes:", clustersizes
            print "    Removing", np.sum(clustersizes==0), "clusters"
        if self.verbose:
            print "    cluster centroids:", self.clusterCentroids
        self.clusters = (
            [self.clusters[n] for n in np.where(clustersizes>0)[0]])
        self.clusterCentroids = (
            [self.clusterCentroids[n] for n in np.where(clustersizes>0)[0]])
    
    def remove_weak_clusters(self, distanceArray=None,
                             clusterDistBounds=None):
        if distanceArray is None:
            distanceArray = self.get_centroid_distance()
            
        if clusterDistBounds is None:
            clusterDistBounds = [cc.clusterDistBound() for
                                 cc in self.clusterCentroids]
            
        # removing clusters that are too close to the others:
        nbClusters = len(self.clusters)
        indGoodClusters = []
        for n in range(nbClusters):
            clustersMinusN = range(nbClusters)
            clustersMinusN.remove(n)
            isGoodCluster = True
            for m in clustersMinusN:
                if self.verbose>1 and False: # DEBUG
                    print "[min(n,m), max(n,m)]", [min(n,m), max(n,m)]
                dist_n_m = np.copy(distanceArray[min(n,m), max(n,m)])
                if (self.clusterCentroids[n].confidence>
                    self.clusterCentroids[m].confidence):
                    dist_n_m = np.Inf
                isGoodCluster *= (dist_n_m>clusterDistBounds[n])
            if isGoodCluster:
                indGoodClusters.append(n)
                
        self.clusters = [self.clusters[n] for n in indGoodClusters]
        self.clusterCentroids = (
            [self.clusterCentroids[n] for n in indGoodClusters])
    
    def spatial_filtering(self):
        """using optimal spatial filters to obtain separated signals
        
        this is a beamformer implementation.
        MVDR or assuming the sources are normal, independent and
        with same variance (not sure whether this does not mean that
        we can't separate them...)
        
        From::
        
         Maazaoui, M.; Grenier, Y. & Abed-Meraim, K.
         ``Blind Source Separation for Robot Audition using
         Fixed Beamforming with HRTFs'', 
         in proc. of INTERSPEECH, 2011.
        
        per channel, the filter steering vector, source p:

        .. math::
        
            b(f,p) = \\frac{R_{aa,f}^{-1} a(f,p)}{a^{H}(f,p) R_{aa,f}^{-1} a(f,p)}
            
        """
        A = self.steeringVectorsFromCentroids()
        nsrc = len(self.clusterCentroids)
        Raa_00 = np.mean(np.abs(A[:,:,0]) ** 2, axis=0)
        Raa_11 = np.mean(np.abs(A[:,:,1]) ** 2, axis=0)
        Raa_01 = np.mean(A[:,:,0] * np.conj(A[:,:,1]), axis=0)
        
        # invert the matrices, in one pass, easy since 2D and hermitian:
        invRaa_00, invRaa_01, invRaa_11 = st.invHermMat2D(Raa_00, Raa_01,
                                                          Raa_11)
        # note: if all steering vectors are the same, then Raa is
        # degenerate, and the result might be unstable...
        
        if not hasattr(self, "sig_repr"):
            self.comp_sig_repr()
        sep_src = []
        for p in range(nsrc):
            # the beamformer filter B is nfreqs x 2
            B = np.vstack([invRaa_00 * A[p,:,0] + invRaa_01 * A[p,:,1],
                           np.conj(invRaa_01) * A[p,:,0] + invRaa_11 * A[p,:,1]])
            B /= ([np.conj(A[p,:,0]) * B[0] +
                   np.conj(A[p,:,1]) * B[1]])
            B = np.conj(B.T)
            # this is not in the formula of the cited paper,
            # but should be theoretically correct... TBC!
            S = (np.vstack(B[:,0]) * self.sig_repr[0] +
                 np.vstack(B[:,1]) * self.sig_repr[1])
            s = []
            # ... and recreating the image given the mixing matrix A
            if self.sig_repr_params['tfrepresentation'] == 'stftold':
                s.append(ao.istft(
                    X=S * np.vstack(A[p,:,0]),
                    window=np.hanning(self.sig_repr_params['wlen']),
                    analysisWindow=None,
                    hopsize=self.sig_repr_params['hopsize'],
                    nfft=self.sig_repr_params['fsize']))
                s.append(ao.istft(
                    X=S * np.vstack(A[p,:,1]),
                    window=np.hanning(self.sig_repr_params['wlen']),
                    analysisWindow=None,
                    hopsize=self.sig_repr_params['hopsize'],
                    nfft=self.sig_repr_params['fsize']))
            else:
                self.tft.transfo = S * np.vstack(A[p,:,0])
                s.append(self.tft.invertTransform())
                self.tft.transfo = S * np.vstack(A[p,:,1])
                s.append(self.tft.invertTransform())
                
            s = np.array(s).T
            sep_src.append(s)
        return sep_src
    
    def steeringVectorsFromCentroids(self):
        """Generates the steering vectors a(p,f,c) for source p,
        (reduced) freq f and channel c.

        .. math::
        
          a[p,f,0] = \\cos(\\theta_p)
          
          a[p,f,1] = \\sin(\\theta_p) \\exp(- 2 j \\pi f \\delta_p)
        
        """
        # TODO: should check that
        #    * clusterCentroids exists,
        #    * nchannels of audioObject is 2
        #    * etc.
        
        if not hasattr(self, "freqs"):
            self.comp_sig_repr()
        
        thetas = [cc.theta for cc in self.clusterCentroids]
        deltas = [cc.delta for cc in self.clusterCentroids]
        # frequencies in reduced form (from 0 to 1/2)
        freqs = self.freqs * 1. / self.audioObject.samplerate
        nsrc = len(self.clusterCentroids)
        
        A = np.zeros([nsrc, self.freqs.size,
                      self.audioObject.channels], dtype=np.complex)
        if nsrc>0:
            # left channel:
            A[:,:,0] = np.vstack(np.cos(thetas))
            # right channel:
            A[:,:,1] = (np.vstack(np.sin(thetas)) *
                        np.exp(- 1j * 2. * np.pi *
                               np.outer(deltas,
                                        freqs)))
        else:
            warnings.warn("Caution: the number of sources is %d." %nsrc)
        return A
    
class TFPoint(object):
    def __init__(self, demixinstance=None, thetaphidelta=None,
                 index_scale=0, index_freq=0, index_time=0,
                 verbose=2):
        
        if demixinstance is None and thetaphidelta is not None:
            self.theta = thetaphidelta[0]
            self.phi = thetaphidelta[1]
            self.delta = thetaphidelta[2]
            self.confidence = 100. 
            self.index_scale = index_scale # for future integration
            self.index_freq = index_freq
            self.index_time = index_time
            self.frequency = 0.
        elif demixinstance is not None:
            self.theta = demixinstance.thetaAll[index_freq,
                                                index_time]
            self.phi = demixinstance.phiAll[index_freq,
                                            index_time]
            self.confidence = demixinstance.confidence[index_freq,
                                                       index_time]
            self.frequency = (
                index_freq * 1. / demixinstance.sig_repr_params['fsize'])
            self.index_scale = index_scale # for future integration
            self.index_freq = index_freq
            self.index_time = index_time
            
            # to be able to use general purpose methods like:
            #    estim_bound_var_theta
            self.demixInst = demixinstance
        else:
            print "TFPoint: generating dummy TFPoint"
        
        self.verbose = verbose
    
    def estimDeltaPhi(self):
        """Estimates a bound on the distance
        confidence interval, in samples, to the TFpoint delta
        """
        ## globally set:
        #uVarBound = 2.33
        #uDistBound = 2.33
        
        dmax = np.sqrt(
            self.demixInst.estim_bound_var_theta(
            confidence=self.confidence,
            infOrSup='sup',
            u=uVarBound)
            ) * uDistBound
        
        if self.verbose>1 and False:
            print "dmax", dmax
        
        cos_ds = (dmax**2 - 2)**2 / 2 - 1 # where does this come from...
        if np.arccos(cos_ds) <= min(2*self.theta, np.pi - 2*self.theta):
            alpha = 0.5 * np.arccos(np.cos(2 * self.theta) / cos_ds)
            deltaPhi = np.arccos(
                (cos_ds - np.cos(2 * self.theta) * np.cos(2 * alpha)) /
                (np.sin(2 * self.theta) * np.sin(2 * alpha))
                )
            return deltaPhi / (2 * np.pi * self.frequency)
        else:
            return np.Inf
    
    def getPeakCandidateIndices(self, zoom=1, distBound=None):
        """computes the indices of compatible delta bins
        """
        if distBound is None:
            _, distBound = (
                self.demixInst.getTFpointsNearTheta_OneScale(
                centroid_tfpoint=self)
                )
        
        Nft = self.demixInst.sig_repr_params['fsize']
        
        kmax = np.floor((2 * np.pi * self.frequency *
                         Nft / 2 -
                         np.abs(self.phi)) /
                        (2 * np.pi))
        
        kmax = np.maximum(kmax, 0)
        ks = np.arange(-kmax, kmax+1)
        centroidDeltas = (
            (- self.phi + 2 * np.pi * ks) /
            (2 * np.pi * self.frequency)
            )
        deltaXiFFTmin = - (Nft * zoom / 2 - 1)
        deltaXiFFTmax = (Nft * zoom / 2)
        
        infBounds = np.zeros(len(ks))
        supBounds = np.zeros(len(ks))
        
        bound1 = np.ceil((centroidDeltas - distBound) * zoom)
        bound2 = np.floor((centroidDeltas + distBound) * zoom)
        
        bound1 = np.maximum(deltaXiFFTmin, bound1)
        bound1 = np.minimum(deltaXiFFTmax, bound1)
        
        bound2 = np.maximum(deltaXiFFTmin, bound2)
        bound2 = np.minimum(deltaXiFFTmax, bound2)
        
        infBounds[bound1>=0] = bound1[bound1>=0]
        infBounds[bound1<0]  = (
            bound1[bound1<0] + deltaXiFFTmax - deltaXiFFTmin + 1)
        
        supBounds[bound2>=0] = bound2[bound2>=0]
        supBounds[bound2<0]  = (
            bound2[bound2<0] + deltaXiFFTmax - deltaXiFFTmin + 1)
        
        if np.isinf(distBound):
            # return a True vector for all time bins:
            self.indices_candidates = np.ones(Nft * zoom, dtype=bool)
        else:
            self.indices_candidates = np.zeros(Nft * zoom, dtype=bool)
            for n in range(len(ks)):
                if infBounds[n] <= supBounds[n]:
                    self.indices_candidates[infBounds[n]:(supBounds[n]+1)]=True
                else:
                    self.indices_candidates[infBounds[n]:] = True
                    self.indices_candidates[:(supBounds[n]+1)] = True
        
        return self.indices_candidates
    
    def clusterDistBound(self, ):
        # this is a bit strange....
        #
        # %u_nbpPoints(1) <=> 200000pts,u_nbpPoints(2) <=> 400000pts...
        u_nbpPoints = [2.33, 2.58, 2.72, 2.81, 2.88, 2.93, 2.98, 3.05]
        indNbPts = 7 # dont get why this choice
        uVarBound = u_nbpPoints[indNbPts]
        uDistBound = u_nbpPoints[indNbPts]
        varBounds = self.demixInst.estim_bound_var_theta(
            confidence=self.confidence,
            infOrSup='sup',
            u=uVarBound)
        return np.sqrt(varBounds) * uDistBound # distBound
    
    def __str__(self):
        return ("[confidence: "+str(self.confidence)+
                ", delta: "+str(self.delta)+", "+
                "phi :"+str(self.phi)+
                ", theta: "+str(self.theta)+", "+
                "freq : "+str(self.frequency)+"]")
    
    def __repr__(self):
        return self.__str__()
