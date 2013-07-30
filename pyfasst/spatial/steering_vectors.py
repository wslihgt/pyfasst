"""
================
STEERING VECTORS
================
generating steering vectors for arrays of sensors

Content
=======
"""

def gen_steer_vec_far_src_uniform_linear_array(freqs,
                                               nchannels,
                                               theta,
                                               distanceInterMic):
    """generate steering vector with relative far source,
    uniformly spaced sensor array
    
    **Description**:
    
    assuming the source is far (compared to the dimensions of the array)
    The sensor array is also assumed to be a linear array, the direction of
    arrival (DOA) theta is defined as in the following incredible ASCII
    art drawing::
    
          theta
        ----->/              /
        |    /              /
        y   /              /
           /              /
        ^ /              /
        |/              /
        +---> x
        o    o    o    o    o    o
        M1   M2   M3  ...
        <--->
          d = distanceInterMic
    
    That is more likely valid for electro-magnetic fields, for acoustic
    wave fields, one should probably take into account the difference of
    gain between the microphones (see :py:func:`gen_steer_vec_acous` )
    
    **Output**:
    
    a (nc, nfreqs) ndarray
        contains the steering vectors, one for each channel, and
    
    """
    a = np.exp(- 1j * 2. * np.pi *
               np.outer(np.arange(nchannels),
                        freqs) *
               (distanceInterMic / soundCelerity) * 
               np.sin(theta))
    return a

def gen_steer_vec_acous(freqs,
                        dist_src_mic):
    """generates a steering vector for the given frequencies and given
    distances between the microphones and the source.
    
    To the difference with
    :py:func:`gen_steer_vec_far_src_uniform_linear_array`, this function
    also includes gains depending on the distance between the source and
    the mics.
    
    """
    gains = 1 / (np.sqrt(4. * np.pi) * dist_src_mic)
    a = (np.vstack(gains) *
         np.exp(- 1j * 2. * np.pi *
                np.outer(dist_src_mic,
                         freqs) /
                soundCelerity
                )
         )
    return a

def dir_diag_stereo(Cx,
                    nft=2048,
                    ntheta=512,
                    samplerate=44100,#Hz
                    distanceInterMic=0.3,#m
                    ):
    """Compute the diagram of directivity for the input
    short time Fourier transform second order statistics in Cx
    (this Cx is compatible with the attribute from an instantiation
    of :py:class:`pyfasst.audioModel.FASST`)
    
    .. math::
    
        C_x[0] = E[|x_0|^2]
        
        C_x[2] = E[|x_1|^2]
        
        C_x[1] = E[x_0 x_1^H]
        
    
    **Method**:
    
    We use the Capon method, on each of the Fourier channel :math:`k`:
    
    .. math::
    
        \phi_k(\\theta) = a_k(\\theta)^H R_{xx}^{-1} a_k(\\theta)
    
    The algorithm therefore returns one directivity graph for each
    frequency band. 
    
    **Remarks**:
    
    One can compute a summary directivity by adding the directivity functions
    across all the frequency channels. The invert of the resulting array may
    also be of interest (looking at peaks and not valleys to find directions)::
    
     >>> directivity_diag = dir_diag_stereo(Cx)
     >>> summary_dir_diag = 1./directivity_diag.sum(axis=1)
    
    Some tests show that it is very important that the distance between the
    microphone is known. Otherwise, little can be infered from the resulting
    directivity measure...
    """
    nchannels = 2 # this function only works for stereo audio
    
    # for capon, we need the average of Cx:
    meanCx_diag = np.array([Cx[0].mean(axis=1),
                            Cx[2].mean(axis=1)],
                           dtype=np.float64)
    meanCx_off  = Cx[1].mean(axis=1)
    # ... and its inverse:
    inv_mat_diag, inv_mat_off, det_mat = (
        inv_mat(meanCx_diag, meanCx_off)
        )
    
    if not np.all(det_mat):
        raise ValueError(
            "Not possible to compute directivity, singular covariance. "+
            "\nThe channels are probably either identical or colinear.")
    
    nfreqs = nft / 2 + 1
    freqs = np.arange(nfreqs) * 1. / nft * samplerate
    
    # now computing the directivity diagram, angle after angle
    directivity_diagram = np.zeros([ntheta, nfreqs], dtype=np.float64)
    # theta from -pi/2 to +pi/2
    theta = np.arange(1, ntheta+1) * np.pi / (ntheta + 1.) - np.pi / 2.
    for nth in range(ntheta):
        # Compute steering vectors for each frequency
        filt = gen_steer_vec_far_src_uniform_linear_array(freqs,
                                                          nchannels,
                                                          theta[nth],
                                                          distanceInterMic)
        
        directivity_diagram[nth] = (
            (np.abs(filt[0]**2)) * inv_mat_diag[0] +
            (np.abs(filt[1]**2)) * inv_mat_diag[1] +
            2. * np.real((np.conjugate(filt[0]) * filt[1]) *
                         inv_mat_off)
            #filt[0] * np.conjugate(filt[1]) * np.conjugate(inv_mat_off) +
            #np.conjugate(filt[0]) * filt[1] * inv_mat_off
            )
    
    return directivity_diagram, theta

