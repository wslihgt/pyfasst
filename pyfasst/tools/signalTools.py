"""signalTools.py

gathers signal processing tools

"""

import numpy as np
import scipy.signal as spsig
import warnings

eps = 1e-10

def medianFilter(inputArray, length = 10):
    """median filter
    """
    N = inputArray.size
    outputArray = np.zeros_like(inputArray)
    for n in range(N):
        outputArray[n] = np.median(
            inputArray[max(n - length, 0):min(n + length, N - 1)]
            )
        if np.isnan(outputArray[n]):
            outputArray[n] = inputArray[n] 
    return outputArray

def prinComp2D(X0, X1, neighborNb=10, verbose=0):
    """Computes the eigen values and eigen vectors for a
    matrix X of shape 2 x F x N, computing the 2 x 2 covariance matrices
    for the F x N over the temporal neighborhood of size neighborNb.
    
    
    """
    F, N = X0.shape
    F2, N2 = X1.shape
    if F!=F2 or N!=N2:
        raise ValueError("X1 and X2 of size " + str(X0.shape)
                         + ' and ' + str(X1.shape))
    
    lenCov = N - neighborNb + 1
    
    onesNeigh = np.ones(neighborNb) # to compute the means/
    mu0_2 = np.zeros([F, lenCov],)
    mu1_2 = np.zeros([F, lenCov],)
    mu01 = np.zeros([F, lenCov], dtype=np.complex)
    
    cov0_2 = np.zeros([F, lenCov],)
    cov1_2 = np.zeros([F, lenCov],)
    cov01 = np.zeros([F, lenCov], dtype=np.complex)
    
    for f in range(F):
        mu0 = (
            (spsig.fftconvolve(in1=X0[f], in2=onesNeigh,
                               mode='valid')
             )
            ) / neighborNb
        mu1 = (
            (spsig.fftconvolve(in1=X1[f], in2=onesNeigh,
                               mode='valid')
             )
            ) / neighborNb
        # means: in original, equal to 0...
        mu01[f] = mu0 * np.conj(mu1)
        mu0_2[f] = np.abs(mu0)**2
        mu1_2[f] = np.abs(mu1)**2
        
        x_2 = np.abs(X0[f])**2
        cov0_2[f] = (
            np.abs(spsig.fftconvolve(in1=x_2, in2=onesNeigh,
                                     mode='valid')
                   ) #  - neighborNb * mu0_2[f]
            ) / (neighborNb - 1.)
        
        x_2 = np.abs(X1[f])**2
        cov1_2[f] = (
            np.abs(spsig.fftconvolve(in1=x_2, in2=onesNeigh,
                                     mode='valid')
                   ) # - neighborNb * mu1_2[f]
            ) / (neighborNb - 1.)
        
        x_2 = X0[f] * np.conj(X1[f])
        cov01[f] = (
            (spsig.fftconvolve(in1=x_2, in2=onesNeigh,
                               mode='valid')
             ) # - neighborNb * mu01[f]
            ) / (neighborNb - 1.)
        
    traSig = cov0_2 + cov1_2
    detSig = cov0_2 * cov1_2 - np.abs(cov01)**2
    delta = np.sqrt(traSig**2 - 4 * detSig)
    
    # eigen values
    lambdaM = .5 * (traSig + delta)
    lambdam = .5 * (traSig - delta)
    if np.any(lambdaM < 0):
        warnings.warn("lambdaM is negative:" + str(lambdaM))
    if np.any(lambdam < 0):
        warnings.warn("lambdam is negative:" + str(lambdam))
    ##lambdaM = np.copy(lambda0)
    ##lambdam = np.copy(lambda1)
    
    # eigen vectors
    vM = np.zeros([2, F, lenCov], dtype=np.complex)
    vm = np.zeros([2, F, lenCov], dtype=np.complex)
    # 20130613T1512 there seems to be issues with NaNs:
    den = np.sqrt(np.abs(cov01)**2 + np.abs(lambdaM - cov0_2)**2)
    vM[0] = cov01 / den
    vM[1] = (lambdaM - cov0_2) / den
    # setting the den=0 eigen vectors to u_x:
    vM[0][den==0] = 1
    vM[1][den==0] = 0
    # same issue on the small lambda eigenvector:
    den = np.sqrt(np.abs(cov01)**2 + np.abs(lambdam - cov0_2)**2)
    vm[0] = cov01 / den
    vm[1] = (lambdam - cov0_2) / den
    vm[0][den==0] = 0
    vm[1][den==0] = 1
    
    return lambdaM, lambdam, vM, vm, #mu0_2, mu1_2, m01, cov0_2, cov1_2, cov01

def invHermMat2D(a_00, a_01, a_11):
    """This inverts a set of 2x2 Hermitian matrices

    better check :py:func:`inv_herm_mat_2d` instead, and replace all
    reference to this by the former.
    """
    det = a_00 * a_11 - np.abs(a_01)**2
    if np.any(det==0):
        warnings.warn("The matrix is probably non invertible! %s"
                      %str(det[det==0]))
    return a_11/det, -a_01/det, a_00/det
    
def inv_herm_mat_2d(sigma_x_diag, sigma_x_off, verbose=False):
    """Computes the inverse of 2D hermitian matrices.

    **Inputs**
    
     `sigma_x_diag`
        ndarray, with (dim of axis=0) = 2

        The diagonal elements of the matrices to invert.
        `sigma_x_diag[0]` are the (0,0) elements and
        `sigma_x_diag[1]` are the (1,1) ones.

     `sigma_x_off`
        ndarray, with the same dimensions as `sigma_x_diag[0]`

        The off-diagonal elements of the matrices, more precisely the
        (0,1) element (since the matrices are assumed Hermitian,
        the (1,0) element is the complex conjugate)

    **Outputs**
    
     `inv_sigma_x_diag`
        ndarray, 2 x shape(sigma_x_off)

        Diagonal elements of the inverse matrices.
        [0] <-> (0,0)
        [1] <-> (1,1)

     `inv_sigma_x_off`
        ndarray, shape(sigma_x_off)

        Off-diagonal (0,1) elements of the inverse matrices

     `det_sigma_x`
        ndarray, shape(sigma_x_off)

        For each inversion, the determinant of the matrix.

    **Remarks**
    
     The inversion is done explicitly, by computing the determinant
     (explicit formula for 2D matrices), then the elements of the
     inverse with the corresponding formulas.
 
     To deal with ill-conditioned matrices, a minimum (absolute) value of
     the determinant is guaranteed. 

    """
    #if len(sigma_x_diag.shape) != 3:
    #    raise ValueError("Something weird happened to sigma_x")
    det_sigma_x = np.prod(sigma_x_diag, axis=0) - np.abs(sigma_x_off)**2
    if verbose:
        print "number of 0s in det ",(det_sigma_x==0.).sum()
    # issue when det sigma x is 0... 
    det_sigma_x = (
        np.sign(det_sigma_x + eps) *
        np.maximum(np.abs(det_sigma_x), eps))
    if verbose:
        print "number of 0s left in det", (det_sigma_x==0.).sum()
    inv_sigma_x_diag = np.zeros_like(sigma_x_diag)
    inv_sigma_x_off = - sigma_x_off / det_sigma_x
    inv_sigma_x_diag[0] = sigma_x_diag[1] / det_sigma_x
    inv_sigma_x_diag[1] = sigma_x_diag[0] / det_sigma_x

    return inv_sigma_x_diag, inv_sigma_x_off, det_sigma_x

def f0detectionFunction(TFmatrix, freqs=None, axis=None,
                        samplingrate=44100, fouriersize=2048,
                        f0min=80, f0max=3000, stepnote=16,
                        numberHarmonics=20, threshold=0.5,
                        detectFunc=np.sum, weightFreqs=None ,
                        debug=False):
    """Computes the Harmonic Sum

    detectFunc should be a function taking an array as argument, and

    
    `threshold` is homogenous to a tone on the western musical scale
    """
    if axis is None and TFmatrix.ndim == 1:
        axis = 0
    if TFmatrix.ndim == 1:
        nframes = 1
        nfreqs = TFmatrix.size
    else:
        nfreqs, nframes = TFmatrix.shape
        
    if freqs is None:
        # assuming STFT
        freqs = np.arange(nfreqs) * samplingrate / np.double(fouriersize)
        
    if weightFreqs is None:
        weightFreqs = np.ones(nfreqs)
    
    F0number = np.ceil(
        (12. * stepnote) * np.log2(f0max / f0min))
    F0Table = (
        f0min * (
            2. ** (np.arange(F0number)
                   / (12. * stepnote)
                   )
            )
        )
    
    TFmatrixSum = TFmatrix.sum(axis=0)
    
    hs = np.zeros([F0number, nframes])
    for nf0, f0 in enumerate(F0Table):
        ##indexToSum = (
        ##    np.argmin(
        ##        np.abs(12 * np.log2(np.vstack(freqs)
        ##                / (np.arange(1, numberHarmonics + 1) * f0))),
        ##        axis=0))
        ##indexToSum = np.sum(((np.abs(
        ##    12 * np.log2(np.vstack(freqs)
        ##                 / (np.arange(1, numberHarmonics + 1) * f0))))
        ##    < threshold), axis=1) > 0
        #### indexToSum
        ## indexToSum = (
        ##     np.min(
        ##         np.abs(
        ##             12 * np.log2(
        ##                 np.vstack(freqs)
        ##                 / (np.arange(1, numberHarmonics + 1) * f0)
        ##                 )
        ##             ),
        ##         axis=1
        ##         )
        ##     ) < threshold
        
        ##if indexToSum.sum():
        ##    hs[nf0] = detectFunc(np.vstack(weightFreqs[indexToSum])
        ##                         * TFmatrix[indexToSum] / TFmatrixSum,
        ##                         axis=0)
        ##else:
        ##    print "No freq bins for f0:", f0
        
        # reworking this whole by looking for the bin within the allowed ones
        # for which the TFmat is max:
        subTFMat = []
        for nh in range(1, numberHarmonics + 1):
            indexToSum = ((np.abs(
                12 * np.log2(freqs / (nh * f0))))
                                 < threshold)
            if debug:
                print indexToSum.sum()
            if indexToSum.sum() > 1:
                if not len(subTFMat):
                    subTFMat = (
                        np.vstack(weightFreqs[indexToSum])
                        * TFmatrix[indexToSum]).max(axis=0)
                else:
                    if debug:
                        print subTFMat.shape
                        print weightFreqs[indexToSum]
                        print (np.vstack(weightFreqs[indexToSum])
                               * TFmatrix[indexToSum]).max(axis=0)
                    subTFMat = np.vstack(
                        [subTFMat,
                         (np.vstack(weightFreqs[indexToSum])
                          * TFmatrix[indexToSum]).max(axis=0)])
            elif indexToSum.sum() == 1:
                if not len(subTFMat):
                    subTFMat = (weightFreqs[indexToSum]
                                * TFmatrix[indexToSum])
                else:
                    if debug:
                        print subTFMat.shape
                        print (weightFreqs[indexToSum]
                               * TFmatrix[indexToSum]).shape
                    subTFMat = np.vstack(
                        [subTFMat,
                         weightFreqs[indexToSum] * TFmatrix[indexToSum]])
            else:
                print "No freq bins for f0:", f0, "harmo", nh
                
        if len(subTFMat):
            if subTFMat.ndim == 2:
                hs[nf0] = detectFunc(subTFMat / TFmatrixSum, axis=0)
            elif subTFMat.ndim == 1:
                hs[nf0] = detectFunc(np.atleast_2d(subTFMat) / TFmatrixSum,
                                     axis=0)
                
        else:
            print "No input for f0:", f0
    
    return hs, F0Table

def harmonicSum(TFmatrix, **kwargs):
    """Computes the harmonic sum
    """
    return f0detectionFunction(TFmatrix, detectFunc=np.sum, **kwargs)

def harmonicProd(TFmatrix, **kwargs):
    """Computes the harmonic sum
    """
    return f0detectionFunction(TFmatrix, detectFunc=np.prod, **kwargs)

def sortSpectrum(spectrum,
                 numberHarmonicsHS=50,
                 numberHarmonicsHP=1,
                 **kwargs):
    """Sort the spectra in ``spectrum`` with respect to their F0
    values, as estimated by HS * HP function. 
    
    20130521 DJL sort of works, but periodicity detection should be reworked
    according to YIN and the like, in order to obtain better estimates.
    
    """
    hs, f0tablehs = harmonicSum(
        spectrum,
        numberHarmonics=numberHarmonicsHS,
        **kwargs)
    hp, f0tablehp = harmonicProd(
        spectrum,
        numberHarmonics=numberHarmonicsHP,
        **kwargs)
    f0s = f0tablehs[np.argmax(hs * hp, axis=0)]
    indsort = np.argsort(f0s)
    sortedSpectrum = spectrum[:, indsort]
    return sortedSpectrum, f0s[indsort], hs, hp, f0tablehs


