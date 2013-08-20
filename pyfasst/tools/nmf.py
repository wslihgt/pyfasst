"""Simple Nonnegative Matrix Factorization (NMF) routines to be used
to estimate initial parameters in FASST.

Relevant references:

.. [Durrieu2010] J.-L. Durrieu, G. Richard, B. David and C. Fevotte,
   Source/Filter Model for Main Melody Extraction From Polyphonic Audio 
   Signals, IEEE Transactions on Audio, Speech and Language Processing, 
   special issue on Signal Models and Representations of Musical and 
   Environmental Sounds, March 2010, Vol. 18 (3), pp. 564 -- 575.

.. [Fevotte2009] C. Fevotte and N. Bertin and J.-L. Durrieu,
   Nonnegative matrix factorization with the Itakura-Saito divergence.
   With application to music analysis,
   Neural Computation, vol. 21 (3), pp. 793-830, March 2009.
   [`pdf <http://www.unice.fr/cfevotte/publications/journals/neco09_is-nmf.pdf>`_]

"""

import numpy as np

eps = 1e-10

def NMF_decomposition(SX, nbComps=10, niter=10, verbose=0):
    """NMF multiplicative gradient, for Itakura Saito
    divergence measure between SX and ``np.dot(W,H)``
    
    See for instance [Fevotte2009]_.
    """
    freqs, nframes = SX.shape
    W = np.random.randn(freqs, nbComps)**2
    H = np.random.randn(nbComps, nframes)**2
    W /= W.sum(axis=0)
    
    for i in range(niter):
        if verbose:
            print "    NMF iteration %d out of %d" %(i+1, niter)
        # updating W
        hatSX = np.dot(W, H)
        num = np.dot(SX / np.maximum(hatSX**2, eps),
                     H.T)
        den = np.dot(1 / np.maximum(hatSX, eps),
                     H.T)
        
        W *= num / np.maximum(den, eps)
        
        sumW = W.sum(axis=0)
        sumW[sumW==0] = 1.
        W /= sumW
        H *= np.vstack(sumW)
        
        # updating H
        hatSX = np.dot(W, H)
        num = np.dot(W.T,
                     SX / np.maximum(hatSX**2, eps))
        den = np.dot(W.T,
                     1 / np.maximum(hatSX, eps))
        
        H *= num / np.maximum(den, eps)
    
    return W, H

def NMF_decomp_init(SX, nbComps=10, niter=10, verbose=0,
                    Winit=None, Hinit=None,
                    updateW=True, updateH=True):
    """\
    NMF multiplicative gradient, for Itakura Saito
    divergence measure between ``SX`` and ``np.dot(W,H)``

    .. math::

        \\mathbf{S}_X \\approx \\mathbf{W} \\mathbf{H}
    
        s_{X, fn} \\approx \\sum_{k=1}^K w_{fk} h_{kn}
    
    See for instance [Fevotte2009]_.
    
    :param numpy.ndarray SX: 
        Matrix to be factorized
    :param integer nbComps:
        Number of components / factors into which to decompose `SX`
    :param integer niter: 
        Number of iterations for the NMF algorithm
    :param integer verbose:
        0 for null verbosity, 1 for normal and more for debug
    :param numpy.ndarray Winit:
        Initial array for matrix `W`
    :param numpy.ndarray Hinit:
        Initial array for matrix `H`
    :param boolean updateW: 
        whether to update W or not
    :param boolean updateH: 
        whether to update H or not

    :returns:
      `W` and `H` (:py:class:`numpy.ndarray`) -
      the "spectral" component dictionary matrix and the "activation"
      coefficient matrix.
    
    Notes :
    For (probably marginal) efficiency, the amplitude matrix ``H``
    is "transposed", such that its use in the ``np.dot`` operations uses
    a C-ordered contiguous array. The output is however in the "correct" form.
    """
    freqs, nframes = SX.shape
    if Winit is None or (Winit.shape != (freqs, nbComps)):
        W = np.random.randn(freqs, nbComps)**2
        if verbose and not updateW:
            print "    NMF decomp init: not updating randomly initialized W..."
    else:
        W = np.copy(Winit)
        if verbose and updateW:
            print "    NMF decomp init: updating provided initial W..." 

    if Hinit is not None:
        if Hinit.shape == (nbComps, nframes):
            H = np.copy(Hinit.T)
            if verbose and updateH:
                print "    NMF decomp init: updating provided initial H..."
        elif  Hinit.shape == (nframes, nbComps):
            H = np.copy(Hinit)
            if verbose and updateH:
                print "    NMF decomp init: updating provided initial H..."
        else: raise AttributeError('Hinit not in the right shape.')
    else:
        H = np.random.randn(nframes, nbComps, )**2
        if verbose and not updateH:
            print "    NMF decomp init: not updating randomly initialized H..."
    
    if updateW:
        W /= W.sum(axis=0)
    
    for i in range(niter):
        if verbose:
            print "    NMF iteration %d out of %d" %(i+1, niter)
        if updateW:# updating W
            hatSX = np.dot(W, H.T)
            num = np.dot(SX / np.maximum(hatSX**2, eps),
                         H)
            den = np.dot(1 / np.maximum(hatSX, eps),
                         H)
            
            W *= num / np.maximum(den, eps)
            
            sumW = W.sum(axis=0)
            sumW[sumW==0] = 1.
            W /= sumW
            H *= sumW
        
        if updateH:# updating H
            hatSX = np.dot(H, W.T)
            num = np.dot(SX.T / np.maximum(hatSX**2, eps),
                         W)
            den = np.dot(1 / np.maximum(hatSX, eps),
                         W)
            
            H *= num / np.maximum(den, eps)
    
    return W, H.T

def SFNMF_decomp_init(SX, nbComps=10, nbFiltComps=10,
                      niter=10, verbose=0,
                      Winit=None, Hinit=None,
                      WFiltInit=None, HFiltInit=None,
                      updateW=True, updateH=True,
                      updateWFilt=True, updateHFilt=True,
                      nbResComps=2):
    """\
    Implements a simple source/filter NMF algorithm, similar to that introduced in
    [Durrieu2010]_
       
    """
    freqs, nframes = SX.shape
    if Winit is None or (Winit.shape != (freqs, nbComps)):
        W = np.random.randn(freqs, nbComps)**2
        if verbose and not updateW:
            print "    NMF decomp init: not updating randomly initialized W..."
    else:
        W = np.copy(Winit)
        if verbose and updateW:
            print "    NMF decomp init: updating provided initial W..." 
    if Hinit is None or (Hinit.shape != (nframes, nbComps)):
        H = np.random.randn(nframes, nbComps, )**2
        if verbose and not updateH:
            print "    NMF decomp init: not updating randomly initialized H..."
    else:
        H = np.copy(Hinit)
        if verbose and updateH:
            print "    NMF decomp init: updating provided initial H..."
    
    if updateW:
        W /= W.sum(axis=0)
    
    if WFiltInit is None or (WFiltInit.shape != (freqs, nbFiltComps)):
        WFilt = np.random.randn(freqs, nbFiltComps)**2
        if verbose and not updateWFilt:
            print "    NMF decomp init: not "+\
                  "updating randomly initialized WFilt..."
    else:
        WFilt = np.copy(WFiltInit)
        if verbose and updateWFilt:
            print "    NMF decomp init: updating provided initial WFilt..." 
    if HFiltInit is None or (HFiltInit.shape != (nframes, nbFiltComps)):
        HFilt = np.random.randn(nframes, nbFiltComps, )**2
        if verbose and not updateHFilt:
            print "    NMF decomp init: not updating "+\
                  "randomly initialized HFilt..."
    else:
        HFilt = np.copy(HFiltInit)
        if verbose and updateHFilt:
            print "    NMF decomp init: updating provided initial H..."
    
    if updateWFilt:
        WFilt /= WFilt.sum(axis=0)
    
    Wres = (1 + np.random.randn(freqs, nbResComps))**2
    Hres = (1 + np.random.randn(nframes, nbResComps))**2
    
    if verbose>1:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(211)
        im = ax.imshow(SX)
        fig.colorbar(im)
        ax2 = fig.add_subplot(212, sharex=ax)
        im2 = ax2.imshow(SX)
        fig.colorbar(im2)
    
    for i in range(niter):
        if verbose:
            print "    NMF iteration %d out of %d" %(i+1, niter)
        if updateW:# updating W
            if verbose:
                print "        updating w f0"
            SF0 = np.dot(W, H.T)
            SPHI = np.dot(WFilt, HFilt.T)
            Sres = np.dot(Wres, Hres.T)
            hatSX = SF0 * SPHI + Sres
            num = np.dot(SX * SPHI/ np.maximum(hatSX ** 2, eps),
                         H)
            den = np.dot(SPHI / np.maximum(hatSX, eps),
                         H)
            
            W *= num / np.maximum(den, eps)
            
            sumW = W.sum(axis=0)
            sumW[sumW==0] = 1.
            W /= sumW
            H *= sumW
        
        if updateH:# updating H
            if verbose:
                print "        updating h f0"
            SF0 = np.dot(H, W.T)
            SPHI = np.dot(HFilt, WFilt.T)
            Sres = np.dot(Hres, Wres.T)
            hatSX = SF0 * SPHI + Sres
            num = np.dot(SX.T * SPHI/ np.maximum(hatSX ** 2, eps),
                         W)
            den = np.dot(SPHI / np.maximum(hatSX, eps),
                         W)
            
            H *= num / np.maximum(den, eps)
            if verbose>1:
                im.set_data(np.log(hatSX.T))
                im.set_clim([im.get_array().min(),
                             im.get_array().max()])
                plt.draw()
            
        if updateWFilt:# updating WFilt
            if verbose:
                print "        updating w filter"
            SF0 = np.dot(W, H.T)
            SPHI = np.dot(WFilt, HFilt.T)
            Sres = np.dot(Wres, Hres.T)
            hatSX = SF0 * SPHI + Sres
            num = np.dot(SX * SF0 / np.maximum(hatSX ** 2, eps),
                         HFilt)
            den = np.dot(SF0 / np.maximum(hatSX, eps),
                         HFilt)
            
            WFilt *= num / np.maximum(den, eps)
            
            # normalization of Wfilt
            sumW = WFilt.sum(axis=0)
            sumW[sumW==0] = 1.
            WFilt /= sumW
            HFilt *= sumW
            # normalizing Hfilt and sending energy to H
            ##sumH = HFilt.sum(axis=1)
            ##HFilt /= np.vstack(sumH)
            ##H *= np.vstack(sumH)
        
        if updateHFilt:# updating HFilt
            if verbose:
                print "        updating h filter"
            SF0 = np.dot(H, W.T)
            SPHI = np.dot(HFilt, WFilt.T)
            Sres = np.dot(Hres, Wres.T)
            hatSX = SF0 * SPHI + Sres
            
            if verbose>1:
                im2.set_data(np.log(hatSX.T))
                im2.set_clim([im2.get_array().min(),
                              im2.get_array().max()])
                plt.draw()
                
            num = np.dot(SX.T * SF0 / np.maximum(hatSX ** 2, eps),
                         WFilt)
            den = np.dot(SF0 / np.maximum(hatSX, eps),
                         WFilt)
            
            HFilt *= num / np.maximum(den, eps)
            
            # normalizing Hfilt and sending energy to H
            sumH = HFilt.sum(axis=1)
            H *= np.vstack(sumH)
            sumH[sumH==0] = 1.
            HFilt /= np.vstack(sumH)
            
            ##if verbose>1:
            ##    im2.set_data(np.log(HFilt.T))
            ##    im2.set_clim([im2.get_array().min(),
            ##                  im2.get_array().max()])
            ##    plt.draw()
        # update residual comps:
        if verbose:
            print "        updating w residual"
        SF0 = np.dot(W, H.T)
        SPHI = np.dot(WFilt, HFilt.T)
        Sres = np.dot(Wres, Hres.T)
        hatSX = SF0 * SPHI + Sres
        num = np.dot(SX / np.maximum(hatSX ** 2, eps),
                     Hres)
        den = np.dot(1 / np.maximum(hatSX, eps),
                     Hres)
        
        Wres *= num / np.maximum(den, eps)
        
        # normalization of Wfilt
        sumW = Wres.sum(axis=0)
        sumW[sumW==0] = 1.
        Wres /= sumW
        Hres *= sumW
        
        if verbose:
            print "        updating h residual"
        SF0 = np.dot(H, W.T)
        SPHI = np.dot(HFilt, WFilt.T)
        Sres = np.dot(Hres, Wres.T)
        hatSX = SF0 * SPHI + Sres
            
        num = np.dot(SX.T  / np.maximum(hatSX ** 2, eps),
                     Wres)
        den = np.dot(1 / np.maximum(hatSX, eps),
                     Wres)
        
        Hres *= num / np.maximum(den, eps)
    
    return W, H.T, WFilt, HFilt.T, Wres, Hres
