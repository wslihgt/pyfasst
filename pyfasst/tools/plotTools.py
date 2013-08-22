"""Plotting tools to be used with PyFASST and audioModel classes

**Examples:**

::

    >>> import pyfasst.tools.plotTools as pt
    >>> # display the estimated spectral components
    >>> # (one per row of subplot)
    >>> pt.subplotsAudioModelSpecComps(model)
    >>> # display a graph showing where the sources have been "spatially"
    >>> # estimated: in an anechoic case, ideally, the graph for the 
    >>> # corresponding source is null everywhere, except at the delay 
    >>> # between the two channels:
    >>> pt.plotTimeCorrelationMixingParams(model)


"""

import matplotlib.pyplot as plt
import numpy as np

def subplotsAudioModelSpecComps(model, fig=None, diffdispdb = 60 ):
    """Computes the spectral powers for each of the spatial components
    of model, and displays them on a single figure.

    model should be instance of a sub-class of FASST
    
    
    """
    nbComps = model.nbComps
    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()
    ax = fig.add_subplot(nbComps,1,1)
    for n in range(nbComps-1):
        axp = fig.add_subplot(nbComps,1,n+1, sharex=ax, sharey=ax)
        ##axp.imshow(
        ##    np.log(
        ##    model.spec_comps[n]['factor'][
        ##        min(factor,maxfact-1)]['TW']))
        logspec = 0.
        ##np.log(
        #            np.dot(
        #            np.dot(
        #            model.spec_comps[n]['factor'][0]['FB'],
        #            model.spec_comps[n]['factor'][0]['FW']),
        #            model.spec_comps[n]['factor'][0]['TW']))
        for nfact, fact in model.spec_comps[n]['factor'].items():
            logspec += 10*np.log10(
                np.dot(
                np.dot(
                fact['FB'],
                fact['FW']),
                fact['TW']))
        axp.imshow(logspec)
        logspecMax = logspec.max()
        axp.get_images()[0].set_clim([logspecMax-diffdispdb, logspecMax])
        #plt.colorbar()
    n += 1
    axp = fig.add_subplot(nbComps,1,n+1, sharex=ax, sharey=ax)
    axp.imshow(
        10*np.log10(
        np.dot(
        np.dot(
        model.spec_comps[n]['factor'][0]['FB'],
        model.spec_comps[n]['factor'][0]['FW']),
        model.spec_comps[n]['factor'][0]['TW']))
        )
    logspecMax = logspec.max()
    axp.get_images()[0].set_clim([logspecMax-diffdispdb, logspecMax])
    plt.draw()
    return fig

def subplotsAudioModelSpecTW(model):
    """Displays the time weights for each of the spectral component of
    `model`. 
    
    `model` should be instance of (at least) FASST
    """
    nbComps = model.nbComps
    fig = plt.figure()
    ax = fig.add_subplot(nbComps,1,1)
    for n in range(nbComps-1):
        axp = fig.add_subplot(nbComps,1,n+1, sharex=ax,)# sharey=ax)
        ##axp.imshow(
        ##    np.log(
        ##    model.spec_comps[n]['factor'][
        ##        min(factor,maxfact-1)]['TW']))
        axp.imshow(
            np.log(
            model.spec_comps[n]['factor'][0]['TW'])) 
    n += 1
    axp = fig.add_subplot(nbComps, 1, n + 1, sharex=ax)
    axp.imshow(
        np.log(
        model.spec_comps[n]['factor'][0]['TW']))



def subplotsAudioModelSpecCompNimp(model, nimp='FB'):
    """Display any of the field for the first factor of each
    spectral component of `model`, `nimp` should therefore either be
    'FB', 'FW', 'TW' or 'TB'. 
    
    model should be instance of (at least) FASST
    """
    nbComps = model.nbComps
    fig = plt.figure()
    ax = fig.add_subplot(nbComps,1,1)
    for n in range(nbComps-1):
        axp = fig.add_subplot(nbComps,1,n+1, )# sharex=ax, sharey=ax)
        ##axp.imshow(
        ##    np.log(
        ##    model.spec_comps[n]['factor'][
        ##        min(factor,maxfact-1)]['TW']))
        axp.imshow(
            np.log(
            model.spec_comps[n]['factor'][0][nimp])) 
    n += 1
    axp = fig.add_subplot(nbComps, 1, n + 1, )
    axp.imshow(
        np.log(
        model.spec_comps[n]['factor'][0][nimp]))

def plotTimeCorrelationMixingParams(model, **kwargs):
    """Computes the inverse Fourier transform of the ratio of each of the
    'steering vectors', for each spatial component. This correlation provides
    some insight in the spatial mixing process, since for an anechoic pair of
    steering vectors, the ratio is almost a complex exponential, with spatial
    frequency equal to the delay of arrival between the 2 channels.


    **Inputs:**
    
    :param model:
        a source model, instance of :py:class:`pyfasst.audioModel.FASST`
    :param kwargs:
        the other keyword arguments are passed to a :py:func:`plt.plot`
        Please refer to that function to see what keywords are accepted.

    **Outputs:**
    
    :returns:
       1. `delays` -
       an array containing the delays in samples, the axis corresponding to
       the other returned arrays
       
       2. `delayDetectionFunction` -
       an array of which each column corresponds to a spatial component in
       :py:attr:`model`
       
    """
    delayDetectionFunction = np.array(
        [np.fft.fftshift(
            np.abs(np.fft.ifft(model.spat_comps[spat_ind]['params'][0][1]
                               /model.spat_comps[spat_ind]['params'][0][0],
                               n=model.sig_repr_params['fsize'])))
         for spat_ind in model.spat_comps.keys()]).T
    delays = (
        np.arange(model.sig_repr_params['fsize'])
        - model.sig_repr_params['fsize']/2)
    plt.figure()
    plt.plot(
        delays,
        delayDetectionFunction,
        **kwargs)
    plt.legend(model.spat_comps.keys())
    plt.xlabel('Delay in samples')
    
    return delays, delayDetectionFunction
