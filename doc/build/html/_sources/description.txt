=======
pyFASST
=======

 :contributors: Jean-Louis Durrieu
 :web: https://git.epfl.ch/repo/pyfasst/ https://github.com/wslihgt/pyfasst

A Python implementation to the Flexible Audio Source Separation Toolbox

Abstract
========
This toolbox is meant to allow to use the framework FASST and extend it within a python program. It is primarily a re-write in Python of the original Matlab (C) version. The object programming framework allows to extend and create new models an easy way, by subclassing the :py:class:`pyfasst.audioModel.FASST` and re-implementing some methods (in particular methods like :py:meth:`pyfasst.audioModel.MultiChanNMFInst_FASST._initialize_structures`)

Using the Python package
========================

Dependencies
------------

Most of the code is written in `Python <http://www.python.org>`_, but occasionally, there may be some C source code, requiring either Cython or SWIG for compiling. In general, to run this code, the required components are:

  * Matplotlib http://matplotlib.sourceforge.net 
  * Numpy http://numpy.scipy.org
  * Scipy http://www.scipy.org
  * setuptool https://pypi.python.org/pypi/setuptools

Install
-------

Unpack the tarball, change directory to it, and run the installation with `setup.py`. Namely:
 1. ``tar xvzf pyFASST-X.Y.Z.tar.gz``
 2. ``cd pyFASST-X.Y.Z``
 3. ``python setup.py build``
 4. [if you want to install it] ``[sudo] python setup.py install [--user]``

In addition to the aforementioned packages, installing this package requires to compile the tracking part, :py:mod:`pyfasst.SeparateLeadStereo.tracking._tracking`. In the corresponding folder, type::

  python setup.py build_ext --inplace

Examples
--------

Using the provided audio model classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have implemented several classes that can be used directly, without the need to re-implement or sub-class :py:class:`pyfasst.audioModel.FASST`. In particular, we have:

 * :py:class:`pyfasst.audioModel.MultiChanNMFInst_FASST`, :py:class:`pyfasst.audioModel.MultiChanNMFConv`, :py:class:`pyfasst.audioModel.MultiChanHMM`: these classes originate from the distributed Matlab version of FASST_. For example, the separation of the voice and the guitar on the `tamy <>` example gives, with a simple model with 2 sources, with instantaneous mixing parameters and NMF model on the spectral parameters (to run from where one can find the `tamy.wav` file) - don't expect very good results!::

    >>> import pyfasst.audioModel as am
    >>> filename = 'data/tamy.wav'
    >>> # initialize the model
    >>> model = am.MultiChanNMFInst_FASST(
            audio=filename,
            nbComps=2, nbNMFComps=32, spatial_rank=1,
            verbose=1, iter_num=50)
    >>> # estimate the parameters
    >>> model.estim_param_a_post_model()
    >>> # separate the sources using these parameters
    >>> model.separate_spat_comps(dir_results='data/')

   Somewhat improving the results could be to use the convolutive mixing parameters::
  
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
    >>> model.estim_param_a_post_model()
    >>> # separate the sources using these parameters
    >>> model.separate_spat_comps(dir_results='data/')

   The following example shows the results for a more synthetic example (synthetis anechoic mixture of the voice and the guitar, with a delay of 0 for the voice and 10 samples from the left to the right channel for the guitar)::

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
    >>> model.estim_param_a_post_model()
    >>> # separate the sources using these parameters
    >>> model.separate_spat_comps(dir_results='data/')

 * :py:class:`pyfasst.audioModel.multiChanSourceF0Filter`: this class assumes that all the sources share the same spectral shape dictionary and spectral structure, _i.e._ a source/filter model (2 _factors_, in FASST terminology), with a filter spectral shape dictionary generated as a collection of *smooth* windows (overlapping Hann windows), and the source dictionary is computed as a collection of spectral *combs* following a simple vocal glottal model (see [Durrieu2010]_). The advantage of this class is that in terms of memory, all the sources share the same dictionaries. However, that means it makes no sense to modify these dictionaries (at least not individually - which is the case in this algorithm) and they are therefore fixed by default. This class also provides methods that help to initialize the various parameters, assuming the specific structure presented above.

 * :py:class:`pyfasst.audioModel.multichanLead`

 * Additionally, we provide a (not-very-exhaustive) plotting module which helps in displaying some interesting features from the model, such as::

    >>> import pyfasst.tools.plotTools as pt
    >>> # display the estimated spectral components
    >>> # (one per row of subplot)
    >>> pt.subplotsAudioModelSpecComps(model)
    >>> # display a graph showing where the sources have been "spatially"
    >>> # estimated: in an anechoic case, ideally, the graph for the 
    >>> # corresponding source is null everywhere, except at the delay 
    >>> # between the two channels:
    >>> delays, delayDetectionFunc = pt.plotTimeCorrelationMixingParams(model)

TODO: add typical SDR/SIR results for these examples.

Creating a new audio model class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* In the base class :py:class:`pyfasst.audioModel.FASST`, there are already some basic implementations that should fit any ensuing model. In particular, the :py:meth:`pyfasst.audioModel.estim_param_a_post_model` method estimates the parameters of the model, using the GEM algorithm [Ozerov2012]_. It is therefore very likely that the only things that one should take care of is to initialize the model and construct the model such that it corresponds to the desired structure.

* The base class does not implement a :py:meth:`_initialize_structures` method. The different subclasses that concretely correspond to different models do however define such a method, where the following parameters need to be initiated:

 - :py:attr:`FASST.spat_comps`:
    
    +-----------------------------+--------------------------+---------------------------------------------+
    | variable                    | description              |   possible values                           |
    +=============================+==========================+=============================================+
    |`spat_comps[n]`              | `n`-th spatial component | dictionary with the fields detailled below  |
    +-----------------------------+--------------------------+---------------------------------------------+
    |`spat_comps[n]['time_dep']`  | define the time          | 'indep'                                     |
    |                             | dependencies             |                                             |
    +-----------------------------+--------------------------+---------------------------------------------+
    |`spat_comps[n]['mix_type']`  | which type of mixing     | * 'inst' - instantaneous                    |
    |                             | should be considered     | * 'conv' - convolutive                      |
    +-----------------------------+--------------------------+---------------------------------------------+
    |`spat_comps[n]['frdm_prior']`|                          | * 'free' to update the mixing parameters    |
    |                             |                          | * 'fixed' to keep the parameters unchanged  |
    +-----------------------------+--------------------------+---------------------------------------------+
    |`spat_comps[n]['params']`    | the actual mixing        | * mix_type == 'inst' :                      |
    |                             | parameters.              |      n_channels x rank `numpy.ndarray`      |
    |                             |                          | * mix_type == 'conv' :                      |
    |                             |                          |      rank x n_chan x n_freq `numpy.ndarray` |
    +-----------------------------+--------------------------+---------------------------------------------+

   Note: the way the parameters are stored is a bit convoluted and making a more consistent ordering of the parameters (between instantaneous and convolutive) would be an improvement.

 - :py:attr:`FASST.spec_comps`:

   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | variable                                      | description                                | values                                                             |
   +===============================================+============================================+====================================================================+
   | `spec_comps[n]`                               | `n`-th spectral component                  | dictionary with the following fields                               |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['spat_comp_ind']`              | the associated spatial component           | (integer)                                                          |
   |                                               | in `spat_comps`.                           |                                                                    |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]`                  | `f`-th factor of `n`-th spectral component | dictionary with the following parameters                           |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['FB']`            | Frequency Basis                            | (`nbFreqsSigRepr` x `n_FB_elts`) `ndarray`:                        |
   |                                               |                                            | `n_FB_elts` is the number of elements in the basis (or dictionary) |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['FW']`            | Frequency Weights                          | (`n_FB_elts` x `n_FW_elts`) `ndarray`:                             |
   |                                               |                                            | `n_FW_elts` is the number of desired combinations of FB elements   |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['TW']`            | Time Weights                               | (`n_FW_elts` x `n_TB_elts`) `ndarray`:                             |
   |                                               |                                            | `n_TB_elts` is the number of elements in the time basis            |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['TB']`            | Temporal Basis                             | empty list `[]` or (`n_TB_elts` x `nbFramesSigRepr`) `ndarray`:    |
   |                                               |                                            | if `[]`, then `n_TB_elts` in TW should be `nbFramesSigRepr`.       |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['TW_constr']`     |                                            |  * 'NMF': Non-negative Matrix Factorization                        |
   |                                               |                                            |  * 'GMM', 'GSMM': Gaussian (Scale) Mixture Model                   |
   |                                               |                                            |  * 'HMM', 'SHMM': (Scaled) Hidden Markov Model                     |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['TW_all']`        | for discrete state models                  |  same as `spec_comps[n]['factor'][f]['TW']`                        |
   |                                               | (TW_constr != 'NMF'), keeps track of the   |                                                                    |
   |                                               | scales for all the possible states         |                                                                    |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['TW_DP_params']`  | *Dynamic Programming* (?) parameters       | * TW_constr in ('GMM', 'GSMM'):                                    |
   |                                               | prior or transition probabilities.         |   (`number_states`) `ndarray`.                                     |
   |                                               |                                            |   Prior probabilites for each state.                               | 
   |                                               |                                            |   `number_states` is the number of states                          | 
   |                                               |                                            |   (typically `spec_comp[n]['factor'][f]['TW'].shape[0]`).          |
   |                                               |                                            | * TW_constr in ('HMM', 'SHMM'):                                    |
   |                                               |                                            |   (`number_states` x `number_states`) `ndarray`.                   |
   |                                               |                                            |   Transition probabilites for each state.                          |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+
   | `spec_comps[n]['factor'][f]['XX_frdm_prior']` | whether to update a parameter set          |  * 'free' update the parameters                                    |
   |                                               | or not, where `XX` is one of               |  * 'fixed' do not update                                           |
   |                                               | `FB`, `FW`, `TW`, `TB`, `TW_DP`            |                                                                    |
   +-----------------------------------------------+--------------------------------------------+--------------------------------------------------------------------+

   The key names are reproduced from the Matlab toolbox. 
    
* When instantiating a subclass, in the :py:meth:`pyfasst.audioModel.FASST.__init__` method, or at least before running :py:meth:`pyfasst.audioModel.FASST.estim_param_a_post_model`, two things should be done: computing the signal representation by calling :py:meth:`pyfasst.audioModel.FASST.comp_transf_Cx` and initializing the model parameters (see above). By default, the base class does not compute the signal representation for memory saving. Some other processing can therefore be run (initializing with :py:class:`pyfasst.demixTF.DEMIX` or with a run of :py:class:`pyfasst.SeparateLeadStereo.SeparateLeadProcess`) without the burden of the (unused) memory in the current instance. Just call it when needed. 

Algorithms
==========

The FASST framework and the audio signal model are described in [Ozerov2012]_. We have implemented this Python version mostly thanks to the provided Matlab (C) code available at http://bass-db.gforge.inria.fr/fasst/. 

For initialization purposes, several side algorithms and systems have also been implemented:
* SIMM model (Smooth Instantaneous Mixture Model) from [Durrieu2010]_ and [Durrieu2011]_: allows to analyze, detect and separate the lead instrument from a polyphonic audio (musical) mixture. Note: the original purpose of this implementation was to provide a sensible way of using information from the SIMM model into the more general multi-channel audio source separation model provided, for instance, by FASST.  It is implemented in the :py:mod:`pyfasst.SeparateLeadStereo.SeparateLeadStereoTF` module.

* DEMIX algorithm (Direction Estimation of Mixing matrIX) [Arberet2010]_ for spatial mixing parameter initialization. It is implemented as the :py:mod:`pyfasst.demixTF` module.

Notes and remarks
=================

* Reworking on the source code, it seems that the use of `spat_comps` and `spec_comps` to allow the various ranks is slightly complicated. A major refactoring of this algorithm could be to define, for instance, a class that represents one source or component, including its spatial and spectral parameters. This would allow to avoid to have to retrieve the association between spatial and spectral parameters (through the `spec_comps[n]['spat_comp_ind']` variable) during their re-estimation.

* As of 20130823: documentation still *Work In Progress*. Hopefully the most important information is provided in this document. Specific implementation issues may come in time.

* TODO: one should check that the computations are similar to those provided by the Matlab Toolbox. So far, in many cases, this implementation has provided the author with satisfying results, but a more formal evaluation to compare the performance of both implementations would be welcome. 

.. [Arberet2010] Arberet, S.; Gribonval, R. and Bimbot, F., 
   `A Robust Method to Count and Locate Audio Sources in a Multichannel 
   Underdetermined Mixture`, IEEE Transactions on Signal Processing, 2010, 
   58, 121 - 133. [`web <http://infoscience.epfl.ch/record/150461/>`_]

.. [Durrieu2010] J.-L. Durrieu, G. Richard, B. David and C. F\\'{e}votte, 
   `Source/Filter Model for Main Melody Extraction From Polyphonic Audio 
   Signals`, IEEE Transactions on Audio, Speech and Language Processing, 
   special issue on Signal Models and Representations of Musical and 
   Environmental Sounds, March 2010, Vol. 18 (3), pp. 564 -- 575.

.. [Durrieu2011] J.-L. Durrieu, G. Richard and B. David, 
   `A Musically Motivated Representation For Pitch Estimation And Musical 
   Source Separation <http://www.durrieu.ch/research/jstsp2010.html>`_, 
   IEEE Journal of Selected Topics on Signal Processing, October 2011, 
   Vol. 5 (6), pp. 1180 - 1191.

.. [Ozerov2012] A. Ozerov, E. Vincent, and F. Bimbot, 
   `A general flexible framework for the handling of prior information in audio
   source separation <http://hal.inria.fr/hal-00626962/>`_, 
   IEEE Transactions on Audio, Speech and Signal Processing, Vol.  20 (4), 
   pp. 1118-1133 (2012).

.. _FASST: http://bass-db.gforge.inria.fr/fasst/
