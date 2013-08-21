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

In addition to the aforementioned packages, installing this package requires to compile the tracking part, in :py:mod:`pyfasst.SeparateLeadStereo.tracking`. In the corresponding folder, type::

  python setup.py build_ext --inplace



Examples
--------

Using the provided audio model classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have implemented several classes that can be used directly, without the need to re-implement or sub-class :py:class:`pyfasst.audioModel.FASST`. In particular, we have:

 * :py:class:`pyfasst.audioModel.MultiChanNMFInst_FASST`, :py:class:`pyfasst.audioModel.MultiChanNMFConv`, :py:class:`pyfasst.audioModel.MultiChanHMM`: these classes originate from the distributed Matlab version of FASST_

 * :py:class:`pyfasst.audioModel.multiChanSourceF0Filter`

 * :py:class:`pyfasst.audioModel.multichanLead`

Creating a new audio model class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Algorithms
==========

The FASST framework is described in [Ozerov2012]_. We have implemented this Python version mostly thanks to the provided Matlab (C) code available at http://bass-db.gforge.inria.fr/fasst/. 

For initialization purposes, several side algorithms and systems have also been implemented:
  * SIMM model (Smooth Instantaneous Mixture Model) from [Durrieu2010]_ and [Durrieu2011]_: allows to analyze, detect and separate the lead instrument from a polyphonic audio (musical) mixture. Note: the original purpose of this implementation was to provide a sensible way of using information from the SIMM model into the more general multi-channel audio source separation model provided, for instance, by FASST. 
  
  * DEMIX algorithm (Direction Estimation of Mixing matrIX) [Arberet2010]_ for spatial mixing parameter initialization.

References
==========
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
