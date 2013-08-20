"""Time-Frequency Transforms

TODO: turn this into something more self-contained (like defining a super class
for all the possible time-freq transforms)
"""
from minqt import MinQTransfo, CQTransfo, sqrt_blackmanharris

from stft import STFT # TODO: should be the opposite, should import stft from here into audioObject
from nsgt import NSGMinQT

# Possible super class transform: 
class TFTransform(object):
    """TFTransform is the Time-Frequency Transform base class. All the
    TF representations sub-classing it should implement the following
    methods:
    
    * :py:func:`TFTransform.computeTransform` to compute the desired transform
      on data. The transform is then stored in `TFTransform.transfo`_
      
    * :py:func:`TFTransform.invertTransform` to invert the transform from the
      stored transform in `TFTransform.transfo`_
      
    """
    transformname = 'dummy'
    
    transfo = None # risky to have it defined here
    """.. _TFTransform.transfo:
    
    `TFTransform.transfo` receives the transform when computeTransform
    is called.
    
    """
    
    def __init__(self, 
                 fmin=25, fmax=1000, bins=12,
                 fs=44100,
                 q=1,
                 atomHopFactor=0.25,
                 thresh=0.0005, 
                 winFunc=None,
                 perfRast=0,
                 cqtkernel=None,
                 lowPassCoeffs=None,
                 data=None,
                 verbose=0,
                 **kwargs):
        """We define all the possible input for the transform here.
        Maybe not the brightest of all ideas. One has to be able to
        call all sub-classes even if they don't have the same input
        parameters.
        
        For convenience, the parameterization should be done at this
        call, and not when computing the transform (with only ``data``
        as input). This is not compatible with the NSQT framework, which
        requires to know the size of the signal for initialization. 
        """
        pass
    
    def computeTransform(self, data):
        """Computes the transform on the provided data.
        The sub-classes should re-implement this method,
        and store the result in the attribute
        `TFTransform.transfo`_.
        """
        pass
    
    def invertTransform(self):
        """Computes the inverse transform from the stored
        transform in `TFTransform.transfo`_
        """
        pass
    

tftransforms = {
    'stftold': TFTransform, # just making dummy
    'stft': STFT,
    'mqt': MinQTransfo,
    'minqt': MinQTransfo,
    'nsgmqt': NSGMinQT,
    'cqt': CQTransfo}
"""A convenience dictionary, with abbreviated names for the transforms."""
    
