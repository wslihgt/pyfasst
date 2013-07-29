"""Time-Frequency Transforms

TODO: turn this into something more self-contained (like defining a super class
for all the possible time-freq transforms)
"""
from minqt import MinQTransfo, CQTransfo, sqrt_blackmanharris

from stft import STFT # TODO: should be the opposite, should import stft from here into audioObject
from nsgt import NSGMinQT

# Possible super class transform:
class TFTransform(object):
    transformname = 'dummy'
    
    transfo = None # risky to have it defined here
    
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
        pass
    
    def computeTransform(self):
        pass
    
    def invertTransform(self):
        pass
    

tftransforms = {
    'stftold': TFTransform, # just making dummy
    'stft': STFT,
    'mqt': MinQTransfo,
    'minqt': MinQTransfo,
    'nsgmqt': NSGMinQT,
    'cqt': CQTransfo}
    
