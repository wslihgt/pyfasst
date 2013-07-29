"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2012
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)


covered by Creative Commons Attribution-NonCommercial-ShareAlike license (CC BY-NC-SA)
http://creativecommons.org/licenses/by-nc-sa/3.0/at/deed.en


--
Original matlab code copyright follows:

AUTHOR(s) : Monika Drfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

"""

from cq import NSGT,CQ_NSGT
from slicq import NSGT_sliced,CQ_NSGT_sliced
from fscale import Scale,OctScale,LogScale,LinScale,MelScale,MinQScale
from warnings import warn

try:
    from audio import SndReader,SndWriter
except ImportError:
    warn("Audio IO routines (scikits.audio module) could not be imported")

import unittest

class NonStatGaborT(object ):
    transformname = 'nsgt'
    def __init__(self, scale, fs, datalength, real=True,
                 matrixform=1,
                 reducedform=0, **kwargs):
        """Just a wrapper for the wonderful nsgt framework
        
        to be in adequation with the MinQT program
        (from Schorkhuber's framework)
        
        this wrapper is loaded from tft.py
        """
        # getting the keyword arguments for nsgt
        nsgtkwargs = {}
        for k,v in kwargs.items():
            if k in NSGT.__init__.func_code.co_varnames and \
                   k not in ('scale', 'fs', 'datalength', 'real', 'matrixform',
                             'reducedform', 'verbose'):
                nsgtkwargs[k] = v
        self.nsgt = NSGT(scale=scale, fs=fs, Ls=datalength,
                         real=real,
                         matrixform=matrixform,
                         reducedform=reducedform,
                         **nsgtkwargs)
        self.kwargs = kwargs
        self.kwargs['matrixform'] = matrixform
        self.kwargs['reducedform'] = reducedform
        self.kwargs['real'] = real
        self.fs = fs
        "TODO: check the following"
        self.freqbins = len(self.nsgt.scale.F())+2
        
    def computeTransform(self, data):
        """compute the (forward) transform
        """
        # assumes monohponic data
        self.checkDataLength(data.size)
        # TODO: is that the right dim...
        self._transfo = self.nsgt.forward(data)
        
    def invertTransform(self):
        """compute the inverse transform
        """
        # assumes monohponic data
        self.checkDataLength(data.size)
        return self.nsgt.backward(self._transfo)
        
    def checkDataLength(self, datalength):
        """
        """
        if datalength!=self.nsgt.Ls:
            self.__init__(scale=self.nsgt.scale,
                          fs=self.nsgt.fs,
                          datalength=datalength,
                          **self.kwargs)
    
    def _set_transfo(self, transfo):
        self._transfo = transfo
    def _get_transfo(self):
        return self._transfo
    def _del_transfo(self):
        del self._transfo
    transfo = property(fset=_set_transfo,
                       fget=_get_transfo,
                       fdel=_del_transfo)

class NSGMinQT(NonStatGaborT):
    transformname = 'nsgminqt'
    def __init__(self, fmin, fmax, bins, linFTLen, fs, datalength=None,
                 **kwargs):
        """convenient NSGMinQT wrapper
        """
        self.scale = MinQScale(fmin=fmin,
                               fmax=fmax,
                               bpo=bins,
                               ftbnds=linFTLen,
                               fs=fs)
        if datalength is None:
            datalength = fs # 1s of signal, for a start
        super(nsgtMinQT, self).__init__(scale=self.scale,
                                        fs=fs,
                                        datalength=datalength,
                                        **kwargs)

class Test_CQ_NSGT(unittest.TestCase):

    def test_transform(self,length=100000,fmin=50,fmax=22050,bins=12,fs=44100):
        import numpy as N
        s = N.random.random(length)
        nsgt = CQ_NSGT(fmin,fmax,bins,fs,length)
        
        # forward transform 
        c = nsgt.forward(s)
        # inverse transform 
        s_r = nsgt.backward(c)
        
        norm = lambda x: N.sqrt(N.sum(N.square(N.abs(x))))
        rec_err = norm(s-s_r)/norm(s)

        self.assertAlmostEqual(rec_err,0)

if __name__ == "__main__":
    unittest.main()
