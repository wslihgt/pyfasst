"""testing SeparateLeadStereoTF


"""
from ...testing import *

import pyfasst.SeparateLeadStereo.SeparateLeadStereoTF as SLS
from unittest import TestCase

class SLSTestCase(TestCase):
    
    def setUp(self):
        self.SLSkwargs = {
            'inputAudioFilename': 'data/tamy.wav',
            'nbIter': 2,
            'verbose' : True,
            }
        
class STFTSLStest(SLSTestCase):
    """Testing SeparateLeadStereo with STFT
    """
    def setUp(self):
        super(STFTSLStest, self).setUp()
        self.SLSkwargs['tfrepresentation'] = 'stft'
    
    def testInstantiate(self):
        """Testing instantiation of SeparateLeadStereo with STFT
        """
        self.model = SLS.SeparateLeadProcess(
            **self.SLSkwargs
            )
        pass
    
    def testRun(self):
        """Launching the estimation
        """
        if not(hasattr(self, 'model')):
            self.testInstantiate()
        self.model.autoMelSepAndWrite()
        pass

class MinQTSLStest(STFTSLStest):
    """Testing SeparateLeadStereo with MinQT: takes a lot of time the first time, because generating the dictionary of spectra for the source part
    """
    def setUp(self):
        super(MinQTSLStest, self).setUp()
        self.SLSkwargs['tfrepresentation'] = 'mqt'


##class CQTSLStest(STFTSLStest):
##    """Testing SeparateLeadStereo with CQT 
##    """
##    def setUp(self):
##        super(MinQTSLStest, self).setUp()
##        self.SLSkwargs['tfrepresentation'] = 'cqt'
    

