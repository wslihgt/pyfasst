"""tests for pyfasst.audioModel

2013 Jean-Louis Durrieu
"""

from ..testing import * # is this really legal?

import numpy as np
import pyfasst.audioModel as am

from unittest import TestCase

class FASSTTestCase(TestCase):

    def __init__(self, ):
        super(FASSTTestCase, self).__init__()
        self.fasstkwargs = {
            'audio': 'data/tamy.wav',
            'iter_num': 2,
            'verbose' : False,
            'nbComps' : 3,
            }
        
# class FASSTTestCase
