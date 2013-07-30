"""tests for pyfasst.tools.distances

2013 Jean-Louis Durrieu
"""

from ...testing import * # is this really legal?

import numpy as np
import pyfasst.tools.distances as dist

def test_ISDistortion():
    """ISDistortion is the Itakura Saito divergence
    """
    assert_equal(dist.ISDistortion(1,1.), 0)
    assert_true(dist.ISDistortion(1,2.))

