"""tests for pyfasst.tools.utils

2013 Jean-Louis Durrieu
"""

from ...testing import * # is this really legal?

import numpy as np
import pyfasst.tools.utils as utils

def test_db():
    """db returns the decibel value of a given number
    """
    assert_equal(utils.db(1), 0)
    assert_equal(utils.db(10), 10)

def test_ident():
    """ident is identity function
    """
    assert_equal(utils.ident(1), 1)
    assert_equal(utils.ident(10), 10)

def test_nextpow2():
    """nextpow2 should output the next power of 2 of a given number
    """
    assert_equal(utils.nextpow2(2), 2)
    assert_equal(utils.nextpow2(2**10+1), 2**(11))
    assert_equal(utils.nextpow2(2**20+1), 2**(21))

def test_sinebell():
    """sinebell is a window function
    """
    assert_array_almost_equal(utils.sinebell(5),
                       np.array([0., 0.58778525,  0.95105652,
                                 0.95105652,  0.58778525]))
    assert_array_almost_equal(
        utils.sinebell(10),
        np.array([ 0.        ,  0.30901699,  0.58778525,  0.80901699,
                   0.95105652,  1.        ,  0.95105652,  0.80901699,
                   0.58778525,  0.30901699]))
    
def test_hann():
    """Hann window function, erronously called hanning everywhere!
    """
    assert_array_almost_equal(utils.hann(11), np.hanning(11))
    assert_array_almost_equal(utils.hann(22), np.hanning(22))

def test_sqrt_blackmanharris():
    """Blackman-Harris window function
    """
    assert_array_almost_equal(
        utils.sqrt_blackmanharris(10),
        np.array([ 0.00774597,  0.12276471,  0.38345737,  0.72150884,
                   0.96522498,  0.96522498,  0.72150884,  0.38345737,
                   0.12276471,  0.00774597]))
    assert_array_almost_equal(
        utils.sqrt_blackmanharris(22),
        np.array([
            0.00774597,  0.04002509,  0.09757199,  0.18273402,  0.29549176,
            0.43029881,  0.57691724,  0.72150884,  0.84850893,  0.94306403,
            0.99353553,  0.99353553,  0.94306403,  0.84850893,  0.72150884,
            0.57691724,  0.43029881,  0.29549176,  0.18273402,  0.09757199,
            0.04002509,  0.00774597]))
    
