"""
``utils.py``

Useful functions for (audio) signal processing 

2013 Jean-Louis Durrieu

http://www.durrieu.ch

Content
-------
"""

import numpy as np
import scipy.signal as spsig  # for the windows

def db(val):
    """
    :py:func:`db` db(positiveValue)
    
    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(val)

def ident(energy):
    ''':py:func:`ident` : identity function, return the inputs unchanged
    '''
    return energy

def nextpow2(i):
    """
    Find :math:`2^n` that is equal to or greater than.
    
    code taken from the website:
    
     http://www.phys.uu.nl/~haque/computing/WPark_recipes_in_python.html
    """
    n = 2
    while n < i:
        n = n * 2
    return n

def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)
    
    Computes a \"sinebell\" window function of length L=lengthWindow
    
    The formula is:

    .. math::
    
        window(t) = sin(\pi \\frac{t}{L}), t=0..L-1
        
    """
    window = np.sin((np.pi*(np.arange(lengthWindow)))/(1.0*lengthWindow))
    return window

def hann(args):
    """
    window = hann(args)
    
    Computes a Hann window, with NumPy's function hanning(args).
    """
    return np.hanning(args)

def sqrt_blackmanharris(M):
    """A root-squared Blackman-Harris window function.
    
    For use in scholkhuber and klapuri's framework.
    """
    return np.sqrt(spsig.blackmanharris(M))
