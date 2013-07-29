"""Getting imports

adapted from sklearn module testing.py file (for the imports) 

run at the root of the package::

  nosetests tests -v --with-coverage --cover-package=pyfasst

which enables verbose output, and shows how much of the package in
``pyfasst`` is being covered by the tests

2013 Jean-Louis Durrieu
"""
import numpy as np

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_true
from nose.tools import assert_false
from nose.tools import assert_raises
from nose.tools import raises
from nose import SkipTest
from nose import with_setup

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_less

