from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("_tracking",
                         ["_tracking.pyx"],
                         include_dirs=[numpy.get_include(), ],
                         language="c++"
                         # this is for mac, for the compiler to use
                         # g++ and not gcc
                         ),]

setup(
  name = 'ViterbiTrackingAlgorithm',
  description = 'Viterbi Tracking Algorithm',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

# Extension
