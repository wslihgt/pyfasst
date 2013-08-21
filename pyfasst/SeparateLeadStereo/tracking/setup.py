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

### from scikits.talkbox
##def configuration(parent_package='', top_path=None):
##    from numpy.distutils.misc_util import Configuration
##    config = Configuration('tracking', parent_package, top_path)
##    config.add_extension(name='_tracking',
##                         sources=['_tracking.pyx'],
##                         include_dirs=[numpy.get_include(), ],
##                         language="c++")

##    return config

##if __name__=='__main__':
##    from numpy.distutils.core import setup
##    setup(**configuration(top_path='').todict())
