"""pyFASST package setup file

to build it::

  python setup.py build

to install it::

  python setup.py install [--user | --record files.txt]
  
to remove:

  check the files in `files.txt` (from above command)
  and remove them manually. (`pip` or `easy_install` may allow you
  to do this in a cleaner way)

TODO: include data files for tests
"""

import setuptools
from distutils.core import setup
from distutils.extension import Extension
import numpy

# import pyfasst.audioModel as am
long_description="""\
FASST (Flexible Audio Source Separation Toolbox) class
    subclass it to obtain your own flavoured source separation model!

You can find more about the technique and how to use this module in the
provided documentation in `doc/` (`using the python package
<../description.html#using-the-python-package>`_)
    
Adapted from the Matlab toolbox available at:
http://bass-db.gforge.inria.fr/fasst/

Jean-Louis Durrieu, EPFL-STI-IEL-LTS5
::

    jean DASH louis AT durrieu DOT ch

2012-2013
http://www.durrieu.ch

"""

# Needed to fix pip
# See https://pypi.python.org/pypi/setuptools_cython/,
# http://mail.python.org/pipermail/distutils-sig/2007-September/thread.html#8204
# DJL: from http://comments.gmane.org/gmane.comp.python.cython.user/9503
# Added this, on Linux, this seems necessary otherwise Cython is not
# called and the C/C++ source file is not generated.
import sys
if 'setuptools.extension' in sys.modules: 
    m = sys.modules['setuptools.extension']        
    m.Extension.__dict__ = m._Extension.__dict__

# http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension(
            name="pyfasst.SeparateLeadStereo.tracking._tracking",
            sources=["pyfasst/SeparateLeadStereo/tracking/_tracking.pyx"],
            include_dirs=[numpy.get_include(), ],
            language="c++"
            # this is for mac, for the compiler to use
            # g++ and not gcc
            ),
        ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("pyfasst.SeparateLeadStereo.tracking._tracking",
                  ["pyfasst/SeparateLeadStereo/tracking/_tracking.cpp"],
                  include_dirs=[numpy.get_include(), ],
                  language="c++"
                  # this is for mac, for the compiler to use
                  # g++ and not gcc
                  ),
        ]

setup(
    name='pyFASST',
    version='0.9.2',
    author='Jean-Louis Durrieu',
    author_email='jean-louis@durrieu.ch',
    packages=setuptools.find_packages(),
    scripts=[],
    url='http://github.com/wslihgt/pyfasst/',
    license='GNU GENERAL PUBLIC LICENSE',
    description=(
        'Python implementation of the Flexible Audio Source Separation Toolbox'+
        ' (FASST)'),
    install_requires=[
        'numpy',
        'scipy'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    long_description=long_description,
    )

