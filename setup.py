"""pyFASST package setup file
"""


import setuptools
from distutils.core import setup
from distutils.extension import Extension
import numpy

# http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []
data_files = [
    'data/tamy.wav',]

if use_cython:
    ext_modules += [
        Extension("pyfasst.SeparateLeadStereo.tracking._tracking",
                  ["pyfasst/SeparateLeadStereo/tracking/_tracking.pyx"],
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
                  ["pyfasst/SeparateLeadStereo/tracking/_tracking.c"],
                  include_dirs=[numpy.get_include(), ],
                  language="c++"
                  # this is for mac, for the compiler to use
                  # g++ and not gcc
                  ),
        ]
    

setup(
    name='pyFASST',
    version='0.9.0',
    author='Jean-Louis Durrieu',
    author_email='jean-louis@durrieu.ch',
    packages=setuptools.find_packages(),
    scripts=[],
    url='http://github.com/wslihgt/pyfasst/',
    license='LICENSE',
    description=(
        'Python implementation of the Flexible Audio Source Separation Toolbox'+
        ' (FASST)'),
    install_requires=[
        'numpy',
        'scipy'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    
    )
