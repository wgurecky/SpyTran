# Execute with:
# python2 setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [Extension("scattSrc", ["scattSrc.pyx"])]

setup(
    name="1D Sn",
    ext_modules=cythonize(extensions),
)
