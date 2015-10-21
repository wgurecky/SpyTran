# Execute with:
# python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [Extension("cythSn1Dcell", ["cythSn1Dcell.pyx"]),
              Extension("scattSrc", ["scattSrc.pyx"])]

setup(
    name="1D Sn",
    ext_modules=cythonize(extensions),
)
