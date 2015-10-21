from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions=[Extension("sn1Dcell", ["sn1Dcell.pyx"])]

setup(
    name="1D Sn",
    ext_modules=cythonize(extensions),
)
