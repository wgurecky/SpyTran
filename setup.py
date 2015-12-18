# SPyTran Install script
#
# Use:
# ----
# python setup.py install --user
# or
# python setup.py develop --user

from setuptools import setup, find_packages
setup(
    name="SPyTran",
    version="0.1",
    packages=find_packages(),
    install_requires=['numpy>=1.7.0', 'scipy>=0.12.0', 'h5py'],
    optional_required={'plotting': ['matplotlib']},
    package_data={'': ['*.txt']},
    author='William Gurecky',
    license="BSD",
    author_email="william.gurecky@gmail.com",
)
