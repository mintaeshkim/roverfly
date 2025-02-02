from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("./envs/utility_functions.pyx")
)
