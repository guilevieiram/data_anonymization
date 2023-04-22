from setuptools import setup
from Cython.Build import cythonize

setup(
    name="Project",
    ext_modules=cythonize("src/*.pyx"),
)