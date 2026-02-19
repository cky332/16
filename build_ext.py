"""Pre-compile Cython extensions (exp_utils/levenshtein.pyx).

Run this once before using accelerate launch:
    python build_ext.py
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "exp_utils.levenshtein",
        sources=["exp_utils/levenshtein.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    script_args=["build_ext", "--inplace"],
)
