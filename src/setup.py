from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy as np

mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("mpic++ --showme:link").read().strip().split(' ')
link_args = ["-L/home/shigh2/src/fftw/lib", "-lfftw3", "-lfftw3_mpi", "-lm"]

setup(
    cmdclass = {'build_ext':build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [Extension("testsolvers",
                             ["testsolvers.pyx"],
                             extra_objects=["fftwlocal.o", "problem.o", "mpi_stats.o",
                                            "fftwmpi.o"],
                             extra_compile_args=mpi_compile_args,
                             extra_link_args=link_args+mpi_link_args,
                             language="c++"),]
    )
