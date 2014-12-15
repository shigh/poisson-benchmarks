from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy as np

HYPRE_DIR = os.environ["HYPRE_DIR"]
FFTW_DIR  = os.environ["FFTW_DIR"]

mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("mpic++ --showme:link").read().strip().split(' ')
compile_args = ["-I%s/include"%(HYPRE_DIR,), "-I%s/include"%(FFTW_DIR,)]
link_args = ["-L%s/lib"%(FFTW_DIR,), "-lfftw3", "-lfftw3_mpi", "-lm",
             "-L%s/lib"%(HYPRE_DIR,), "-lHYPRE"]

setup(
    cmdclass = {'build_ext':build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [Extension("testsolvers",
                             ["testsolvers.pyx"],
                             extra_objects=["fftwlocal.o", "problem.o", "mpi_stats.o",
                                            "fftwmpi.o", "hypre.o"],
                             extra_compile_args=compile_args+mpi_compile_args,
                             extra_link_args=link_args+mpi_link_args,
                             language="c++"),]
    )
