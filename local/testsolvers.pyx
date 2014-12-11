
import numpy as np
cimport numpy as np
import scipy as sp
from libc.math cimport floor, ceil, exp, erf, fabs, sqrt, cos, sin

ctypedef np.float64_t DOUBLE

cdef extern from "solvers.hpp":

    void solve_1d(int N, double *x, double L)

def solve1d(double[:] x, double L):
    cdef int N = len(x)
    solve_1d(N, &x[0], L)



