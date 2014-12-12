
import numpy as np
cimport numpy as np
import scipy as sp
from libc.math cimport floor, ceil, exp, erf, fabs, sqrt, cos, sin

ctypedef np.float64_t DOUBLE

cdef extern from "solvers.hpp":

    void solve_1d(int N, double *x, double L)

    void solve_2d(int ny, int nx, double *x, double Ly, double Lx)

    void solve_3d(int nz, int ny, int nx, double *x, double Lz, double Ly, double Lx)

def solve1d(double[:] x, double L):
    cdef int N = len(x)
    solve_1d(N, &x[0], L)

def solve2d(double[:,:] x, double Ly, double Lx):
    cdef int ny, nx
    ny = x.shape[0]
    nx = x.shape[1]
    solve_2d(ny, nx, &x[0,0], Ly, Lx)

def solve3d(double[:,:,:] x, double Lz, double Ly, double Lx):
    cdef int nz, ny, nx
    nz = x.shape[0]
    ny = x.shape[1]
    nx = x.shape[2]
    solve_3d(nz, ny, nx, &x[0,0,0], Lz, Ly, Lx)
