
import numpy as np
cimport numpy as np
import scipy as sp
from libc.math cimport floor, ceil, exp, erf, fabs, sqrt, cos, sin

ctypedef np.float64_t DOUBLE

cdef extern from "fftwlocal.hpp":

    void solve_1d(int N, double *x, double L)

    void solve_2d(int ny, int nx, double *x, double Ly, double Lx)

    void solve_3d(int nz, int ny, int nx, double *x, double Lz, double Ly, double Lx)

cdef extern from "problem.hpp":

    void build_problem(double *x,
                       int xstart, int nx, double dx,
                       double k)

    void build_solution(double *x,
                        int xstart, int nx, double dx,
                        double k)


    void build_problem(double *x,
                       int ystart, int ny, double dy,
                       int xstart, int nx, double dx,
                       double k)

    void build_solution(double *x,
                        int ystart, int ny, double dy,
                        int xstart, int nx, double dx,
                        double k)


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

def problem_setup_1d(int xstart, int nx, double dx,
                     double k):

    cdef double[:] x = np.zeros(nx, dtype=np.double)
    cdef double[:] s = np.zeros(nx, dtype=np.double)

    build_problem(&x[0],  xstart, nx, dx, k)
    build_solution(&s[0], xstart, nx, dx, k)

    return (np.array(x), np.array(s))

def problem_setup_2d(int ystart, int ny, double dy,
                     int xstart, int nx, double dx,
                     double k):

    cdef double[:,:] x = np.zeros((ny, nx), dtype=np.double)
    cdef double[:,:] s = np.zeros((ny, nx), dtype=np.double)

    build_problem(&x[0,0],  ystart, ny, dy, xstart, nx, dx, k)
    build_solution(&s[0,0], ystart, ny, dy, xstart, nx, dx, k)

    return (np.array(x), np.array(s))
