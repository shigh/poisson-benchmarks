
from collections import namedtuple, Iterable
import numpy as np
cimport numpy as np
import scipy as sp
from libc.math cimport floor, ceil, exp, erf, fabs, sqrt, cos, sin
from libcpp cimport bool
import mpi4py

ctypedef np.float64_t DOUBLE

cdef extern from "mpi_stats.hpp":
    bool check_mpi()
    int get_rank()
    int get_comm_size()
    int thread_level()
    bool has_thread_multiple()

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

    void build_problem(double *x,
                       int zstart, int nz, double dz,
                       int ystart, int ny, double dy,
                       int xstart, int nx, double dx,
                       double k)

    void build_solution(double *x,
                        int zstart, int nz, double dz,
                        int ystart, int ny, double dy,
                        int xstart, int nx, double dx,
                        double k)

cdef extern from "fftwmpi.hpp":

    void setup_fftw_mpi()

    cdef cppclass FFTWPoisson2DMPI:

        FFTWPoisson2DMPI(ptrdiff_t N0, double Ly,
                         ptrdiff_t N1, double Lx) except +

        void solve(double *x)
        int get_nx()
        int get_ny()
        int get_y0()

    cdef cppclass FFTWPoisson3DMPI:

        FFTWPoisson3DMPI(ptrdiff_t N0, double Lz,
                         ptrdiff_t N1, double Ly,
                         ptrdiff_t N2, double Lx) except +

        void solve(double *x)
        int get_nx()
        int get_ny()
        int get_nz()
        int get_z0()

cdef extern from "hypre.hpp":

    cdef cppclass HypreSolver2D:

        HypreSolver2D(ptrdiff_t N0, double Ly,
                      ptrdiff_t N1, double Lx) except +

        void solve(double *x)
        int get_nx()
        int get_ny()
        int get_y0()

    cdef cppclass HypreSolver3D:

        HypreSolver3D(ptrdiff_t N0, double Lz,
                      ptrdiff_t N1, double Ly,
                      ptrdiff_t N2, double Lx) except +

        void solve(double *x)
        int get_nx()
        int get_ny()
        int get_nz()
        int get_z0()
        

cdef class PyFFTWPoisson2DMPI:
    cdef FFTWPoisson2DMPI *thisptr
    def __cinit__(self,
                  ptrdiff_t N0, double Ly,
                  ptrdiff_t N1, double Lx):
        setup_fftw_mpi()
        self.thisptr = new FFTWPoisson2DMPI(N0, Ly, N1, Lx)
    def __dealloc__(self):
        del self.thisptr
    property nx:
        def __get__(self): return self.thisptr.get_nx()
    property ny:
        def __get__(self): return self.thisptr.get_ny()
    property y0:
        def __get__(self): return self.thisptr.get_y0()
    
    def solve(self, double[:,:] x):
        self.thisptr.solve(&x[0,0])
        

cdef class PyFFTWPoisson3DMPI:
    cdef FFTWPoisson3DMPI *thisptr
    def __cinit__(self,
                  ptrdiff_t N0, double Lz,
                  ptrdiff_t N1, double Ly,
                  ptrdiff_t N2, double Lx):
        setup_fftw_mpi()
        self.thisptr = new FFTWPoisson3DMPI(N0, Lz, N1, Ly, N2, Lx)
    def __dealloc__(self):
        del self.thisptr
    property nx:
        def __get__(self): return self.thisptr.get_nx()
    property ny:
        def __get__(self): return self.thisptr.get_ny()
    property nz:
        def __get__(self): return self.thisptr.get_nz()
    property z0:
        def __get__(self): return self.thisptr.get_z0()
    
    def solve(self, double[:,:,:] x):
        self.thisptr.solve(&x[0,0,0])


cdef class PyHypreSolver2D:
    cdef HypreSolver2D *thisptr
    def __cinit__(self,
                  ptrdiff_t N0, double Ly,
                  ptrdiff_t N1, double Lx):
        self.thisptr = new HypreSolver2D(N0, Ly, N1, Lx)
    def __dealloc__(self):
        del self.thisptr
    property nx:
        def __get__(self): return self.thisptr.get_nx()
    property ny:
        def __get__(self): return self.thisptr.get_ny()
    property y0:
        def __get__(self): return self.thisptr.get_y0()
    
    def solve(self, double[:,:] x):
        self.thisptr.solve(&x[0,0])

cdef class PyHypreSolver3D:
    cdef HypreSolver3D *thisptr
    def __cinit__(self,
                  ptrdiff_t N0, double Lz,
                  ptrdiff_t N1, double Ly,
                  ptrdiff_t N2, double Lx):
        self.thisptr = new HypreSolver3D(N0, Lz, N1, Ly, N2, Lx)
    def __dealloc__(self):
        del self.thisptr
    property nx:
        def __get__(self): return self.thisptr.get_nx()
    property ny:
        def __get__(self): return self.thisptr.get_ny()
    property nz:
        def __get__(self): return self.thisptr.get_nz()
    property z0:
        def __get__(self): return self.thisptr.get_z0()
    
    def solve(self, double[:,:,:] x):
        self.thisptr.solve(&x[0,0,0])


MPIStats = namedtuple("MPIStats", ["check_mpi", "rank", "size",
                                   "thread_level", "has_thread_multiple"])

def mpi_stats():
    return MPIStats(check_mpi(), get_rank(), get_comm_size(),
                    thread_level(), has_thread_multiple())

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

def problem_setup_3d(int zstart, int nz, double dz,
                     int ystart, int ny, double dy,
                     int xstart, int nx, double dx,
                     double k):

    cdef double[:,:,:] x = np.zeros((nz, ny, nx), dtype=np.double)
    cdef double[:,:,:] s = np.zeros((nz, ny, nx), dtype=np.double)

    build_problem(&x[0,0,0],  zstart, nz, dz, ystart, ny, dy, xstart, nx, dx, k)
    build_solution(&s[0,0,0], zstart, nz, dz, ystart, ny, dy, xstart, nx, dx, k)

    return (np.array(x), np.array(s))
    
