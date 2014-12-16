
import numpy as np
from testsolvers import *
from mpi4py import MPI
import sys
import os
import time

comm = MPI.COMM_WORLD
rank = comm.rank
host = os.environ["HOSTNAME"]

solver_num = int(sys.argv[1])
if solver_num==0:
    SolverClass = PyFFTWPoisson3DMPI
    solver_name = "FFTW"
else:
    SolverClass = PyHypreSolver3D
    solver_name = "HYPRE"

N = int(sys.argv[2])
N0 = N1 = N2 = N
k = 10
L = 2*np.pi
dz = L/N0
dy = L/N1
dx = L/N2

comm.Barrier()

build_start = time.clock()
solver = SolverClass(N0, L, N1, L, N2, L)
build_end   = time.clock()

nz = solver.nz
ny = solver.ny
nx = solver.nx
z0 = solver.z0

x, s = problem_setup_3d(z0, nz, dz, 0, ny, dy, 0, nx, dx, k) 
x0 = x.copy()

comm.Barrier()

solve_start = time.clock()
solver.solve(x)
solve_end   = time.clock()

build_times = comm.gather(build_end-build_start, root=0)
solve_times = comm.gather(solve_end-solve_start, root=0)
host_names = comm.gather(host, root=0)
if rank==0:
    print solver_name, N, sys.argv[3], np.max(build_times), np.max(solve_times), len(np.unique(host_names))

