

#include "fftwmpi.hpp"
#include "problem.hpp"
#include "hypre.hpp"

void example_2d()
{

	const double Ly = 2*M_PI;
	const double Lx = 2*M_PI;

	const ptrdiff_t N0 = 100;
	const ptrdiff_t N1 = 100;

	const double dy = Ly/N0;
	const double dx = Lx/N1;

	FFTWPoisson2DMPI solver(N0, Ly, N1, Lx);

	const int nx = solver.get_nx();
	const int ny = solver.get_ny();
	const int y0 = solver.get_y0();

	double *x = new double[nx*ny];
	double *s = new double[nx*ny];
	build_problem(x, y0, ny, dy, 0, nx, dx, 10);
	build_solution(s, y0, ny, dy, 0, nx, dx, 10);

	solver.solve(x);

	double err = 0;
	for(int i=0; i<nx*ny; i++)
		err = std::max(err, std::abs(x[i]-s[i]));
	std::cout << err << std::endl;

	delete x, s;

}

void example_3d()
{

	const double Lz = 2*M_PI;
	const double Ly = 2*M_PI;
	const double Lx = 2*M_PI;

	const ptrdiff_t N0 = 100;
	const ptrdiff_t N1 = 100;
	const ptrdiff_t N2 = 100;

	const double dz = Lz/N0;
	const double dy = Ly/N1;
	const double dx = Lx/N2;

	FFTWPoisson3DMPI solver(N0, Lz, N1, Ly, N2, Lx);

	const int nz = solver.get_nz();
	const int nx = solver.get_nx();
	const int ny = solver.get_ny();
	const int z0 = solver.get_z0();

	double *x = new double[nx*ny*nz];
	double *s = new double[nx*ny*nz];
	build_problem(x, z0, nz, dz, 0, ny, dy, 0, nx, dx, 10);
	build_solution(s, z0, nz, dz, 0, ny, dy, 0, nx, dx, 10);

	solver.solve(x);

	double err = 0;
	for(int i=0; i<nx*ny*nz; i++)
		err = std::max(err, std::abs(x[i]-s[i]));
	std::cout << err << std::endl;

	delete x, s;

}


int main(int argc, char* argv[])
{

	MPI_Init(&argc, &argv);
	fftw_mpi_init();

	example_2d();

	example_3d();

	hypre_solve(argc, argv);

	MPI_Finalize();

}
