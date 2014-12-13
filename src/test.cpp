

#include "fftwmpi.hpp"
#include "problem.hpp"

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
	build_problem(x, y0, ny, dy, 0, nx, dx, 20);
	build_solution(s, y0, ny, dy, 0, nx, dx, 20);

	solver.solve(x);

	double err = 0;
	for(int i=0; i<nx*ny; i++)
		err = std::max(err, std::abs(x[i]-s[i]));
	std::cout << err << std::endl;

	delete x, s;
}

int main(int argc, char* argv[])
{

	MPI_Init(&argc, &argv);
	fftw_mpi_init();

	example_2d();

	MPI_Finalize();

}
