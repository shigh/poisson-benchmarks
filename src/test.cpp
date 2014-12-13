

#include "fftwmpi.hpp"
#include "problem.hpp"

void example_2d()
{

	const double Ly = 2*M_PI;
	const double Lx = 2*M_PI;
	const double tplx  = 2*M_PI/Lx;
	const double tplx2 = tplx*tplx;
	const double tply  = 2*M_PI/Ly;
	const double tply2 = tply*tply;

	const ptrdiff_t N0 = 100;
	const ptrdiff_t N1 = 100;

	const double dy = Ly/N0;
	const double dx = Lx/N1;

	fftw_complex *out;
	double *in;
	fftw_plan p, pi;
	int ind;
	ptrdiff_t local_n0, local_0_start, alloc_local;
	
	alloc_local = fftw_mpi_local_size_2d(N0,((int)N1/2)+1, MPI_COMM_WORLD,
										 &local_n0, &local_0_start);

	in  = (double*)fftw_malloc(sizeof(double)*2*alloc_local);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);

	const int nx_pad = 2*((int)(N1/2)+1);
	const int nx  = N1;
	const int ny  = local_n0;
	const int ny0 = local_0_start;

	double *x = new double[nx*ny];
	double *s = new double[nx*ny];
	build_problem(x, ny0, ny, dy, 0, nx, dx, 1);
	build_solution(s, ny0, ny, dy, 0, nx, dx, 1);
	
	// Size of complex array
	const int nyc = ny;
	const int nxc = (int)(nx/2+1);

	std::vector<double> kx2_vals(nxc, 0);
	for(int i=0; i<kx2_vals.size(); i++)
		kx2_vals[i] = i*i*tplx2;

	std::vector<double> ky2_vals(nyc, 0);
	for(int i=0; i<ky2_vals.size(); i++)
	{
		ind = ny0+i;
		if(2*ind<N0)
			ky2_vals[i] = ind*ind*tply2;
		else
			ky2_vals[i] = (N0-ind)*(N0-ind)*tply2;
	}

	p  = fftw_mpi_plan_dft_r2c_2d(N0, N1,  in, out, MPI_COMM_WORLD, FFTW_ESTIMATE);
	pi = fftw_mpi_plan_dft_c2r_2d(N0, N1, out, in,  MPI_COMM_WORLD, FFTW_ESTIMATE);

	for(int j=0; j<ny; j++)
		for(int i=0; i<nx; i++)
			in[j*nx_pad+i] = x[j*nx+i];

	fftw_execute(p);

	double k2;
	for(int j=0; j<nyc; j++)
		for(int i=0; i<nxc; i++)
		{
			ind = j*nxc+i;
			k2 = ky2_vals[j] + kx2_vals[i];
			if(i!=0 || j!=0)
			{
				out[ind][0] = out[ind][0]/k2;
				out[ind][1] = out[ind][1]/k2;
			}
		}
	out[0][0] = 0;
	out[0][1] = 0;

	fftw_execute(pi);

	for(int j=0; j<ny; j++)
		for(int i=0; i<nx; i++)
			x[j*nx+i] = in[j*nx_pad+i]/(N0*N1);
	
	double err = 0;
	for(int i=0; i<nx*ny; i++)
		err = std::max(err, std::abs(x[i]-s[i]));
	std::cout << err << std::endl;

	fftw_free(in); fftw_free(out);
	delete x;
}

int main(int argc, char* argv[])
{

	MPI_Init(&argc, &argv);
	fftw_mpi_init();

	example_2d();

	MPI_Finalize();

}
