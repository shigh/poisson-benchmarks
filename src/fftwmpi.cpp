
#include "fftwmpi.hpp"

void setup_fftw_mpi()
{
	fftw_mpi_init();
}

void cleanup_fftw_mpi()
{
	fftw_mpi_cleanup();
}

FFTWPoisson2DMPI::FFTWPoisson2DMPI(ptrdiff_t N0_, double Ly_, ptrdiff_t N1_, double Lx_):
	N0(N0_), Ly(Ly_), N1(N1_), Lx(Lx_)
{

	// Allocate and set up fftw structures
	alloc_local = fftw_mpi_local_size_2d(N0,((int)N1/2)+1, MPI_COMM_WORLD,
										 &local_n0, &local_0_start);

	in  = (double*)fftw_malloc(sizeof(double)*2*alloc_local);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);

	p  = fftw_mpi_plan_dft_r2c_2d(N0, N1,  in, out, MPI_COMM_WORLD, FFTW_ESTIMATE);
	pi = fftw_mpi_plan_dft_c2r_2d(N0, N1, out, in,  MPI_COMM_WORLD, FFTW_ESTIMATE);

	nx_pad = 2*((int)(N1/2)+1);
	nx  = N1;
	ny  = local_n0;
	y0 = local_0_start;
	nyc = ny;
	nxc = (int)(nx/2+1);


	// Build k val vectors
	int ind;
	const double tplx  = 2*M_PI/Lx;
	const double tplx2 = tplx*tplx;
	const double tply  = 2*M_PI/Ly;
	const double tply2 = tply*tply;

	kx2_vals.resize(nxc);
	for(int i=0; i<kx2_vals.size(); i++)
		kx2_vals[i] = i*i*tplx2;

	ky2_vals.resize(nyc);
	for(int i=0; i<ky2_vals.size(); i++)
	{
		ind = y0+i;
		if(2*ind<N0)
			ky2_vals[i] = ind*ind*tply2;
		else
			ky2_vals[i] = (N0-ind)*(N0-ind)*tply2;
	}



}

void FFTWPoisson2DMPI::solve(double *x)
{

	int ind;

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

}

int FFTWPoisson2DMPI::get_nx()
{
	return nx;
}

int FFTWPoisson2DMPI::get_ny()
{
	return ny;
}

int FFTWPoisson2DMPI::get_y0()
{
	return y0;
}

FFTWPoisson2DMPI::~FFTWPoisson2DMPI()
{

	fftw_free(in);
	fftw_free(out);

}
