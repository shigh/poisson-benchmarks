
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
	fftw_destroy_plan(p);
	fftw_destroy_plan(pi);

}


FFTWPoisson3DMPI::FFTWPoisson3DMPI(ptrdiff_t N0_, double Lz_,
								   ptrdiff_t N1_, double Ly_,
								   ptrdiff_t N2_, double Lx_):
	N0(N0_), Lz(Lz_), N1(N1_), Ly(Ly_), N2(N2_), Lx(Lx_)
{

	// Allocate and set up fftw structures
	alloc_local = fftw_mpi_local_size_3d(N0, N1,((int)N2/2)+1, MPI_COMM_WORLD,
										 &local_n0, &local_0_start);

	in  = (double*)fftw_malloc(sizeof(double)*2*alloc_local);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);

	p  = fftw_mpi_plan_dft_r2c_3d(N0, N1, N2,  in, out, MPI_COMM_WORLD, FFTW_ESTIMATE);
	pi = fftw_mpi_plan_dft_c2r_3d(N0, N1, N2, out, in,  MPI_COMM_WORLD, FFTW_ESTIMATE);

	nx_pad = 2*((int)(N2/2)+1);
	nx  = N2;
	ny  = N1;
	nz  = local_n0;
	z0  = local_0_start;
	nzc = nz;
	nyc = ny;
	nxc = (int)(nx/2+1);

	// Build k val vectors
	int ind;
	const double tplx  = 2*M_PI/Lx;
	const double tplx2 = tplx*tplx;
	const double tply  = 2*M_PI/Ly;
	const double tply2 = tply*tply;
	const double tplz  = 2*M_PI/Lz;
	const double tplz2 = tplz*tplz;

	kx2_vals.resize(nxc);
	for(int i=0; i<kx2_vals.size(); i++)
		kx2_vals[i] = i*i*tplx2;

	ky2_vals.resize(nyc);
	for(int i=0; i<ky2_vals.size(); i++)
	{
		ind = i;
		if(2*ind<N1)
			ky2_vals[i] = ind*ind*tply2;
		else
			ky2_vals[i] = (N1-ind)*(N1-ind)*tply2;
	}

	kz2_vals.resize(nzc);
	for(int i=0; i<kz2_vals.size(); i++)
	{
		ind = z0+i;
		if(2*ind<N0)
			kz2_vals[i] = ind*ind*tplz2;
		else
			kz2_vals[i] = (N0-ind)*(N0-ind)*tplz2;
	}

}

void FFTWPoisson3DMPI::solve(double *x)
{

	int ind;

	for(int k=0; k<nz; k++)
		for(int j=0; j<ny; j++)
			for(int i=0; i<nx; i++)
				in[(k*ny+j)*nx_pad+i] = x[(k*ny+j)*nx+i];

	fftw_execute(p);

	double k2;
	for(int k=0; k<nzc; k++)
		for(int j=0; j<nyc; j++)
			for(int i=0; i<nxc; i++)
			{
				ind = (k*nyc+j)*nxc+i;
				k2 = kz2_vals[k] + ky2_vals[j] + kx2_vals[i];
				if((i!=0 || j!=0 || k!=0) && k2!=0)
				{
					out[ind][0] = out[ind][0]/k2;
					out[ind][1] = out[ind][1]/k2;
				}
			}
	out[0][0] = 0;
	out[0][1] = 0;

	fftw_execute(pi);

	for(int k=0; k<nz; k++)
		for(int j=0; j<ny; j++)
			for(int i=0; i<nx; i++)
				x[(k*ny+j)*nx+i] = in[(k*ny+j)*nx_pad+i]/(N0*N1*N2);

}

int FFTWPoisson3DMPI::get_nx()
{
	return nx;
}

int FFTWPoisson3DMPI::get_ny()
{
	return ny;
}

int FFTWPoisson3DMPI::get_nz()
{
	return nz;
}

int FFTWPoisson3DMPI::get_z0()
{
	return z0;
}

FFTWPoisson3DMPI::~FFTWPoisson3DMPI()
{

	fftw_free(in);
	fftw_free(out);
	fftw_destroy_plan(p);
	fftw_destroy_plan(pi);

}
