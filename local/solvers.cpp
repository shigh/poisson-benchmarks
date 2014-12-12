
#include "solvers.hpp"

void solve_1d(int N, double *x, double L)
{

	fftw_complex *out;
	double *in;
	fftw_plan p, pi;
	const double dx = L/N;
	const double tpl = 2*M_PI/L;
	const double tpl2 = tpl*tpl;
	// Size of complex array
	const double Nc = (int)(floor(N/2)+1);

	std::vector<double> k2_vals(Nc, 0);
	for(int i=0; i<k2_vals.size(); i++)
		k2_vals[i] = i*i*tpl2;

	in  = (double*)fftw_malloc(sizeof(double)*N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*Nc);

	p  = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
	pi = fftw_plan_dft_c2r_1d(N, out, in, FFTW_ESTIMATE);

	for(int i=0; i<N; i++)
		in[i] = x[i];

	fftw_execute(p);

	for(int i=1; i<Nc; i++)
	{
		out[i][0] = out[i][0]/k2_vals[i];
		out[i][1] = out[i][1]/k2_vals[i];
	}
	out[0][0] = out[0][1] = 0;

	fftw_execute(pi);

	for(int i=0; i<N; i++)
		x[i] = in[i]/N;

	fftw_free(in); fftw_free(out);
	
}


void solve_2d(int ny, int nx, double *x, double Ly, double Lx)
{

	fftw_complex *out, *in;
	//double *in;
	fftw_plan p, pi;
	const double tplx  = 2*M_PI/Lx;
	const double tplx2 = tplx*tplx;
	const double tply  = 2*M_PI/Ly;
	const double tply2 = tply*tply;
	
	// Size of complex array
	const int nyc = ny;
	const int nxc = nx;//(int)(floor(nx/2)+1);

	std::vector<double> kx2_vals(nxc, 0);
	for(int i=0; i<kx2_vals.size(); i++)
	{
		if(2*i<nx)
			kx2_vals[i] = i*i*tplx2;
		else
			kx2_vals[i] = (nx-i)*(nx-i)*tplx2;
	}

	std::vector<double> ky2_vals(nyc, 0);
	for(int i=0; i<ky2_vals.size(); i++)
	{
		if(2*i<ny)
			ky2_vals[i] = i*i*tply2;
		else
			ky2_vals[i] = (ny-i)*(ny-i)*tply2;
	}

	//in  = (double*)fftw_malloc(sizeof(double)*nx*ny);
	in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nxc*nyc);

	// p  = fftw_plan_dft_r2c_2d(ny,  nx,  in, out, FFTW_ESTIMATE);
	// pi = fftw_plan_dft_c2r_2d(nyc, nxc, out, in, FFTW_ESTIMATE);
	p  = fftw_plan_dft_2d(ny,  nx,  in, out, FFTW_FORWARD,  FFTW_ESTIMATE);
	pi = fftw_plan_dft_2d(nyc, nxc, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

	for(int i=0; i<ny*nx; i++)
	{
		in[i][0] = x[i];
		in[i][1] = 0;
	}

	fftw_execute(p);

	double k2;
	int ind;
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

	for(int i=0; i<ny*nx; i++)
		x[i] = in[i][0]/(ny*nx);

	fftw_free(in); fftw_free(out);
	
}
