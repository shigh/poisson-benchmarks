
#include "fftwlocal.hpp"

void solve_1d_mpi(int N, double *x, double L)
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
