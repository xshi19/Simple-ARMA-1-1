#include <iostream>
#include <iomanip>
#include <armadillo>
#include "ARMA11.hpp"

using namespace std;
using namespace arma;

int main()
{
	vec r;
	r.load("sp500ret.csv",csv_ascii);

	int win_size = 2000;
	int periods = 100;
	int t_start = r.n_elem-periods;
	
	ARMA11 arma_mom;
	ARMA11 arma_mle;
	vec x_mom;
	vec x_mle;

	mat results = zeros<mat>(periods,5);
	mat params = zeros<mat>(periods,7);
	for (int t=t_start; t<r.n_elem; t++) {
		arma_mom.fit_mom(r.rows(t-500,t-1));
		x_mom = arma_mom.residual(r.rows(t-500,t-1));
		
	//	arma_mle = arma_mom;
		arma_mle.fit_mle(r.rows(t-500,t-1));
		x_mle = arma_mle.residual(r.rows(t-500,t-1));	
	
		// cout << r[t] << "," << arma_mom.predict(1,r.rows(t-500,t)) << "," << arma_mle.predict(1,r.rows(t-500,t)) << endl;	
		results(t-t_start,0) = r[t];
		results(t-t_start,1) = arma_mom.predict(1,r.rows(t-500,t-1))[0];
		results(t-t_start,2) = arma_mle.predict(1,r.rows(t-500,t-1))[0];
		results(t-t_start,3) = norm(x_mom,2);
		results(t-t_start,4) = norm(x_mle,2);

		params(t-t_start,0) = r[t];
                params(t-t_start,1) = arma_mle.phi;
        	params(t-t_start,2) = arma_mle.psi;        
		params(t-t_start,3) = arma_mle.mu;
		params(t-t_start,4) = arma_mle.sigma;
		params(t-t_start,5) = x_mle[x_mle.n_elem-1];
		params(t-t_start,6) = arma_mle.predict(1,r.rows(t-500,t-1))[0];
	}
	
	cout << results.rows(0,5)  << endl;
	cout << params.rows(0,5) << endl;
	results.save("results.csv", csv_ascii);
	params.save("params.csv", csv_ascii);
	return 0;
}
