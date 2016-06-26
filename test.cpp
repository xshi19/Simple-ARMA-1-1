#include <iostream>
#include <iomanip>
#include <armadillo>
#include "ARMA11.hpp"

using namespace std;
using namespace arma;

int main() 
{
	// Generate random parameters
	srand (time(NULL));
	vec theta = randu<vec>(4);	
	ARMA11 arma(0.1*theta[0],-theta[1],theta[2],theta[3]);
	
	// Generate ARMA process
	vec y = arma.simulate(500);

	// Purtubations on volatiliy and drift
	/*
	y.rows(1000,1500) = 1.5*y.rows(1000,1500);
	y.rows(2000,2500) = 0.5*y.rows(2000,2500);
	y.rows(3000,3500) = y.rows(3000,3500)+0.05*randn<vec>(501);
	*/
	// Method of moments for ARMA
	ARMA11 arma_mom;
	arma_mom.fit_mom(y);

	// Maximum likelihood for ARMA
	ARMA11 arma_mle = arma_mom;
	arma_mle.fit_mle(y);
	
	// Compute residuals
	vec x = arma.residual(y);
	vec x_mom = arma_mom.residual(y);
	vec x_mle = arma_mle.residual(y);

	// Jacobians
	mat J = arma.Jacobian(x,y);
	mat J_mom = arma.Jacobian(x_mom,y);
	mat J_mle = arma.Jacobian(x_mle,y);
	
	// Compare the estimators with the true parameters
	// Row "res" are the 2-norms of residuals
	// Row "grad" are the 2-norms of gradients
	// MLE should have the smallest "res" and "grad" 
	cout.precision(5);
	cout << endl;
	cout << setw(10) << " " << setw(10) << "True" << setw(10) << "MoM" << setw(10) << "MLE" << endl;	
	cout << setw(10) << "mu" << setw(10) << arma.mu << setw(10) << arma_mom.mu << setw(10) << arma_mle.mu << endl;
	cout << setw(10) << "phi" << setw(10) << arma.phi << setw(10) << arma_mom.phi << setw(10) << arma_mle.phi << endl;
	cout << setw(10) << "psi" << setw(10) << arma.psi << setw(10) << arma_mom.psi << setw(10) << arma_mle.psi << endl;
	cout << setw(10) << "sigma" << setw(10) << arma.sigma << setw(10) << arma_mom.sigma << setw(10) << arma_mle.sigma << endl;
	cout << setw(10) << "res" << setw(10) << norm(x,2) << setw(10) << norm(x_mom,2) << setw(10) << norm(x_mle,2) << endl;
	cout << setw(10) << "grad" << setw(10) << norm(J*x,2) << setw(10) << norm(J_mom*x_mom,2) << setw(10) << norm(J_mle*x_mle,2) << endl;
	return 0;
}
