#ifndef ARMA11_H
#define ARMA11_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <armadillo>

using namespace std;
using namespace arma;

// Simple ARMA(1,1) class
class ARMA11
{
public: 
	// Parameters
	double mu;
	double phi;
	double psi;
	double sigma;

	// Initializer
	ARMA11(double mu=0, double phi=0, double psi=0, double sigma=1.0);

	// Simulation
        vec simulate(int n);

        // Prediction
        vec predict(int n, vec y);
	
	// Fit data with method of moments
	void fit_mom(vec y);

	// Fit data with conditional maximum likelihood (least squares)
	void fit_mle(vec y, double WolfCoe=1e-5, double LineSize=0.8, double FuncTol=1e-6, double OptiTol=1e-6, double StepTol=1e-10, int MaxIter=100);
	
	// Extract residuals
	vec residual(vec y);
	
	// Compute Jacobian matrix
	mat Jacobian(vec x, vec y);
};

// Autocorrelation function
vec autocorr(vec x, int n);	

#endif
