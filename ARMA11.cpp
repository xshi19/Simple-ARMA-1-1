#include "ARMA11.hpp"

ARMA11::ARMA11(double mu_, double phi_, double psi_, double sigma_) 
{
	mu = mu_;
	phi = phi_;
	psi = psi_;
	sigma = sigma_;
}

// Simulate an ARMA(1,1) process with length n
vec ARMA11::simulate(int n) 
{
	vec x = sigma*randn<vec>(n);
	vec y = x;
	
	for (int i=1; i<n; i++) {
		y[i] = x[i] + psi*x[i-1] - phi*y[i-1];
	}
	y = y+mu;
	return y;	
}

// Generate expected return of next n periods
vec ARMA11::predict(int n, vec y)
{
	int m = y.n_elem-1;
	vec x = residual(y);
	vec er(n);

	er[0] = mu-phi*(y[m]-mu)+psi*x[m];
	if (n>1) {
		for (int i=1; i<n; i++) {
			er[i] = mu-phi*er[i-1];
		}
	}	
	return er;
}

// Fit observation y via method of moments
void ARMA11::fit_mom(vec y) 
{
	double rho = 0.9;	
	double ratio1 = -1e-10;
	double ratio2 = 1e-15;	

	// Sample mean
	mu = sum(y)/double(y.n_elem);

	// First 3 autocorrelations
	vec gamma = autocorr(y-mu,3);

	// Solve other parameters from Yule-Walker equations
	phi = -gamma[2]/gamma[1];
	double b = (gamma[0]+phi*gamma[1])/(phi*gamma[0]+gamma[1])+phi;
	if (b>0) {
                psi = (b-sqrt(b*b-4.0))/2.0;
        }
        else {
                psi = (b+sqrt(b*b-4.0))/2.0;
        }
        sigma = (gamma[1]+phi*gamma[0])/psi;	

	// If abs(b)<2.0 or abs(phi)>=1.0 or sigma<=0.0, then we shrink
	// gamma[1]/gamma[0] and gamma[2]/gamma[0] to ratio1 and ratio2.
	// A full shrinkage to ratio1 and ratio2 will give 
	// phi=1e-5, b=1e5 and sigma=1 approximately. 

	while (abs(b)<2.0 || abs(phi)>=1.0 || sigma<0.0) {
		cout << "Warning: MOM does not have a solution. Autocorrelations are shrinked." << endl;
		
		gamma[1] = rho*gamma[1]+(1.0-rho)*gamma[0]*ratio1;
		gamma[2] = rho*gamma[2]+(1.0-rho)*gamma[0]*ratio2;

		phi = -gamma[2]/gamma[1];
        	b = (gamma[0]+phi*gamma[1])/(phi*gamma[0]+gamma[1])+phi;
		if (b>0) {
                	psi = (b-sqrt(b*b-4.0))/2.0;
        	}
        	else {
                	psi = (b+sqrt(b*b-4.0))/2.0;
        	}	
        	sigma = (gamma[1]+phi*gamma[0])/psi;
		rho = rho*rho;
	}
	sigma = sqrt(sigma);
}

// Simple Gauss-Newton method for ARMA(1,1) MLE
// y: observation
// WolfCoe: Wolfe condition coefficienti
// LineSize: line search step size
// FuncTol: function value tolerance
// OptiTol: first order tolerance
// StepTol: step tolerance
// MaxIter: total number of iterations 
void ARMA11::fit_mle(vec y, double WolfCoe, double LineSize, double FuncTol, double OptiTol, double StepTol, int MaxIter)
{
	// Set initial parameters as method of moments estimates
	ARMA11 temp(mu, phi, psi, sigma);	
	vec x = temp.residual(y);

	// Initialize other parameters
	ARMA11 temp_next = temp; 
	vec x_next = x;
	double alpha;
	mat J(3,y.n_elem);
	vec Jx(3);
	vec p(3);
	double L;
	double L_next;
	double L_next_approx;
	
	// Print iteration info
	cout.precision(5);	
	cout << setw(5) << "Iter" << setw(8) << "Alpha" << setw(10) << "Grad" << setw(10) << "FuncDiff" << setw(10) << "StepSize" << setw(10) << "LLH" << setw(10) << endl;

	for (int k=0; k<MaxIter; k++) {
		// Compute Jacobian matrix	
		J = temp.Jacobian(x,y);
		Jx = J*x;
		
		if (k>1 && (norm(Jx,2)<OptiTol || abs(L_next-L)/abs(L)<FuncTol || norm(alpha*p,2)<StepTol)) {
                        break;
                }
	
		// Gauss-Newton direction
		// Approxmiate Hessian with J*J' for least square problems
		p = -solve(J*J.t(),Jx);
		alpha = 1.0;
			
		// Update next parameters
		temp_next.mu = temp.mu+alpha*p[0];
		temp_next.phi = temp.phi+alpha*p[1];
		temp_next.psi = temp.psi+alpha*p[2];
		x_next = temp_next.residual(y);
			
		// Minus log-likelihood function
		L = L_next;
		L_next = 0.5*dot(x_next,x_next);

		// Taylor expansion of L_next
		L_next_approx = 0.5*dot(x,x)+WolfCoe*alpha*dot(p,Jx);
			
		// Line search step size alpha
		// Wolfe condition: L_next<=L_next_approx
		while (L_next>L_next_approx) {
			alpha = LineSize*alpha;
			temp_next.mu = temp.mu+alpha*p[0];
                        temp_next.phi = temp.phi+alpha*p[1];
                        temp_next.psi = temp.psi+alpha*p[2];
                        x_next = temp_next.residual(y);

                        L_next = 0.5*dot(x_next,x_next);
                        L_next_approx = 0.5*dot(x,x)+WolfCoe*alpha*dot(p,Jx);
		}
			
		temp = temp_next;
		x = x_next;
		if (k>0) {	
			cout << setw(5) << fixed << k << setw(8) << fixed << alpha << setw(10) << fixed << norm(Jx,2) << setw(10) << fixed << (L_next-L)/abs(L) << setw(10) << fixed << norm(alpha*p,2) << setw(10) << fixed << L_next << endl;
		}
	}
	mu = temp.mu;
	phi = temp.phi;
	psi = temp.psi;
	sigma = sqrt(dot(x,x)/y.n_elem);	
}

// Compute ARMA(1,1) residual from observation y
vec ARMA11::residual(vec y)
{
	y = y-mu;
	vec x = y;
	for (int i=1; i<y.n_elem; i++) {
		x[i] = y[i] + phi*y[i-1] - psi*x[i-1];
	}
	return x;
}

// Compute ARMA(1,1) residual using residual x and observation y
mat ARMA11::Jacobian(vec x, vec y)
{
	mat J(3,y.n_elem);
	J(0,0) = -1;
	J(1,0) = 0;
	J(2,0) = 0;

	for (int i=1; i<y.n_elem; i++) {
		J(0,i) = -1-phi-psi*J(0,i-1);
		J(1,i) = y[i-1]-psi*J(1,i-1);
		J(2,i) = -x[i-1]-psi*J(2,i-1);
	}

	return J;
}

// Compute first n autocorrelations of a data x
vec autocorr(vec x, int n) 
{
	vec gamma(n);
	int m = x.n_elem;
	gamma.zeros();
	
	if (n>m) { n = m; }
	
	for (int i=0; i<n; i++) {
		gamma[i] = dot(x.rows(i,m-1), x.rows(0,m-1-i))/double(m-i);
	}
	return gamma;
}
