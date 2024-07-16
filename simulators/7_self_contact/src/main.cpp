#include "simulator.h"
#include "Eigen/Dense"
#include <iostream>
int main()
{
	double nu = 0.4, E = 1e5;
	double Mu = E / (2 * (1 + nu)), Lam = E * nu / ((1 + nu) * (1 - 2 * nu));
	double rho = 1000,
		   k = 4e4, initial_stretch = 1, n_seg = 4, h = 0.01, side_len = 0.45, tol = 0.01, mu = 0.4;
	// printf("Running mass-spring simulator with parameters: rho = %f, k = %f, initial_stretch = %f, n_seg = %d, h = %f, side_len = %f, tol = %f\n", rho, k, initial_stretch, n_seg, h, side_len, tol);
	SelfContactSimulator<double, 2> simulator(rho, side_len, initial_stretch, k, h, tol, mu, Mu, Lam, n_seg);
	simulator.run();
}