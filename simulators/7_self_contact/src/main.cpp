#include "simulator.h"
#include "Eigen/Dense"
#include <iostream>
int main()
{
	double nu = 0.4, E = 1e5;
	double Mu = E / (2 * (1 + nu)), Lam = E * nu / ((1 + nu) * (1 - 2 * nu));
	double rho = 1000,
		   k = 4e4, initial_stretch = 1, n_seg = 6, h = 0.01, side_len = 0.45, tol = 0.01, mu = 0.4;
	SelfContactSimulator<double, 2> simulator(rho, side_len, initial_stretch, k, h, tol, mu, Mu, Lam, n_seg);
	simulator.run();
}