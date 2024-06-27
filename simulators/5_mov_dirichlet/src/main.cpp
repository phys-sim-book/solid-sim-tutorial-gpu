#include "simulator.h"

int main()
{
	float rho = 1000, k = 4e4, initial_stretch = 1, n_seg = 3, h = 0.01, side_len = 1, tol = 0.01, mu = 0.11;
	// printf("Running mass-spring simulator with parameters: rho = %f, k = %f, initial_stretch = %f, n_seg = %d, h = %f, side_len = %f, tol = %f\n", rho, k, initial_stretch, n_seg, h, side_len, tol);
	MovDirichletSimulator<float, 2> simulator(rho, side_len, initial_stretch, k, h, tol, mu, n_seg);
	simulator.run();
}