#include "simulator.h"

int main()
{
	float rho = 1000, k = 1e3, initial_stretch = 1, n_seg = 10, h = 0.02, side_len = 1, tol = 0.01;
	// printf("Running mass-spring simulator with parameters: rho = %f, k = %f, initial_stretch = %f, n_seg = %d, h = %f, side_len = %f, tol = %f\n", rho, k, initial_stretch, n_seg, h, side_len, tol);
	DirichletSimulator<float, 2> simulator(rho, side_len, initial_stretch, k, h, tol, n_seg);
	simulator.run();
}