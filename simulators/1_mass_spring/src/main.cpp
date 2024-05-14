#include "simulator.cpp"

int main()
{
	float rho = 1000, k = 1e5, initial_stretch = 1.4, n_seg = 20, h = 0.004, side_len = 2, tol = 0.01;
	// printf("Running mass-spring simulator with parameters: rho = %f, k = %f, initial_stretch = %f, n_seg = %d, h = %f, side_len = %f, tol = %f\n", rho, k, initial_stretch, n_seg, h, side_len, tol);
	MassSpringSimulator<float, 2> simulator(rho, side_len, initial_stretch, k, h, tol, n_seg);
	simulator.run();
}