#include "simulator.h"

int main()
{
	double rho = 1000, k = 1e5, initial_stretch = 1.4, n_seg = 10, h = 0.004, side_len = 1, tol = 0.01;
	MassSpringSimulator<double, 2> simulator(rho, side_len, initial_stretch, k, h, tol, n_seg);
	simulator.run();
}