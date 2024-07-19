#include "simulator.h"

int main()
{
	double rho = 1000, k = 2e4, initial_stretch = 1, n_seg = 1, h = 0.01, side_len = 1, tol = 0.01;
	ContactSimulator<double, 2> simulator(rho, side_len, initial_stretch, k, h, tol, n_seg);
	simulator.run();
}