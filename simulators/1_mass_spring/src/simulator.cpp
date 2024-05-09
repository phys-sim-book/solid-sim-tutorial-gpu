#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include "InertialEnergy.h"
#include <iostream>

int main()
{
	int size = 3;
	std::vector<float> x(size);
	std::vector<float> x_t(size);
	std::vector<float> grad(3);
	for (int i = 0; i < size; i++)
	{
		x[i] = i;
		x_t[i] = i + 2;
	}
	float m = 1;
	InertialEnergy ie(x, x_t, m);
	grad = ie.grad();
	for (int i = 0; i < size; i++)
	{
		std::cout << grad[i] << std::endl;
	}
	float val = ie.val();
	std::cout << val << std::endl;
	return 0;
}
