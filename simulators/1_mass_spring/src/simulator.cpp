#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include "InertialEnergy.h"
#include "MassSpringEnergy.h"
#include <iostream>

int main()
{
	int size = 3;
	std::vector<float> x(size * 2);
	std::vector<float> x_t(size * 2);
	std::vector<float> grad(3);
	for (int i = 0; i < size; i++)
	{
		x[2 * i] = i;
		x[2 * i + 1] = i + 1;
		x_t[2 * i] = i + 2;
		x_t[2 * i + 1] = i + 3;
	}
	float m = 1;
	InertialEnergy<float, 2> ie(x, x_t, m);
	grad = ie.grad();
	for (int i = 0; i < size; i++)
	{
		std::cout << grad[2 * i] << " " << grad[2 * i + 1] << std::endl;
	}
	float val = ie.val();
	std::cout << val << std::endl;

	std::vector<int> edge(size * 2);
	for (int i = 0; i < size; i++)
	{
		edge[2 * i] = i;
		edge[2 * i + 1] = (i + 1) % size;
	}
	std::vector<float> k(size);
	std::vector<float> l2(size);
	for (int i = 0; i < size; i++)
	{
		k[i] = 1;
		l2[i] = 1;
	}

	MassSpringEnergy<float, 2> mse(x, edge, l2, k);
	grad = mse.grad();
	for (int i = 0; i < size; i++)
	{
		std::cout << grad[2 * i] << " " << grad[2 * i + 1] << std::endl;
	}
	val = mse.val();
	std::cout << val << std::endl;
	std::vector<float> hess(4 * size * size);
	hess = mse.hess();
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < 4; k++)
				std::cout << hess[4 * size * i + 4 * j + k] << " ";
			std::cout << std::endl;
		}
	}
	return 0;
}
