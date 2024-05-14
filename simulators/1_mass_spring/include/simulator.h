#pragma once

#include <vector>
#include <cmath>
#include "square_mesh.h"
#include <iostream>
template <typename T, int dim>
class MassSpringSimulator
{
public:
    MassSpringSimulator();
    ~MassSpringSimulator();
    MassSpringSimulator(MassSpringSimulator &&rhs);
    MassSpringSimulator &operator=(MassSpringSimulator &&rhs);
    MassSpringSimulator(T rho, T side_len, T initial_stretch, T K, T h, T tol, int n_seg);
    void run();

private:
    // The implementation details of the VecAdder class are placed in the implementation class declared here.
    struct Impl;
    // The private pointer to the implementation class Impl
    std::unique_ptr<Impl> pimpl_;
};
