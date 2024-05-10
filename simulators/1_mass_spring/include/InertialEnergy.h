#pragma once

#include <memory>
#include <Eigen/Dense>
#include "square_mesh.h"

template <typename T, int dim>
class InertialEnergy
{
public:
    InertialEnergy(const std::vector<T> &x, const std::vector<T> &x_tilde, T m);
    ~InertialEnergy();
    InertialEnergy(InertialEnergy &&rhs);
    InertialEnergy(const InertialEnergy &rhs);
    InertialEnergy &operator=(InertialEnergy &&rhs);
    InertialEnergy &operator=(const InertialEnergy &rhs);

    void update_x(const std::vector<T> &x);
    void update_x_tilde(const std::vector<T> &x_tilde);
    void update_m(T m);
    T val();                // Calculate the value of the energy
    std::vector<T> &grad(); // Calculate the gradient of the energy
    std::vector<T> &hess(); // Calculate the Hessian matrix of the energy

private:
    // The implementation details of the VecAdder class are placed in the implementation class declared here.
    struct Impl;
    // The private pointer to the implementation class Impl
    std::unique_ptr<Impl> pimpl_;
};