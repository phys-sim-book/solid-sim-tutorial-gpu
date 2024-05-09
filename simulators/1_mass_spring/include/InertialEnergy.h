#pragma once

#include <memory>
#include <Eigen/Dense>
#include "square_mesh.h"

class InertialEnergy
{
public:
    InertialEnergy(const std::vector<float> &x, const std::vector<float> &x_tilde, float m);
    ~InertialEnergy();
    InertialEnergy(InertialEnergy &&rhs);
    InertialEnergy(const InertialEnergy &rhs);
    InertialEnergy &operator=(InertialEnergy &&rhs);
    InertialEnergy &operator=(const InertialEnergy &rhs);

    void update_x(const std::vector<float> &x);
    void update_x_tilde(const std::vector<float> &x_tilde);
    void update_m(float m);
    float val();                // Calculate the value of the energy
    std::vector<float> &grad(); // Calculate the gradient of the energy
    std::vector<float> &hess(); // Calculate the Hessian matrix of the energy

private:
    // The implementation details of the VecAdder class are placed in the implementation class declared here.
    struct Impl;
    // The private pointer to the implementation class Impl
    std::unique_ptr<Impl> pimpl_;
};