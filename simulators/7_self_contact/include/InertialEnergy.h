#pragma once

#include <memory>
#include <Eigen/Dense>
#include "square_mesh.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/ext/linear_system.h>

using namespace muda;

template <typename T, int dim>
class InertialEnergy
{
public:
    InertialEnergy(int N, T m);
    InertialEnergy();
    ~InertialEnergy();
    InertialEnergy(InertialEnergy &&rhs);
    InertialEnergy(const InertialEnergy &rhs);
    InertialEnergy &operator=(InertialEnergy &&rhs);

    void update_x(const DeviceBuffer<T> &x);
    void generate_hess();
    void update_x_tilde(const DeviceBuffer<T> &x_tilde);
    void update_m(T m);
    T val();                                 // Calculate the value of the energy
    const DeviceBuffer<T> &grad();           // Calculate the gradient of the energy
    const DeviceTripletMatrix<T, 1> &hess(); // Calculate the Hessian matrix of the energy

private:
    // The implementation details of the VecAdder class are placed in the implementation class declared here.
    struct Impl;
    // The private pointer to the implementation class Impl
    std::unique_ptr<Impl> pimpl_;
};
