#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "uti.h"
#include "device_uti.h"

template <typename T, int dim>
class SpringEnergy
{
public:
    SpringEnergy(const std::vector<T> &x, const std::vector<T> &m, const std::vector<int> &DBC, const std::vector<Eigen::Matrix<T, dim, 1>> &DBC_target, T k);
    SpringEnergy();
    ~SpringEnergy();
    SpringEnergy(SpringEnergy &&rhs);
    SpringEnergy(const SpringEnergy &rhs);
    SpringEnergy &operator=(SpringEnergy &&rhs);

    void update_x(const DeviceBuffer<T> &x);
    T val();                                 // Calculate the value of the energy
    const DeviceBuffer<T> &grad();           // Calculate the gradient of the energy
    const DeviceTripletMatrix<T, 1> &hess(); // Calculate the Hessian matrix of the energy

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
