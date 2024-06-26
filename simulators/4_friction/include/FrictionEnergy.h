#pragma once

#include <memory>
#include <Eigen/Dense>
#include <vector>
#include "uti.h"

template <typename T, int dim>
class FrictionEnergy
{
public:
    FrictionEnergy(const std::vector<T> &v, const std::vector<T> &mu_lambda, T hhat, const Eigen::Matrix<T, dim, 1> &n);
    FrictionEnergy();
    ~FrictionEnergy();
    FrictionEnergy(FrictionEnergy &&rhs);
    FrictionEnergy(const FrictionEnergy &rhs);
    FrictionEnergy &operator=(FrictionEnergy &&rhs);

    void update_v(const DeviceBuffer<T> &v);
    T val();                                 // Calculate the value of the energy
    const DeviceBuffer<T> &grad();           // Calculate the gradient of the energy
    const DeviceTripletMatrix<T, 1> &hess(); // Calculate the Hessian matrix of the energy

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    T f0(T vbarnorm, T Epsv, T hhat);
    T f1_div_vbarnorm(T vbarnorm, T Epsv);
    T f_hess_term(T vbarnorm, T Epsv);
};
