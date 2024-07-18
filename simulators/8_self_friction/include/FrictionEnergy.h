#pragma once

#include <memory>
#include <Eigen/Dense>
#include <vector>
#include "uti.h"

template <typename T, int dim>
class FrictionEnergy
{
public:
    FrictionEnergy(const std::vector<T> &v, T hhat, const std::vector<T> &n, const std::vector<int>& bp, const std::vector<int>& be, int npe);
    FrictionEnergy();
    ~FrictionEnergy();
    FrictionEnergy(FrictionEnergy &&rhs);
    FrictionEnergy(const FrictionEnergy &rhs);
    FrictionEnergy &operator=(FrictionEnergy &&rhs);

    void update_v(const DeviceBuffer<T> &v);
    DeviceBuffer<T> &get_mu_lambda();
    DeviceBuffer<T>& get_mu_lambda_self();
    DeviceBuffer<Eigen::Matrix<T, 2, 1>>& get_n_self();
    DeviceBuffer<T>& get_r_self();
    T val();                                 // Calculate the value of the energy
    const DeviceBuffer<T> &grad();           // Calculate the gradient of the energy
    const DeviceTripletMatrix<T, 1> &hess(); // Calculate the Hessian matrix of the energy

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    T __device__ f0(T vbarnorm, T Epsv, T hhat);
    T __device__ f1_div_vbarnorm(T vbarnorm, T Epsv);
    T __device__ f_hess_term(T vbarnorm, T Epsv);
};
