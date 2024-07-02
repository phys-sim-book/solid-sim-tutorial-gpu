#pragma once

#include <memory>
#include <Eigen/Dense>
#include "uti.h"
#include "device_uti.h"

template <typename T, int dim>
class NeoHookeanEnergy
{
public:
    NeoHookeanEnergy(const std::vector<T> &x, const std::vector<int> &e, T mu, T lam);
    NeoHookeanEnergy();
    ~NeoHookeanEnergy();
    NeoHookeanEnergy(NeoHookeanEnergy &&rhs);
    NeoHookeanEnergy(const NeoHookeanEnergy &rhs);
    NeoHookeanEnergy &operator=(NeoHookeanEnergy &&rhs);

    void update_x(const DeviceBuffer<T> &x);
    void init_vol_IB();
    T val();                                    // Calculate the value of the energy
    const DeviceBuffer<T> &grad();              // Calculate the gradient of the energy
    const DeviceTripletMatrix<T, 1> &hess();    // Calculate the Hessian matrix of the energy
    T init_step_size(const DeviceBuffer<T> &p); // Calculate the initial step size for the line search

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    // static __device__ void polar_svd(const Eigen::Matrix<T, dim, dim> &F, Eigen::Matrix<T, dim, dim> &U, Eigen::Matrix<T, dim, dim> &V, Eigen::Matrix<T, dim, 1> &s);
    // static __device__ Eigen::Matrix<T, dim, 1> dPsi_div_dsigma(const Eigen::Matrix<T, dim, 1> &s, T mu, T lam);
    // static __device__ Eigen::Matrix<T, dim, dim> d2Psi_div_dsigma2(const Eigen::Matrix<T, dim, 1> &s, T mu, T lam);
    // static __device__ T B_left_coef(const Eigen::Matrix<T, dim, 1> &s, T mu, T lam);
    // static __device__ T Psi(const Eigen::Matrix<T, dim, dim> &F, T mu, T lam);
    // static __device__ Eigen::Matrix<T, dim, dim> dPsi_div_dF(const Eigen::Matrix<T, dim, dim> &F, T mu, T lam);
    // static __device__ Eigen::Matrix<T, 4, 4> d2Psi_div_dF2(const Eigen::Matrix<T, dim, dim> &F, T mu, T lam);
    // static __device__ Eigen::Matrix<T, 6, 1> dPsi_div_dx(const Eigen::Matrix<T, dim, dim> &P, const Eigen::Matrix<T, dim, dim> &IB);
    // static __device__ Eigen::Matrix<T, 6, 6> d2Psi_div_dx2(const Eigen::Matrix<T, 4, 4> &dP_div_dF, const Eigen::Matrix<T, dim, dim> &IB);
};
