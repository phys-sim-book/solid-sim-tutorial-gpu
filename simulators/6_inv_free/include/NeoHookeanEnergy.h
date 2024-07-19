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
};
