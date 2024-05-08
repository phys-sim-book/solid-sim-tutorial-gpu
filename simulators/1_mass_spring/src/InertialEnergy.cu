#include "InertialEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>

using namespace muda;

struct InertialEnergy::Impl
{
    DeviceBuffer<float> device_x, device_x_tilde, device_grad, device_hess;
    int N;
    float m, val;
    Eigen::VectorXf host_x, host_x_tilde, host_grad;
    Eigen::MatrixXf host_hess;
};

InertialEnergy::~InertialEnergy() = default;

InertialEnergy::InertialEnergy(InertialEnergy &&rhs) = default;

InertialEnergy &InertialEnergy::operator=(InertialEnergy &&rhs) = default;

InertialEnergy::InertialEnergy(const InertialEnergy &rhs)
    : pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

InertialEnergy &InertialEnergy::operator=(const InertialEnergy &rhs)
{
    *pimpl_ = *rhs.pimpl_;
    return *this;
};

InertialEnergy::InertialEnergy(const Eigen::VectorXf &x, const Eigen::VectorXf &x_tilde, float m) : pimpl_{std::make_unique<Impl>()}
{
    pimpl_->host_x = x;
    pimpl_->host_x_tilde = x_tilde;
    pimpl_->m = m;
    pimpl_->N = x.size();
    pimpl_->device_x.copy_from(std::vector<float>(x.data(), x.data() + x.size()));
    pimpl_->device_x_tilde.copy_from(std::vector<float>(x_tilde.data(), x_tilde.data() + x_tilde.size()));
    pimpl_->device_grad = std::vector<float>(pimpl_->N);
    pimpl_->device_hess = std::vector<float>(pimpl_->N * pimpl_->N);
    pimpl_->host_grad = Eigen::VectorXf(pimpl_->N);
    pimpl_->host_hess = Eigen::MatrixXf(pimpl_->N, pimpl_->N);
}

void InertialEnergy::update_x(const Eigen::VectorXf &x)
{
    pimpl_->host_x = x;
    pimpl_->device_x.copy_from(std::vector<float>(x.data(), x.data() + x.size()));
}

void InertialEnergy::update_x_tilde(const Eigen::VectorXf &x_tilde)
{
    pimpl_->host_x_tilde = x_tilde;
    pimpl_->device_x_tilde.copy_from(std::vector<float>(x_tilde.data(), x_tilde.data() + x_tilde.size()));
}
void InertialEnergy::update_m(float m)
{
    pimpl_->m = m;
}
float InertialEnergy::val()
{
    auto &device_x = pimpl_->device_x;
    auto &device_x_tilde = pimpl_->device_x_tilde;
    auto &m = pimpl_->m;
    auto &N = pimpl_->N;
    auto &val = pimpl_->val;
    ParallelFor(256)
        .apply(N,
               [device_x = device_x.cviewer(), device_x_tilde = device_x_tilde.cviewer(), m, N, val] __device__(int i) mutable
               {
                   val += 0.5 * m * (device_x(i) - device_x_tilde(i)) * (device_x(i) - device_x_tilde(i));
               })
        .wait();
    return val;
}
Eigen::VectorXf &InertialEnergy::grad()
{
    auto &device_x = pimpl_->device_x;
    auto &device_x_tilde = pimpl_->device_x_tilde;
    auto &m = pimpl_->m;
    auto &N = pimpl_->N;
    auto &device_grad = pimpl_->device_grad;
    auto &host_grad = pimpl_->host_grad;
    ParallelFor(256)
        .apply(N,
               [device_x = device_x.cviewer(), device_x_tilde = device_x_tilde.cviewer(), m, N, device_grad = device_grad.viewer()] __device__(int i) mutable
               {
                   device_grad(i) = m * (device_x(i) - device_x_tilde(i));
               })
        .wait();
    host_grad = Eigen::VectorXf::Map(device_grad.data(), device_grad.size());
    return host_grad;
} // Calculate the gradient of the energy
Eigen::MatrixXf &InertialEnergy::hess()
{
    auto &device_x = pimpl_->device_x;
    auto &device_x_tilde = pimpl_->device_x_tilde;
    auto &m = pimpl_->m;
    auto &N = pimpl_->N;
    auto &device_hess = pimpl_->device_hess;
    auto &host_hess = pimpl_->host_hess;
    ParallelFor(256)
        .apply(N,
               [device_x = device_x.cviewer(), device_x_tilde = device_x_tilde.cviewer(), m, N, device_hess = device_hess.viewer()] __device__(int i) mutable
               {
                   device_hess(i * N + i) = m;
               })
        .wait();
    host_hess = Eigen::MatrixXf::Map(device_hess.data(), N, N);
    return host_hess;
} // Calculate the Hessian matrix of the energy
