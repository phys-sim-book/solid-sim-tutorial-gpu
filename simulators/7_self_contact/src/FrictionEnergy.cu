#include "FrictionEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <stdio.h>
#include "device_uti.h"
#define epsv 1e-3

using namespace muda;

template <typename T, int dim>
struct FrictionEnergy<T, dim>::Impl
{
    DeviceBuffer<T> device_v;
    DeviceBuffer<T> device_mu_lambda;
    DeviceBuffer<T> device_grad;
    DeviceTripletMatrix<T, 1> device_hess;
    T hhat;
    Eigen::Matrix<T, dim, 1> n;
    int N;
};

template <typename T, int dim>
FrictionEnergy<T, dim>::FrictionEnergy() = default;

template <typename T, int dim>
FrictionEnergy<T, dim>::~FrictionEnergy() = default;

template <typename T, int dim>
FrictionEnergy<T, dim>::FrictionEnergy(FrictionEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
FrictionEnergy<T, dim> &FrictionEnergy<T, dim>::operator=(FrictionEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
FrictionEnergy<T, dim>::FrictionEnergy(const FrictionEnergy<T, dim> &rhs)
    : pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
FrictionEnergy<T, dim>::FrictionEnergy(const std::vector<T> &v, T hhat, const std::vector<T> &n)
    : pimpl_{std::make_unique<Impl>()}
{
    pimpl_->N = v.size() / dim;
    pimpl_->device_v.copy_from(v);
    pimpl_->device_mu_lambda.resize(pimpl_->N);
    pimpl_->hhat = hhat;
    pimpl_->n = Eigen::Map<const Eigen::Matrix<T, dim, 1>>(n.data());
    pimpl_->device_grad.resize(pimpl_->N * dim);
    pimpl_->device_hess.resize_triplets(pimpl_->N * dim * dim);
    pimpl_->device_hess.reshape(v.size(), v.size());
}

template <typename T, int dim>
void FrictionEnergy<T, dim>::update_v(const DeviceBuffer<T> &v)
{
    pimpl_->device_v.view().copy_from(v);
}
template <typename T, int dim>
DeviceBuffer<T> &FrictionEnergy<T, dim>::get_mu_lambda()
{
    return pimpl_->device_mu_lambda;
}

template <typename T, int dim>
T __device__ FrictionEnergy<T, dim>::f0(T vbarnorm, T Epsv, T hhat)
{
    if (vbarnorm >= Epsv)
    {
        return vbarnorm * hhat;
    }
    else
    {
        T vbarnormhhat = vbarnorm * hhat;
        T epsvhhat = Epsv * hhat;
        return vbarnormhhat * vbarnormhhat * (-vbarnormhhat / 3.0 + epsvhhat) / (epsvhhat * epsvhhat) + epsvhhat / 3.0;
    }
}

template <typename T, int dim>
T __device__ FrictionEnergy<T, dim>::f1_div_vbarnorm(T vbarnorm, T Epsv)
{
    if (vbarnorm >= Epsv)
    {
        return 1.0 / vbarnorm;
    }
    else
    {
        return (-vbarnorm + 2.0 * Epsv) / (Epsv * Epsv);
    }
}

template <typename T, int dim>
T __device__ FrictionEnergy<T, dim>::f_hess_term(T vbarnorm, T Epsv)
{
    if (vbarnorm >= Epsv)
    {
        return -1.0 / (vbarnorm * vbarnorm);
    }
    else
    {
        return -1.0 / (Epsv * Epsv);
    }
}

template <typename T, int dim>
T FrictionEnergy<T, dim>::val()
{
    auto &device_v = pimpl_->device_v;
    auto &device_mu_lambda = pimpl_->device_mu_lambda;
    auto &hhat = pimpl_->hhat;
    auto &n = pimpl_->n;
    int N = device_v.size() / dim;
    DeviceBuffer<T> device_val(N);

    ParallelFor(256).apply(N, [device_val = device_val.viewer(), device_v = device_v.cviewer(), device_mu_lambda = device_mu_lambda.cviewer(), hhat, n, this] __device__(int i) mutable
                           {
        Eigen::Matrix<T, dim, dim> T_mat = Eigen::Matrix<T, dim, dim>::Identity() - n * n.transpose();
        if (device_mu_lambda(i) > 0)
        {
            Eigen::Matrix<T, dim, 1> v;
            for (int j = 0; j < dim; ++j)
            {
                v(j) = device_v(i * dim + j);
            }
            Eigen::Matrix<T, dim, 1> vbar = T_mat * v;
            T vbarnorm = vbar.norm();
            T val = f0(vbarnorm, epsv, hhat);
            device_val(i) = device_mu_lambda(i) * val;
        } })
        .wait();

    return devicesum(device_val);
}

template <typename T, int dim>
const DeviceBuffer<T> &FrictionEnergy<T, dim>::grad()
{
    auto &device_v = pimpl_->device_v;
    auto &device_mu_lambda = pimpl_->device_mu_lambda;
    auto &hhat = pimpl_->hhat;
    auto &n = pimpl_->n;
    int N = device_v.size() / dim;
    auto &device_grad = pimpl_->device_grad;
    device_grad.fill(0);

    ParallelFor(256).apply(N, [device_v = device_v.cviewer(), device_mu_lambda = device_mu_lambda.cviewer(), device_grad = device_grad.viewer(), hhat, n, this] __device__(int i) mutable
                           {
        Eigen::Matrix<T, dim, dim> T_mat = Eigen::Matrix<T, dim, dim>::Identity() - n * n.transpose();
        if (device_mu_lambda(i) > 0)
        {
            Eigen::Matrix<T, dim, 1> v;
            for (int j = 0; j < dim; ++j)
            {
                v(j) = device_v(i * dim + j);
            }
            Eigen::Matrix<T, dim, 1> vbar = T_mat * v;
            T vbarnorm = vbar.norm();
            T grad_factor = f1_div_vbarnorm(vbarnorm, epsv);
            Eigen::Matrix<T, dim, 1> grad = grad_factor * T_mat * vbar;

            for (int j = 0; j < dim; ++j)
            {
                device_grad(i * dim + j) = device_mu_lambda(i) * grad(j);
            }
        } })
        .wait();

    return device_grad;
}
template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &FrictionEnergy<T, dim>::hess()
{
    auto &device_v = pimpl_->device_v;
    auto &device_mu_lambda = pimpl_->device_mu_lambda;
    auto &hhat = pimpl_->hhat;
    auto &n = pimpl_->n;
    auto &device_hess = pimpl_->device_hess;
    auto device_hess_row_idx = device_hess.row_indices();
    auto device_hess_col_idx = device_hess.col_indices();
    auto device_hess_val = device_hess.values();
    int N = device_v.size() / dim;
    ParallelFor(256).apply(N, [device_v = device_v.cviewer(), device_mu_lambda = device_mu_lambda.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), hhat, n, N, this] __device__(int i) mutable
                           {
        Eigen::Matrix<T, dim, dim> T_mat = Eigen::Matrix<T, dim, dim>::Identity() - n * n.transpose();
        for (int j = 0; j < dim; ++j)
        {
            for (int k = 0; k < dim; ++k)
            {
                int idx = i * dim * dim + j * dim + k;
                device_hess_row_idx(idx) = i * dim + j;
                device_hess_col_idx(idx) = i * dim + k;
            }
        }
        if (device_mu_lambda(i) > 0)
        {
            Eigen::Matrix<T, dim, 1> v;
            for (int j = 0; j < dim; ++j)
            {
                v(j) = device_v(i * dim + j);
            }
            Eigen::Matrix<T, dim, 1> vbar = T_mat * v;
            T vbarnorm = vbar.norm();
            Eigen::Matrix<T, dim, dim> inner_term = Eigen::Matrix<T, dim, dim>::Identity() * f1_div_vbarnorm(vbarnorm, epsv);
            if (vbarnorm != 0)
            {
                inner_term += f_hess_term(vbarnorm, epsv) / vbarnorm * vbar * vbar.transpose();
            }
            Eigen::Matrix<T, dim, dim> local_hess;
            make_PSD(inner_term, local_hess);
            local_hess = device_mu_lambda(i) * T_mat * local_hess * T_mat.transpose() / hhat;
            for (int j = 0; j < dim; ++j)
            {
                for (int k = 0; k < dim; ++k)
                {
                    int idx = i * dim * dim + j * dim + k;
                    device_hess_val(idx) = local_hess(j, k);
                }
            }
        } })
        .wait();
    return device_hess;
}
template class FrictionEnergy<float, 2>;
template class FrictionEnergy<float, 3>;
template class FrictionEnergy<double, 2>;
template class FrictionEnergy<double, 3>;
