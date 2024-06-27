#include "SpringEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <stdio.h>
#include "device_uti.h"

using namespace muda;

template <typename T, int dim>
struct SpringEnergy<T, dim>::Impl
{
    DeviceBuffer<T> device_x;
    DeviceBuffer<T> device_m;
    DeviceBuffer<int> device_DBC;
    DeviceBuffer<T> device_DBC_target,device_DBC_v,device_DBC_limit;
    DeviceBuffer<T> device_grad;
    DeviceTripletMatrix<T, 1> device_hess;
    T k,h;
    int N;
};

template <typename T, int dim>
SpringEnergy<T, dim>::SpringEnergy() = default;

template <typename T, int dim>
SpringEnergy<T, dim>::~SpringEnergy() = default;

template <typename T, int dim>
SpringEnergy<T, dim>::SpringEnergy(SpringEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
SpringEnergy<T, dim> &SpringEnergy<T, dim>::operator=(SpringEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
SpringEnergy<T, dim>::SpringEnergy(const SpringEnergy<T, dim> &rhs)
    : pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
SpringEnergy<T, dim>::SpringEnergy(const std::vector<T> &x, const std::vector<T> &m, const std::vector<int> &DBC, const std::vector<T> &DBC_v, const std::vector<T> &DBC_limit,T k,T h)
    : pimpl_{std::make_unique<Impl>()}
{
    pimpl_->N = x.size() / dim;
    pimpl_->device_x.copy_from(x);
    pimpl_->device_m.copy_from(m);
    pimpl_->device_DBC.copy_from(DBC);
    pimpl_->device_DBC_v.copy_from(DBC_v);
    pimpl_->device_DBC_limit.copy_from(DBC_limit);
    pimpl_->device_DBC_target.resize(DBC.size() * dim);
    pimpl_->k = k;
    pimpl_->h = h;
    pimpl_->device_grad.resize(pimpl_->N * dim);
    pimpl_->device_hess.resize_triplets(pimpl_->N * dim * dim);
    pimpl_->device_hess.reshape(x.size(), x.size());
}

template <typename T, int dim>
void SpringEnergy<T, dim>::update_x(const DeviceBuffer<T> &x)
{
    pimpl_->device_x.view().copy_from(x);
}

template <typename T, int dim>
void SpringEnergy<T, dim>::update_DBC_target()
{
        auto &device_x = pimpl_->device_x;
        auto &device_DBC = pimpl_->device_DBC;
        auto &device_DBC_target = pimpl_->device_DBC_target;
        auto h = pimpl_->h;
        auto &device_DBC_v = pimpl_->device_DBC_v;
        auto &device_DBC_limit = pimpl_->device_DBC_limit;
        int N = device_DBC.size();
        device_DBC_target.fill(0);

        ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_DBC = device_DBC.cviewer(), device_DBC_target = device_DBC_target.viewer(), device_DBC_v = device_DBC_v.cviewer(), h, device_DBC_limit = device_DBC_limit.cviewer()] __device__(int i) mutable
                               {
            int idx = device_DBC(i);
            T d=0;
            for (int j = 0; j < dim; ++j)
            {
                d += (device_DBC_limit(i*dim + j) - device_x(idx * dim + j)) * (device_DBC_v(i*dim + j));
            }
            if(d>0)
            {
                for (int j = 0; j < dim; ++j)
                {
                    device_DBC_target(i*dim + j) = device_x(idx * dim + j) + h * device_DBC_v(i*dim + j);
                }
            }
            else
            {
                for (int j = 0; j < dim; ++j)
                {
                    device_DBC_target(i*dim + j) = device_x(idx*dim + j);
                }
            }
        }).wait();


}

template <typename T, int dim>
T SpringEnergy<T, dim>::val()
{
    auto &device_x = pimpl_->device_x;
    auto &device_m = pimpl_->device_m;
    auto &device_DBC = pimpl_->device_DBC;
    auto &device_DBC_target = pimpl_->device_DBC_target;
    T k = pimpl_->k;
    int N = device_DBC.size();
    DeviceBuffer<T> device_val(N);
    device_val.fill(0);

    ParallelFor(256).apply(N, [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_m = device_m.cviewer(), device_DBC = device_DBC.cviewer(), device_DBC_target = device_DBC_target.cviewer(), k] __device__(int i) mutable
                           {

            int idx = device_DBC(i);
            Eigen::Matrix<T, dim, 1> diff;
            for (int j = 0; j < dim; ++j)
            {
                diff(j) = device_x(idx * dim + j) - device_DBC_target(i*dim + j);
            }
            device_val(i) = 0.5 * k * device_m(idx) * diff.dot(diff); })
        .wait();

    return devicesum(device_val);
}

template <typename T, int dim>
const DeviceBuffer<T> &SpringEnergy<T, dim>::grad()
{
    auto &device_x = pimpl_->device_x;
    auto &device_m = pimpl_->device_m;
    auto &device_DBC = pimpl_->device_DBC;
    auto &device_DBC_target = pimpl_->device_DBC_target;
    T k = pimpl_->k;
    int N = device_DBC.size();
    auto &device_grad = pimpl_->device_grad;
    device_grad.fill(0);

    ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_m = device_m.cviewer(), device_DBC = device_DBC.cviewer(), device_DBC_target = device_DBC_target.cviewer(), device_grad = device_grad.viewer(), k] __device__(int i) mutable
                           {

            int idx = device_DBC(i);
            Eigen::Matrix<T, dim, 1> grad;
            for (int j = 0; j < dim; ++j)
            {
                grad(j) = device_x(idx * dim + j) - device_DBC_target(i*dim + j);
            }
            grad *= k * device_m(idx);
            for (int j = 0; j < dim; ++j)
            {
                device_grad(idx * dim + j) = grad(j);
            } })
        .wait();

    return device_grad;
}

template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &SpringEnergy<T, dim>::hess()
{
    auto &device_x = pimpl_->device_x;
    auto &device_m = pimpl_->device_m;
    auto &device_DBC = pimpl_->device_DBC;
    auto &device_DBC_target = pimpl_->device_DBC_target;
    T k = pimpl_->k;
    auto &device_hess = pimpl_->device_hess;
    auto device_hess_row_idx = device_hess.row_indices();
    auto device_hess_col_idx = device_hess.col_indices();
    auto device_hess_val = device_hess.values();
    int N = device_DBC.size();

    ParallelFor(256).apply(N, [device_DBC = device_DBC.cviewer(), device_m = device_m.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), k, N] __device__(int i) mutable
                           {

            int idx = device_DBC(i);
            for (int d = 0; d < dim; ++d)
            {
                int row_idx = idx * dim + d;
                device_hess_row_idx(i * dim + d) = row_idx;
                device_hess_col_idx(i * dim + d) = row_idx;
                device_hess_val(i * dim + d) = k * device_m(idx);
            } })
        .wait();

    return device_hess;
}

template class SpringEnergy<float, 2>;
template class SpringEnergy<float, 3>;
template class SpringEnergy<double, 2>;
template class SpringEnergy<double, 3>;
