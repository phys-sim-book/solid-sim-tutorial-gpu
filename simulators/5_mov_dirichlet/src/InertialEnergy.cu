#include "InertialEnergy.h"
#include "uti.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "device_uti.h"

using namespace muda;

template <typename T, int dim>
struct InertialEnergy<T, dim>::Impl
{
	DeviceBuffer<T> device_x, device_x_tilde, device_grad;
	DeviceTripletMatrix<T, 1> device_hess;
	int N;
	T m, val;
	Impl(int N, T m);
};
template <typename T, int dim>
InertialEnergy<T, dim>::InertialEnergy() = default;

template <typename T, int dim>
InertialEnergy<T, dim>::~InertialEnergy() = default;

template <typename T, int dim>
InertialEnergy<T, dim>::InertialEnergy(InertialEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
InertialEnergy<T, dim> &InertialEnergy<T, dim>::operator=(InertialEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
InertialEnergy<T, dim>::InertialEnergy(const InertialEnergy<T, dim> &rhs)
	: pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
InertialEnergy<T, dim>::InertialEnergy(int N, T m) : pimpl_{std::make_unique<Impl>(N, m)}
{
	generate_hess();
}

template <typename T, int dim>
InertialEnergy<T, dim>::Impl::Impl(int N_, T m_) : N(N_), m(m_)
{
	device_x.resize(N * dim);
	device_x_tilde.resize(N * dim);
	device_hess.resize_triplets(N * dim);
	device_hess.reshape(N * dim, N * dim);
	device_grad.resize(N * dim);
}
template <typename T, int dim>
void InertialEnergy<T, dim>::generate_hess()
{
	auto &device_hess = pimpl_->device_hess;
	auto m = pimpl_->m;
	auto N = pimpl_->N;
	ParallelFor(256)
		.apply(N * dim,
			   [device_hess_row_indices = device_hess.row_indices().viewer(), device_hess_col_indices = device_hess.col_indices().viewer(),
				device_hess_values = device_hess.values().viewer(), m] __device__(int i) mutable
			   {
				   device_hess_row_indices(i) = i;
				   device_hess_col_indices(i) = i;
				   device_hess_values(i) = m;
			   })
		.wait();
}

template <typename T, int dim>
void InertialEnergy<T, dim>::update_x(const DeviceBuffer<T> &x)
{
	pimpl_->device_x.view().copy_from(x);
}

template <typename T, int dim>
void InertialEnergy<T, dim>::update_x_tilde(const DeviceBuffer<T> &x_tilde)
{
	pimpl_->device_x_tilde.view().copy_from(x_tilde);
}

template <typename T, int dim>
void InertialEnergy<T, dim>::update_m(T m)
{
	pimpl_->m = m;
}

template <typename T, int dim>
T InertialEnergy<T, dim>::val()
{
	auto &device_x = pimpl_->device_x;
	auto &device_x_tilde = pimpl_->device_x_tilde;
	auto &m = pimpl_->m;
	auto N = pimpl_->N * dim;
	DeviceBuffer<T> device_val(N);
	ParallelFor(256)
		.apply(N,
			   [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_x_tilde = device_x_tilde.cviewer(), m] __device__(int i) mutable
			   {
				   device_val(i) = 0.5 * m * (device_x(i) - device_x_tilde(i)) * (device_x(i) - device_x_tilde(i));
			   })
		.wait();
	return devicesum(device_val);
}

template <typename T, int dim>
const DeviceBuffer<T> &InertialEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_x_tilde = pimpl_->device_x_tilde;
	auto m = pimpl_->m;
	auto N = pimpl_->N * dim;
	auto &device_grad = pimpl_->device_grad;
	ParallelFor(256)
		.apply(N,
			   [device_x = device_x.cviewer(), device_x_tilde = device_x_tilde.cviewer(), m, N, device_grad = device_grad.viewer()] __device__(int i) mutable
			   {
				   device_grad(i) = m * (device_x(i) - device_x_tilde(i));
			   })
		.wait();
	// display_vec(device_grad);
	return device_grad;
} // Calculate the gradient of the energy

template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &InertialEnergy<T, dim>::hess()
{
	return pimpl_->device_hess;
} // Calculate the Hessian matrix of the energy

template class InertialEnergy<float, 2>;
template class InertialEnergy<float, 3>;
template class InertialEnergy<double, 2>;
template class InertialEnergy<double, 3>;
