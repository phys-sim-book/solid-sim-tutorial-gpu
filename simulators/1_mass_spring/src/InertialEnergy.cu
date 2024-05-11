#include "InertialEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "device_uti.h"
using namespace muda;

template <typename T, int dim>
struct InertialEnergy<T, dim>::Impl
{
	DeviceBuffer<T> device_x, device_x_tilde, device_grad;
	int N;
	T m, val;
	std::vector<T> host_grad;
	SparseMatrix<T> host_hess;
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
InertialEnergy<T, dim> &InertialEnergy<T, dim>::operator=(const InertialEnergy<T, dim> &rhs)
{
	*pimpl_ = *rhs.pimpl_;
	return *this;
};

template <typename T, int dim>
InertialEnergy<T, dim>::InertialEnergy(int N, T m) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->N = N;
	pimpl_->m = m;
	pimpl_->device_grad = std::vector<T>(pimpl_->N * dim);
	pimpl_->host_grad = std::vector<T>(pimpl_->N * dim);
	pimpl_->host_hess = SparseMatrix<T>(pimpl_->N * dim);
	pimpl_->host_hess.set_diagonal(m);
}

template <typename T, int dim>
void InertialEnergy<T, dim>::update_x(const std::vector<T> &x)
{
	pimpl_->device_x.copy_from(x);
}

template <typename T, int dim>
void InertialEnergy<T, dim>::update_x_tilde(const std::vector<T> &x_tilde)
{
	pimpl_->device_x_tilde.copy_from(x_tilde);
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
std::vector<T> &InertialEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_x_tilde = pimpl_->device_x_tilde;
	auto &m = pimpl_->m;
	auto N = pimpl_->N * dim;
	auto &device_grad = pimpl_->device_grad;
	auto &host_grad = pimpl_->host_grad;
	ParallelFor(256)
		.apply(N,
			   [device_x = device_x.cviewer(), device_x_tilde = device_x_tilde.cviewer(), m, N, device_grad = device_grad.viewer()] __device__(int i) mutable
			   {
				   device_grad(i) = m * (device_x(i) - device_x_tilde(i));
			   })
		.wait();
	device_grad.copy_to(host_grad);
	return host_grad;
} // Calculate the gradient of the energy

template <typename T, int dim>
SparseMatrix<T> &InertialEnergy<T, dim>::hess()
{
	return pimpl_->host_hess;
} // Calculate the Hessian matrix of the energy

template class InertialEnergy<float, 2>;
template class InertialEnergy<float, 3>;
template class InertialEnergy<double, 2>;
template class InertialEnergy<double, 3>;
