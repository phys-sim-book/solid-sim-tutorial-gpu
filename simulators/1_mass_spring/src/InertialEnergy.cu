#include "InertialEnergy.h"
#include "uti.h"
#include <muda/muda.h>
#include <muda/container.h>

using namespace muda;

struct InertialEnergy::Impl
{
	DeviceBuffer<float> device_x, device_x_tilde, device_grad;
	int N;
	float m, val;
	std::vector<float> host_x, host_x_tilde, host_grad;
	std::vector<float> host_hess;
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

InertialEnergy::InertialEnergy(const std::vector<float> &x, const std::vector<float> &x_tilde, float m) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->host_x = x;
	pimpl_->host_x_tilde = x_tilde;
	pimpl_->m = m;
	pimpl_->N = x.size() / 2;
	pimpl_->device_x.copy_from(x);
	pimpl_->device_x_tilde.copy_from(x_tilde);
	pimpl_->device_grad = std::vector<float>(pimpl_->N * 2);
	pimpl_->host_grad = std::vector<float>(pimpl_->N * 2);
	pimpl_->host_hess = std::vector<float>(pimpl_->N * pimpl_->N * 4, m);
}

void InertialEnergy::update_x(const std::vector<float> &x)
{
	pimpl_->host_x = x;
	pimpl_->device_x.copy_from(x);
}

void InertialEnergy::update_x_tilde(const std::vector<float> &x_tilde)
{
	pimpl_->host_x_tilde = x_tilde;
	pimpl_->device_x_tilde.copy_from(x_tilde);
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
	auto &N = pimpl_->N * 2;
	DeviceBuffer<float> device_val(N);
	ParallelFor(256)
		.apply(N,
			   [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_x_tilde = device_x_tilde.cviewer(), m] __device__(int i) mutable
			   {
				   device_val(i) = 0.5 * m * (device_x(i) - device_x_tilde(i)) * (device_x(i) - device_x_tilde(i));
			   })
		.wait();
	return devicesum(device_val);
}
std::vector<float> &InertialEnergy::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_x_tilde = pimpl_->device_x_tilde;
	auto &m = pimpl_->m;
	auto &N = pimpl_->N * 2;
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

std::vector<float> &InertialEnergy::hess()
{
	return pimpl_->host_hess;
} // Calculate the Hessian matrix of the energy
