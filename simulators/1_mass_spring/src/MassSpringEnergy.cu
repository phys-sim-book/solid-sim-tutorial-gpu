#include "MassSpringEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "uti.h"
using namespace muda;

struct MassSpringEnergy::Impl
{
	DeviceBuffer<float> device_x;
	DeviceBuffer<float> device_l2, device_k;
	DeviceBuffer<int> device_e;
	int N;
	std::vector<float> host_x;
	std::vector<float> host_l2, host_k, host_grad;
	std::vector<int> host_e;
	std::vector<float> host_hess;
};

MassSpringEnergy::~MassSpringEnergy() = default;

MassSpringEnergy::MassSpringEnergy(MassSpringEnergy &&rhs) = default;

MassSpringEnergy &MassSpringEnergy::operator=(MassSpringEnergy &&rhs) = default;

MassSpringEnergy::MassSpringEnergy(const MassSpringEnergy &rhs)
	: pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

MassSpringEnergy &MassSpringEnergy::operator=(const MassSpringEnergy &rhs)
{
	*pimpl_ = *rhs.pimpl_;
	return *this;
}

MassSpringEnergy::MassSpringEnergy(const std::vector<float> &x, const std::vector<int> &e, const std::vector<float> &l2, const std::vector<float> &k) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->host_x = x;
	pimpl_->host_e = e;
	pimpl_->host_l2 = l2;
	pimpl_->host_k = k;
	pimpl_->N = x.size() / 2;
	pimpl_->device_x.copy_from(x);
	pimpl_->device_e.copy_from(e);
	pimpl_->device_l2.copy_from(l2);
	pimpl_->device_k.copy_from(k);
	pimpl_->host_grad = std::vector<float>(pimpl_->N * 2);
	pimpl_->host_hess = std::vector<float>(pimpl_->N * pimpl_->N * 4);
}

void MassSpringEnergy::update_x(const std::vector<float> &x)
{
	pimpl_->host_x = x;
	pimpl_->device_x.copy_from(x);
}

void MassSpringEnergy::update_e(const std::vector<int> &e)
{
	pimpl_->host_e = e;
	pimpl_->device_e.copy_from(e);
}
void MassSpringEnergy::update_l2(const std::vector<float> &l2)
{
	pimpl_->host_l2 = l2;
	pimpl_->device_l2.copy_from(l2);
}
void MassSpringEnergy::update_k(const std::vector<float> &k)
{
	pimpl_->host_k = k;
	pimpl_->device_k.copy_from(k);
}

float MassSpringEnergy::val()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	int N = device_e.size() / 2;
	DeviceBuffer<float> device_val(N);
	ParallelFor(256).apply(N, [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer()] __device__(int i) mutable
						   {
		int idx1 = device_e(2 * i); // First node index
        int idx2 = device_e(2 * i + 1); // Second node index
        float diff_x = device_x(2 * idx1) - device_x(2 * idx2);
        float diff_y = device_x(2 * idx1 + 1) - device_x(2 * idx2 + 1);
        float diff = sqrtf(diff_x * diff_x + diff_y * diff_y); // Euclidean distance between points
        float spring_energy = 0.5 * device_k(i) * (diff * diff / device_l2(i) - 1) * (diff * diff / device_l2(i) - 1);
        device_val(i) = spring_energy; })
		.wait();

	return devicesum(device_val);
} // Calculate the energy

std::vector<float> &MassSpringEnergy::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto &N = pimpl_->N;
	DeviceBuffer<float> device_grad(N);
	auto &host_grad = pimpl_->host_grad;
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_grad = device_grad.viewer()] __device__(int i) mutable
						   {
		int idx1 = device_e(i * 2) * 2; // Index for the first node (x coordinate)
        int idx2 = device_e(i * 2 + 1) * 2; // Index for the second node (x coordinate)

        // Calculate differences in x and y coordinates
        float diff_x = device_x(idx1) - device_x(idx2);
        float diff_y = device_x(idx1 + 1) - device_x(idx2 + 1);

        // Compute Euclidean distance and its derivatives
        float diff2 = diff_x * diff_x + diff_y * diff_y;
        float force_magnitude = 2 * device_k(i) * ((diff2 / device_l2(i)) - 1);

        // Compute gradient components for x and y directions
        float grad_x = force_magnitude* diff_x ;
        float grad_y = force_magnitude * diff_y;

        // Update gradients using atomicAdd for potential concurrent access
        atomicAdd(&device_grad(idx1), grad_x);
        atomicAdd(&device_grad(idx1 + 1), grad_y);
        atomicAdd(&device_grad(idx2), -grad_x);
        atomicAdd(&device_grad(idx2 + 1), -grad_y); })
		.wait();
	device_grad.copy_to(host_grad);
	return host_grad;
}

std::vector<float> &MassSpringEnergy::hess()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto &N = pimpl_->N;
	auto &host_hess = pimpl_->host_hess;
	DeviceBuffer<float> device_hess(N * N);
	device_hess.fill(0.0f);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_hess = device_hess.viewer(), N] __device__(int i) mutable {}).wait();
	device_hess.copy_to(host_hess);
	return host_hess;

} // Calculate the Hessian of the energy