#include "MassSpringEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <stdio.h>
#include "uti.h"
using namespace muda;

template <typename T, int dim>
struct MassSpringEnergy<T, dim>::Impl
{
	DeviceBuffer<T> device_x;
	DeviceBuffer<T> device_l2, device_k;
	DeviceBuffer<int> device_e;
	int N;
	std::vector<T> host_x;
	std::vector<T> host_l2, host_k, host_grad;
	std::vector<int> host_e;
	std::vector<T> host_hess;
};

template <typename T, int dim>
MassSpringEnergy<T, dim>::~MassSpringEnergy<T, dim>() = default;

template <typename T, int dim>
MassSpringEnergy<T, dim>::MassSpringEnergy<T, dim>(MassSpringEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringEnergy<T, dim> &MassSpringEnergy<T, dim>::operator=(MassSpringEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringEnergy<T, dim>::MassSpringEnergy<T, dim>(const MassSpringEnergy<T, dim> &rhs)
	: pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
MassSpringEnergy<T, dim> &MassSpringEnergy<T, dim>::operator=(const MassSpringEnergy<T, dim> &rhs)
{
	*pimpl_ = *rhs.pimpl_;
	return *this;
}

template <typename T, int dim>
MassSpringEnergy<T, dim>::MassSpringEnergy<T, dim>(const std::vector<T> &x, const std::vector<int> &e, const std::vector<T> &l2, const std::vector<T> &k) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->host_x = x;
	pimpl_->host_e = e;
	pimpl_->host_l2 = l2;
	pimpl_->host_k = k;
	pimpl_->N = x.size() / dim;
	pimpl_->device_x.copy_from(x);
	pimpl_->device_e.copy_from(e);
	pimpl_->device_l2.copy_from(l2);
	pimpl_->device_k.copy_from(k);
	pimpl_->host_grad = std::vector<T>(pimpl_->N * dim);
	pimpl_->host_hess = std::vector<T>(pimpl_->N * pimpl_->N * dim * dim);
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_x(const std::vector<T> &x)
{
	pimpl_->host_x = x;
	pimpl_->device_x.copy_from(x);
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_e(const std::vector<int> &e)
{
	pimpl_->host_e = e;
	pimpl_->device_e.copy_from(e);
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_l2(const std::vector<T> &l2)
{
	pimpl_->host_l2 = l2;
	pimpl_->device_l2.copy_from(l2);
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_k(const std::vector<T> &k)
{
	pimpl_->host_k = k;
	pimpl_->device_k.copy_from(k);
}

template <typename T, int dim>
T MassSpringEnergy<T, dim>::val()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	int N = device_e.size() / dim;
	DeviceBuffer<T> device_val(N);
	ParallelFor(256).apply(N, [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer()] __device__(int i) mutable
						   {
							   int idx1= device_e(2 * i); // First node index
								int idx2 = device_e(2 * i + 1); // Second node index
								T diff = 0;
								for (int d = 0; d < dim;d++){
									T diffi=device_x(2 * idx1 + d) - device_x(2 * idx2 + d);
									diff += diffi * diffi;
						   		}
								device_val(i) = 0.5 * device_l2(i) * device_k(i) * (diff / device_l2(i) - 1) * (diff / device_l2(i) - 1); })
		.wait();

	return devicesum(device_val);
} // Calculate the energy

template <typename T, int dim>
std::vector<T> &MassSpringEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto N = pimpl_->device_e.size() / 2;
	DeviceBuffer<T> device_grad(pimpl_->N * dim);
	auto &host_grad = pimpl_->host_grad;
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_grad = device_grad.viewer()] __device__(int i) mutable
						   {
							int idx1= device_e(2 * i); // First node index
							int idx2 = device_e(2 * i + 1); // Second node index
							T diff = 0;
							T diffi[dim];
							for (int d = 0; d < dim;d++){
								diffi[d]=device_x(2 * idx1 + d) - device_x(2 * idx2 + d);
								diff += diffi[d] * diffi[d];
						   }
						   T factor=2*device_k(i)*(diff - device_l2(i)); 
						   for(int d=0;d<dim;d++){
							   atomicAdd(&device_grad(dim * idx1 + d), factor * diffi[d]);
							   atomicAdd(&device_grad(dim * idx2 + d), -factor * diffi[d]);
						   } })
		.wait();
	device_grad.copy_to(host_grad);
	return host_grad;
}

template <typename T, int dim>
std::vector<T> &MassSpringEnergy<T, dim>::hess()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto N = pimpl_->N;
	auto &host_hess = pimpl_->host_hess;
	DeviceBuffer<T> device_hess(N * N * dim * dim);
	device_hess.fill(0.0f);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_hess = device_hess.viewer(), N] __device__(int i) mutable
						   {
							   int idx[2] = {device_e(2 * i), device_e(2 * i + 1)}; // First node index
							   T diff = 0;
							   T diffi[dim];
							   for (int d = 0; d < dim; d++)
							   {
								   diffi[d] = device_x(2 * idx[0] + d) - device_x(2 * idx[1] + d);
								   diff += diffi[d] * diffi[d];
							   }
							   Eigen::Vector<T, dim> diff_vec(diffi);
							   Eigen::Matrix<T, dim, dim> diff_outer = diff_vec * diff_vec.transpose();
							   T scalar = 2 * device_k(i) / device_l2(i);
							   Eigen::Matrix<T, dim, dim> H_diff = scalar * (2 * diff_outer + (diff_vec.dot(diff_vec) - device_l2(i)) * Eigen::Matrix<T, dim, dim>::Identity());
							   Eigen::Matrix<T, dim * 2, dim * 2> H_block, H_local;
							   H_block << H_diff, -H_diff,
								   -H_diff, H_diff;
							   
							   make_PSD(H_block, H_local);
							   // add to global matrix
							   for (int ni = 0; ni < 2; ni++)
								   for (int nj = 0; nj < 2; nj++)
									   for (int d1 = 0; d1 < dim; d1++)
										   for (int d2 = 0; d2 < dim; d2++)
											   atomicAdd(&device_hess((idx[ni] * dim + d1) * N * dim + idx[nj] * dim + d2), H_local(ni * dim + d1, nj * dim + d2)); })
		.wait();
	device_hess.copy_to(host_hess);
	return host_hess;

} // Calculate the Hessian of the energy

template class MassSpringEnergy<float, 2>;
template class MassSpringEnergy<float, 3>;
template class MassSpringEnergy<double, 2>;
template class MassSpringEnergy<double, 3>;