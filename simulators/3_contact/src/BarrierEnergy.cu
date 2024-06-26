#include "BarrierEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <stdio.h>
#include "device_uti.h"
#define dhat 0.01
#define kappa 1e5
using namespace muda;

template <typename T, int dim>
struct BarrierEnergy<T, dim>::Impl
{
	DeviceBuffer<T> device_x;
	DeviceBuffer<T> device_contact_area;
	int N;
	T y_ground;
	DeviceBuffer<T> device_grad;
	DeviceTripletMatrix<T, 1> device_hess;
};
template <typename T, int dim>
BarrierEnergy<T, dim>::BarrierEnergy() = default;

template <typename T, int dim>
BarrierEnergy<T, dim>::~BarrierEnergy() = default;

template <typename T, int dim>
BarrierEnergy<T, dim>::BarrierEnergy(BarrierEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
BarrierEnergy<T, dim> &BarrierEnergy<T, dim>::operator=(BarrierEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
BarrierEnergy<T, dim>::BarrierEnergy(const BarrierEnergy<T, dim> &rhs)
	: pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
BarrierEnergy<T, dim>::BarrierEnergy(const std::vector<T> &x, const std::vector<T> &contact_area, T y_ground) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->N = x.size() / dim;
	pimpl_->y_ground = y_ground;
	pimpl_->device_x.copy_from(x);
	pimpl_->device_contact_area.copy_from(contact_area);
	pimpl_->device_hess.resize_triplets(pimpl_->N);
	pimpl_->device_hess.reshape(x.size(), x.size());
	pimpl_->device_grad.resize(pimpl_->N * dim);
}

template <typename T, int dim>
void BarrierEnergy<T, dim>::update_x(const DeviceBuffer<T> &x)
{
	pimpl_->device_x.view().copy_from(x);
}

template <typename T, int dim>
T BarrierEnergy<T, dim>::val()
{
	auto &device_x = pimpl_->device_x;
	auto &device_contact_area = pimpl_->device_contact_area;
	auto y_ground = pimpl_->y_ground;
	int N = device_x.size() / dim;
	DeviceBuffer<T> device_val(N);
	ParallelFor(256).apply(N, [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), y_ground] __device__(int i) mutable
						   { T d = device_x(dim * i + 1) - y_ground;
						   if(d<dhat){
							   T s = d / dhat;
							   device_val(i)= kappa * device_contact_area(i) * dhat/2*(s-1)*log(s);
						   } })
		.wait();
	return devicesum(device_val);
} // Calculate the energy

template <typename T, int dim>
const DeviceBuffer<T> &BarrierEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto y_ground = pimpl_->y_ground;
	auto &device_contact_area = pimpl_->device_contact_area;
	int N = device_x.size() / dim;
	auto &device_grad = pimpl_->device_grad;
	device_grad.fill(0);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_grad = device_grad.viewer(), y_ground] __device__(int i) mutable
						   {
							   T d = device_x(dim * i + 1) - y_ground;
							   if (d < dhat)
							   {
								   T s = d / dhat;
								   device_grad(i * dim + 1) = device_contact_area(i) * dhat * (kappa / 2 * (log(s) / dhat + (s - 1) / d));
							   } })
		.wait();
	return device_grad;
}

template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &BarrierEnergy<T, dim>::hess()
{
	auto &device_x = pimpl_->device_x;
	auto &device_contact_area = pimpl_->device_contact_area;
	auto &device_hess = pimpl_->device_hess;
	auto device_hess_row_idx = device_hess.row_indices();
	auto device_hess_col_idx = device_hess.col_indices();
	auto y_ground = pimpl_->y_ground;
	auto device_hess_val = device_hess.values();
	int N = device_x.size() / dim;
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N, y_ground] __device__(int i) mutable
						   {
							device_hess_row_idx(i)=i*dim+1;
							device_hess_col_idx(i)=i*dim+1;
		T d = device_x(dim * i + 1) - y_ground;
		if (d < dhat)
		{
			device_hess_val(i) = device_contact_area(i) * dhat * kappa / (2 * d * d * dhat) * (d + dhat);
		}
		else{
			device_hess_val(i) = 0;
		} })
		.wait();
	return device_hess;

} // Calculate the Hessian of the energy

template <typename T, int dim>
T BarrierEnergy<T, dim>::init_step_size(const DeviceBuffer<T> &p)
{
	auto &device_x = pimpl_->device_x;
	auto y_ground = pimpl_->y_ground;
	int N = device_x.size() / dim;
	DeviceBuffer<T> device_alpha(N);
	device_alpha.fill(1);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), p = p.cviewer(), y_ground, device_alpha = device_alpha.viewer()] __device__(int i) mutable
						   {
							   if (p(i * dim + 1) < 0)
							   {
								   device_alpha(i) = 0.9 * (y_ground - device_x(i * dim + 1)) / p(i * dim + 1);
							   } })
		.wait();
	return min_vector(device_alpha);
}
template class BarrierEnergy<float, 2>;
template class BarrierEnergy<float, 3>;
template class BarrierEnergy<double, 2>;
template class BarrierEnergy<double, 3>;