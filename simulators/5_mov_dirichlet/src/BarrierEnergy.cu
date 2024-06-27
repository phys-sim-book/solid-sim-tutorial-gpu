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
	DeviceBuffer<T> device_contact_area, device_n, device_n_ceil, device_o;
	int N;
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
BarrierEnergy<T, dim>::BarrierEnergy(const std::vector<T> &x, const std::vector<T> n, const std::vector<T> o, const std::vector<T> &contact_area) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->N = x.size() / dim;
	pimpl_->device_x.copy_from(x);
	pimpl_->device_contact_area.copy_from(contact_area);
	std::vector<T> n_ceil(dim);
	n_ceil[1] = -1;
	pimpl_->device_n_ceil.copy_from(n_ceil);
	pimpl_->device_n.copy_from(n);
	pimpl_->device_o.copy_from(o);
	pimpl_->device_hess.resize_triplets((pimpl_->N * 2 - 1) * dim * dim);
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
	auto &device_n = pimpl_->device_n;
	auto &device_n_ceil = pimpl_->device_n_ceil;
	auto &device_o = pimpl_->device_o;
	int N = device_x.size() / dim;
	DeviceBuffer<T> device_val1(N);
	DeviceBuffer<T> device_val2(N);
	ParallelFor(256).apply(N, [device_val1 = device_val1.viewer(), device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_n = device_n.cviewer(), device_o = device_o.cviewer()] __device__(int i) mutable
						   { T d = 0;
						   for(int j=0;j<dim;j++){
							   d += device_n(j)*(device_x(i*dim+j)-device_o(j));
						   }
						   if(d<dhat){
							   T s = d / dhat;
							   device_val1(i)= kappa * device_contact_area(i) * dhat/2*(s-1)*log(s);
						   } })
		.wait();
	ParallelFor(256).apply(N - 1, [device_val2 = device_val2.viewer(), device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_n_ceil = device_n_ceil.cviewer(), device_o = device_o.cviewer(), N] __device__(int i) mutable
						   { T d = 0;
						   for(int j=0;j<dim;j++){
							   d += device_n_ceil(j)*(device_x(i*dim+j)-device_x((N-1)*dim+j));
						   }
						   if(d<dhat){
							   T s = d / dhat;
							   device_val2(i)= kappa * device_contact_area(i) * dhat/2*(s-1)*log(s);
						   } })
		.wait();
	return devicesum(device_val1) + devicesum(device_val2);
} // Calculate the energy

template <typename T, int dim>
const DeviceBuffer<T> &BarrierEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_contact_area = pimpl_->device_contact_area;
	int N = device_x.size() / dim;
	auto &device_n = pimpl_->device_n;
	auto &device_n_ceil = pimpl_->device_n_ceil;
	auto &device_o = pimpl_->device_o;
	auto &device_grad = pimpl_->device_grad;
	device_grad.fill(0);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_grad = device_grad.viewer(), device_n = device_n.cviewer(), device_o = device_o.cviewer()] __device__(int i) mutable

						   {
							   T d = 0;
							   for(int j=0;j<dim;j++){
								   d += device_n(j)*(device_x(i*dim+j)-device_o(j));
							   }
							   if (d < dhat)
							   {
								   T s = d / dhat;
								   for (int j = 0; j < dim; j++)
								   {
									   device_grad(i * dim + j) = device_contact_area(i) * dhat * (kappa / 2 * (log(s) / dhat + (s - 1) / d)) * device_n(j);
								   }
							   } })
		.wait();
	ParallelFor(256).apply(N - 1, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_grad = device_grad.viewer(), device_n_ceil = device_n_ceil.cviewer(), device_o = device_o.cviewer(), N] __device__(int i) mutable

						   {
							   T d = 0;
							   for(int j=0;j<dim;j++){
								   d += device_n_ceil(j)*(device_x(i*dim+j)-device_x((N-1)*dim+j));
							   }
							   if (d < dhat)
							   {
								   T s = d / dhat;
								   for (int j = 0; j < dim; j++)
								   {
									   T grad =device_contact_area(i) * dhat * (kappa / 2 * (log(s) / dhat + (s - 1) / d)) * device_n_ceil(j);
									   device_grad(i * dim + j) += grad;
									   device_grad((N-1) * dim + j) -= grad;
								   }
							   } })
		.wait();
	return device_grad;
}

template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &BarrierEnergy<T, dim>::hess()
{
	auto &device_x = pimpl_->device_x;
	auto &device_contact_area = pimpl_->device_contact_area;
	auto &device_n = pimpl_->device_n;
	auto &device_n_ceil = pimpl_->device_n_ceil;
	auto &device_o = pimpl_->device_o;
	auto &device_hess = pimpl_->device_hess;
	auto device_hess_row_idx = device_hess.row_indices();
	auto device_hess_col_idx = device_hess.col_indices();
	auto device_hess_val = device_hess.values();
	int N = device_x.size() / dim;
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N, device_n = device_n.cviewer(), device_o = device_o.cviewer()] __device__(int i) mutable
						   {
		T d = 0;
		for (int j = 0; j < dim; j++)
		{
			d += device_n(j) * (device_x(i * dim + j) - device_o(j));
		}
		if (d < dhat)
			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim; k++)
				{
					int idx = i * dim * dim + j * dim + k;
					device_hess_row_idx(idx) = i * dim + j;
					device_hess_col_idx(idx) = i * dim + k;
					device_hess_val(idx) = device_contact_area(i) * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * device_n(j) * device_n(k);
				}
			}
		else
			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim; k++)
				{
					int idx = i * dim * dim + j * dim + k;
					device_hess_row_idx(idx) = i * dim + j;
					device_hess_col_idx(idx) = i * dim + k;
					device_hess_val(idx) = 0;
				}
			} })
		.wait();
	ParallelFor(256).apply(N - 1, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N, device_n_ceil = device_n_ceil.cviewer(), device_o = device_o.cviewer()] __device__(int i) mutable
						   {
		T d = 0;
		for (int j = 0; j < dim; j++)
		{
			d += device_n_ceil(j) * (device_x(i * dim + j) - device_x((N-1) * dim + j));
		}
		if (d < dhat)
			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim; k++)
				{
					int idx =N*dim*dim+ i * dim * dim + j * dim + k;
					device_hess_row_idx(idx) = (N-1) * dim + j;
					device_hess_col_idx(idx) = (N-1) * dim + k;
					device_hess_val(idx) = device_contact_area(i) * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * device_n_ceil(j) * device_n_ceil(k);
				}
			}
		else
			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim; k++)
				{
					int idx = N*dim*dim+i * dim * dim + j * dim + k;
					device_hess_row_idx(idx) = (N-1) * dim + j;
					device_hess_col_idx(idx) = (N-1) * dim + k;
					device_hess_val(idx) = 0;
				}
			} })
		.wait();
	return device_hess;

} // Calculate the Hessian of the energy

template <typename T, int dim>
T BarrierEnergy<T, dim>::init_step_size(const DeviceBuffer<T> &p)
{
	auto &device_x = pimpl_->device_x;
	auto &device_n = pimpl_->device_n;
	auto &device_n_ceil = pimpl_->device_n_ceil;
	auto &device_o = pimpl_->device_o;
	int N = device_x.size() / dim;
	DeviceBuffer<T> device_alpha(N);
	device_alpha.fill(1);
	ParallelFor(256)
		.apply(N, [device_x = device_x.cviewer(), p = p.cviewer(), device_alpha = device_alpha.viewer(), device_n = device_n.cviewer(), device_o = device_o.cviewer()] __device__(int i) mutable

			   {
		T p_n = 0;
		for (int j = 0; j < dim; j++)
		{
			p_n += p(i * dim + j) * device_n(j);
		}
		if (p_n < 0)
		{
			T alpha = 0;
			for (int j = 0; j < dim; j++)
			{
				alpha += device_n(j) * (device_x(i * dim + j) - device_o(j));
			}
			device_alpha(i) = min(device_alpha(i), 0.9 * alpha / -p_n);
		} })
		.wait();

	ParallelFor(256)
		.apply(N - 1, [device_x = device_x.cviewer(), p = p.cviewer(), device_alpha = device_alpha.viewer(), device_n_ceil = device_n_ceil.cviewer(), device_o = device_o.cviewer(), N] __device__(int i) mutable

			   {
		T p_n = 0;
		for (int j = 0; j < dim; j++)
		{
			p_n += p(i * dim + j) * device_n_ceil(j);
		}
		if (p_n < 0)
		{
			T alpha = 0;
			for (int j = 0; j < dim; j++)
			{
				alpha += device_n_ceil(j) * (device_x(i * dim + j) - device_x((N-1) * dim + j));
			}
			device_alpha(i) = min(device_alpha(i), 0.9 * alpha / -p_n);
		} })
		.wait();
	return min_vector(device_alpha);
}
template class BarrierEnergy<float, 2>;
template class BarrierEnergy<float, 3>;
template class BarrierEnergy<double, 2>;
template class BarrierEnergy<double, 3>;

template <typename T, int dim>
const DeviceBuffer<T> BarrierEnergy<T, dim>::compute_mu_lambda(T mu)
{
	auto &device_x = pimpl_->device_x;
	auto &device_n = pimpl_->device_n;
	auto &device_o = pimpl_->device_o;
	auto &device_contact_area = pimpl_->device_contact_area;
	int N = device_x.size() / dim;
	DeviceBuffer<T> device_mu_lambda(N);
	device_mu_lambda.fill(0);
	ParallelFor(256)
		.apply(N, [device_x = device_x.cviewer(), device_mu_lambda = device_mu_lambda.viewer(), mu, device_n = device_n.cviewer(), device_o = device_o.cviewer(), device_contact_area = device_contact_area.cviewer()] __device__(int i) mutable
			   {
		T d = 0;
		for (int j = 0; j < dim; j++)
		{
			d += device_n(j) * (device_x(i * dim + j) - device_o(j));
		}
		if (d < dhat)
		{
			T s=d/dhat;
			device_mu_lambda(i) = mu*-device_contact_area(i) * dhat *(kappa / 2 * (log(s) / dhat + (s - 1) / d));
		} })
		.wait();
	return device_mu_lambda;
}