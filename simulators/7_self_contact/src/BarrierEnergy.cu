#include "BarrierEnergy.h"
#include "distance.h"
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
	DeviceBuffer<int> device_bp, device_be;
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
BarrierEnergy<T, dim>::BarrierEnergy(const std::vector<T> &x, const std::vector<T> &n, const std::vector<T> &o, const std::vector<int> &be, const std::vector<int> &bp, const std::vector<T> &contact_area) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->N = x.size() / dim;
	pimpl_->device_x.copy_from(x);
	pimpl_->device_contact_area.copy_from(contact_area);
	std::vector<T> n_ceil(dim);
	n_ceil[1] = -1;
	n_ceil[0] = 0;
	pimpl_->device_n_ceil.copy_from(n_ceil);
	pimpl_->device_n.copy_from(n);
	pimpl_->device_o.copy_from(o);
	pimpl_->device_bp.copy_from(bp);
	pimpl_->device_be.copy_from(be);
	pimpl_->device_hess.resize_triplets(pimpl_->N * dim * dim + (pimpl_->N - 1) * dim * dim * 4 + bp.size() * be.size() / 2 * 36);
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
	auto &device_bp = pimpl_->device_bp;
	auto &device_be = pimpl_->device_be;
	int N = device_x.size() / dim, Nbp = device_bp.size(), Nbe = device_be.size() / 2;
	int Npe = Nbp * Nbe;
	DeviceBuffer<T> device_val1(N);
	DeviceBuffer<T> device_val2(N);
	DeviceBuffer<T> device_val3(Npe);
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
	ParallelFor(256).apply(Npe, [device_val3 = device_val3.viewer(), device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_bp = device_bp.cviewer(), device_be = device_be.cviewer(), Nbp, Nbe] __device__(int i) mutable
						   {
							   int xI = device_bp(i / Nbe);
							   int eI0 = device_be(2*(i % Nbe)), eI1 = device_be(2*(i % Nbe) + 1); 
							   if(xI!=eI0 &&xI!=eI1){
								T dhatsqr= dhat*dhat;
								Eigen::Vector<T, 2> p,e0,e1;
								p<<device_x(xI*dim),device_x(xI*dim+1);
								e0<<device_x(eI0*dim),device_x(eI0*dim+1);
								e1<<device_x(eI1*dim),device_x(eI1*dim+1);
								T d_sqr=PointEdgeDistanceVal(p,e0,e1);
								if(d_sqr<dhatsqr){
									T s = d_sqr / dhatsqr;
									device_val3(i)= kappa * device_contact_area(xI) * dhat/8*(s-1)*log(s);
								}
							   } })
		.wait();
	return devicesum(device_val1) +
		   devicesum(device_val2) +
		   devicesum(device_val3);
} // Calculate the energy

template <typename T, int dim>
const DeviceBuffer<T> &BarrierEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_contact_area = pimpl_->device_contact_area;
	int N = device_x.size() / dim;
	int Nbp = pimpl_->device_bp.size(), Nbe = pimpl_->device_be.size() / 2;
	int Npe = Nbp * Nbe;
	auto &device_n = pimpl_->device_n;
	auto &device_n_ceil = pimpl_->device_n_ceil;
	auto &device_o = pimpl_->device_o;
	auto &device_grad = pimpl_->device_grad;
	auto &device_bp = pimpl_->device_bp;
	auto &device_be = pimpl_->device_be;
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
									   atomicAdd(&device_grad((N-1) * dim + j), -grad);
								   }
							   } })
		.wait();
	ParallelFor(256).apply(Npe, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_grad = device_grad.viewer(), device_bp = device_bp.cviewer(), device_be = device_be.cviewer(), Nbp, Nbe] __device__(int i) mutable
						   {
							   int xI = device_bp(i / Nbe);
							   int eI0 = device_be(2*(i % Nbe)), eI1 = device_be(2*(i % Nbe) + 1); 
							   if(xI!=eI0 &&xI!=eI1){
								T dhatsqr= dhat*dhat;
								Eigen::Vector<T, 2> p,e0,e1;
								p<<device_x(xI*dim),device_x(xI*dim+1);
								e0<<device_x(eI0*dim),device_x(eI0*dim+1);
								e1<<device_x(eI1*dim),device_x(eI1*dim+1);
								T d_sqr=PointEdgeDistanceVal(p,e0,e1);
								if(d_sqr<dhatsqr){
									T s = d_sqr / dhatsqr;
									Eigen::Vector<T, 6> grad =  0.5 * device_contact_area(xI) * dhat * (kappa / 8 * (log(s) / dhatsqr + (s - 1) / d_sqr)) * PointEdgeDistanceGrad(p,e0,e1);
									atomicAdd(&device_grad(xI*dim),grad(0));
									atomicAdd(&device_grad(xI*dim+1),grad(1));
									atomicAdd(&device_grad(eI0*dim),grad(2));
									atomicAdd(&device_grad(eI0*dim+1),grad(3));
									atomicAdd(&device_grad(eI1*dim),grad(4));
									atomicAdd(&device_grad(eI1*dim+1),grad(5));
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
	auto device_bp = pimpl_->device_bp;
	auto device_be = pimpl_->device_be;
	int Nbp = device_bp.size(), Nbe = device_be.size() / 2;
	int Npe = Nbp * Nbe;
	int N = device_x.size() / dim;
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N, device_n = device_n.cviewer(), device_o = device_o.cviewer()] __device__(int i) mutable
						   {
		T d = 0;
		for (int j = 0; j < dim; j++)
		{
			d += device_n(j) * (device_x(i * dim + j) - device_o(j));
		}

			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim; k++)
				{
					int idx = i * dim * dim + j * dim + k;
					device_hess_row_idx(idx) = i * dim + j;
					device_hess_col_idx(idx) = i * dim + k;
					if (d < dhat)
						device_hess_val(idx) = device_contact_area(i) * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * device_n(j) * device_n(k);
					else
						device_hess_val(idx) = 0;
				}
			} })
		.wait();
	ParallelFor(256).apply(N - 1, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N, device_n_ceil = device_n_ceil.cviewer(), device_o = device_o.cviewer()] __device__(int i) mutable
						   {
		T d = 0;
		for (int j = 0; j < dim; j++)
		{
			d += device_n_ceil(j) * (device_x(i * dim + j) - device_x((N - 1) * dim + j));
		}
			int index[2] = {i, N - 1};
			for (int nI = 0; nI < 2; nI++)
				for (int nJ = 0; nJ < 2; nJ++)
					for (int j = 0; j < dim; j++)
					{
						for (int k = 0; k < dim; k++)
						{
							int idx = N * dim * dim + i * dim * dim * 4 + nI * dim * dim * 2 + nJ * dim * dim + j * dim + k;
							device_hess_row_idx(idx) = index[nI] * dim + j;
							device_hess_col_idx(idx) = index[nJ] * dim + k;
							if (d < dhat)
								if (nI == nJ)
									device_hess_val(idx) = device_contact_area(i) * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * device_n_ceil(j) * device_n_ceil(k);
								else
									device_hess_val(idx) = -device_contact_area(i) * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * device_n_ceil(j) * device_n_ceil(k);
							else
								device_hess_val(idx) = 0;
						}
					} })
		.wait();
	ParallelFor(256).apply(Npe, [device_x = device_x.cviewer(), device_contact_area = device_contact_area.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N, device_bp = device_bp.cviewer(), device_be = device_be.cviewer(), Nbp, Nbe] __device__(int i) mutable
						   {
							   int xI = device_bp(i / Nbe);
							   int eI0 = device_be(2*(i % Nbe)), eI1 = device_be(2*(i % Nbe) + 1);
							   int index[3] = {xI, eI0, eI1};
							    for (int nI = 0; nI < 3; nI++)
									for (int nJ = 0; nJ < 3; nJ++)
										for (int j = 0; j < dim; j++)
										{
											for (int k = 0; k < dim; k++)
											{
												int idx = N * dim * dim + (N - 1) * dim * dim * 4 + i * dim * dim*9 + nI * dim * dim*3 + nJ * dim * dim + j * dim + k;
												device_hess_row_idx(idx) = index[nI] * dim + j;
												device_hess_col_idx(idx) = index[nJ] * dim + k;
												device_hess_val(idx) = 0;
											}
										}
								if (xI != eI0 && xI != eI1){
									T dhat_sqr = dhat * dhat;
									Eigen::Vector<T, 2> p, e0, e1;
									p << device_x(xI * dim), device_x(xI * dim + 1);
									e0 << device_x(eI0 * dim), device_x(eI0 * dim + 1);
									e1 << device_x(eI1 * dim), device_x(eI1 * dim + 1);
									T d_sqr = PointEdgeDistanceVal(p, e0, e1);
									if (d_sqr < dhat_sqr)
									{
										Eigen::Vector<T, 6> grad=PointEdgeDistanceGrad(p,e0,e1);
										T s = d_sqr / dhat_sqr;
										Eigen::Matrix<T, 6, 6> localhess,PSD;
										localhess=kappa / (8 * d_sqr * d_sqr * dhat_sqr) * (d_sqr + dhat_sqr) * grad * grad.transpose()
                        + (kappa / 8 * (log(s) / dhat_sqr + (s - 1) / d_sqr)) * PointEdgeDistanceHess(p,e0,e1);
										make_PSD(localhess,PSD);
										localhess=0.5 * device_contact_area(xI) * dhat * PSD;
										for (int nI = 0; nI < 3; nI++)
											for (int nJ = 0; nJ < 3; nJ++)
												for (int j = 0; j < dim; j++)
												{
													for (int k = 0; k < dim; k++)
													{
														int idx = N * dim * dim + (N - 1) * dim * dim * 4 + i * dim * dim*9 + nI * dim * dim*3 + nJ * dim * dim + j * dim + k;
														device_hess_val(idx) = localhess(nI*dim+j,nJ*dim+k);
													}
												}
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
	int Nbp = pimpl_->device_bp.size(), Nbe = pimpl_->device_be.size() / 2;
	int Npe = Nbp * Nbe;
	DeviceBuffer<T> device_alpha(N);
	DeviceBuffer<T> device_alpha1(Npe);
	device_alpha.fill(1);
	device_alpha1.fill(1);
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
			//printf("alpha: %f\n", device_alpha(i));
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
			//printf("alpha: %f\n", device_alpha(i));
		} })
		.wait();
	T current_alpha = min_vector(device_alpha);
	ParallelFor(256)
		.apply(Npe, [current_alpha, device_x = device_x.cviewer(), P = p.cviewer(), device_alpha1 = device_alpha1.viewer(), device_bp = pimpl_->device_bp.cviewer(), device_be = pimpl_->device_be.cviewer(), Nbp, Nbe] __device__(int i) mutable
			   {
				   int xI = device_bp(i / Nbe);
				   int eI0 = device_be(2*(i % Nbe)), eI1 = device_be(2*(i % Nbe) + 1);
				   if (xI != eI0 && xI != eI1){
					   Eigen::Matrix<T, 2, 1> p, e0, e1, dp, de0, de1; 
				   p<<device_x(xI*dim),device_x(xI*dim+1);
				   e0<<device_x(eI0*dim),device_x(eI0*dim+1);
				   e1<<device_x(eI1*dim),device_x(eI1*dim+1);
				   dp<<P(xI*dim),P(xI*dim+1);
				   de0<<P(eI0*dim),P(eI0*dim+1);
				   de1<<P(eI1*dim),P(eI1*dim+1); 
				   if (bbox_overlap(p,e0,e1,dp,de0,de1,current_alpha)){
					 T toc=narrow_phase_CCD(p,e0,e1,dp,de0,de1,current_alpha);
					 device_alpha1(i)=min(device_alpha1(i),toc);
				   }} })
		.wait();
	return min(min_vector(device_alpha1), current_alpha);
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