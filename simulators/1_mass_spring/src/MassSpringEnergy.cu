#include "MassSpringEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <stdio.h>
#include "device_uti.h"

using namespace muda;

template <typename T, int dim>
struct MassSpringEnergy<T, dim>::Impl
{
	DeviceBuffer<T> device_x;
	DeviceBuffer<T> device_l2, device_k;
	DeviceBuffer<int> device_e;
	int N;
	DeviceBuffer<T> device_grad;
	DeviceTripletMatrix<T, 1> device_hess;
};
template <typename T, int dim>
MassSpringEnergy<T, dim>::MassSpringEnergy() = default;

template <typename T, int dim>
MassSpringEnergy<T, dim>::~MassSpringEnergy() = default;

template <typename T, int dim>
MassSpringEnergy<T, dim>::MassSpringEnergy(MassSpringEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringEnergy<T, dim> &MassSpringEnergy<T, dim>::operator=(MassSpringEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringEnergy<T, dim>::MassSpringEnergy(const MassSpringEnergy<T, dim> &rhs)
	: pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
MassSpringEnergy<T, dim>::MassSpringEnergy(const std::vector<T> &x, const std::vector<int> &e, const std::vector<T> &l2, const std::vector<T> &k) : pimpl_{std::make_unique<Impl>()}
{
	pimpl_->N = x.size() / dim;
	pimpl_->device_x.copy_from(x);
	pimpl_->device_e.copy_from(e);
	pimpl_->device_l2.copy_from(l2);
	pimpl_->device_k.copy_from(k);
	pimpl_->device_hess.resize_triplets(pimpl_->device_e.size() / 2 * dim * dim * 4);
	pimpl_->device_hess.reshape(x.size(), x.size());
	pimpl_->device_grad.resize(pimpl_->N * dim);
	int size = e.size() / 2;
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_x(const DeviceBuffer<T> &x)
{
	pimpl_->device_x.view().copy_from(x);
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_e(const std::vector<int> &e)
{
	pimpl_->device_e.copy_from(e);
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_l2(const std::vector<T> &l2)
{
	pimpl_->device_l2.copy_from(l2);
}

template <typename T, int dim>
void MassSpringEnergy<T, dim>::update_k(const std::vector<T> &k)
{
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
									T diffi = device_x(dim * idx1 + d) - device_x(dim * idx2 + d);
									diff += diffi * diffi;
						   		}
								device_val(i) = 0.5 * device_l2(i) * device_k(i) * (diff / device_l2(i) - 1) * (diff / device_l2(i) - 1); })
		.wait();

	return devicesum(device_val);
} // Calculate the energy

template <typename T, int dim>
const DeviceBuffer<T> &MassSpringEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto N = pimpl_->device_e.size() / 2;
	auto &device_grad = pimpl_->device_grad;
	device_grad.fill(0);
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_grad = device_grad.viewer()] __device__(int i) mutable
						   {
							int idx1= device_e(2 * i); // First node index
							int idx2 = device_e(2 * i + 1); // Second node index
							T diff = 0;
							T diffi[dim];
							for (int d = 0; d < dim;d++){
								diffi[d] = device_x(dim * idx1 + d) - device_x(dim * idx2 + d);
								diff += diffi[d] * diffi[d];
							}
						   T factor = 2 * device_k(i) * (diff / device_l2(i) -1);
						   for(int d=0;d<dim;d++){
							   atomicAdd(&device_grad(dim * idx1 + d), factor * diffi[d]);
							   atomicAdd(&device_grad(dim * idx2 + d), -factor * diffi[d]);
							  
						   } })
		.wait();
	// display_vec(device_grad);
	return device_grad;
}

template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &MassSpringEnergy<T, dim>::hess()
{
	auto &device_x = pimpl_->device_x;
	auto &device_e = pimpl_->device_e;
	auto &device_l2 = pimpl_->device_l2;
	auto &device_k = pimpl_->device_k;
	auto N = device_e.size() / 2;
	auto &device_hess = pimpl_->device_hess;
	auto device_hess_row_idx = device_hess.row_indices();
	auto device_hess_col_idx = device_hess.col_indices();
	auto device_hess_val = device_hess.values();
	ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_l2 = device_l2.cviewer(), device_k = device_k.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), N] __device__(int i) mutable
						   {
		int idx[2] = {device_e(2 * i), device_e(2 * i + 1)}; // First node index
		T diff = 0;
		T diffi[dim];
		for (int d = 0; d < dim; d++)
		{
			diffi[d] = device_x(dim * idx[0] + d) - device_x(dim * idx[1] + d);
			diff += diffi[d] * diffi[d];
		}
		Eigen::Matrix<T, dim, 1> diff_vec(diffi);
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
			{
				int indStart = i * 4*dim*dim + (ni * 2 + nj) * dim*dim;
				for (int d1 = 0; d1 < dim; d1++)
					for (int d2 = 0; d2 < dim; d2++){
						device_hess_row_idx(indStart + d1 * dim + d2)= idx[ni]*dim + d1;
						device_hess_col_idx(indStart + d1 * dim + d2)= idx[nj] * dim + d2;
						device_hess_val(indStart + d1 * dim + d2) = H_local(ni * dim + d1, nj * dim + d2);
					}
			} })
		.wait();
	return device_hess;

} // Calculate the Hessian of the energy

template class MassSpringEnergy<float, 2>;
template class MassSpringEnergy<float, 3>;
template class MassSpringEnergy<double, 2>;
template class MassSpringEnergy<double, 3>;