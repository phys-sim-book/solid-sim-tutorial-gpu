#include "uti.h"
#include <muda/cub/device/device_reduce.h>
#include <muda/ext/linear_system.h>
#include <muda/container.h>
#include <muda/muda.h>

using namespace muda;
template <typename T>
SparseMatrix<T>::SparseMatrix(int size) : size(size)
{
    row_idx = std::vector<int>(size);
    col_idx = std::vector<int>(size);
    val = std::vector<T>(size);
}
template <typename T>
SparseMatrix<T>::SparseMatrix() = default;

template <typename T>
SparseMatrix<T>::~SparseMatrix() = default;

template <typename T>
void SparseMatrix<T>::set_value(int row, int col, T value, int loc)
{
    assert(loc < size);
    row_idx[loc] = row;
    col_idx[loc] = col;
    val[loc] = value;
}
template <typename T>
void SparseMatrix<T>::set_diagonal(T value)
{
    for (int i = 0; i < size; i++)
    {
        set_value(i, i, value, i);
    }
}
template <typename T>
SparseMatrix<T> &SparseMatrix<T>::combine(const SparseMatrix<T> &other)
{
    int old_size = size;
    size += other.size;
    row_idx.resize(size);
    col_idx.resize(size);
    val.resize(size);
    // copy memory
    std::copy(other.row_idx.begin(), other.row_idx.end(), row_idx.begin() + old_size);
    std::copy(other.col_idx.begin(), other.col_idx.end(), col_idx.begin() + old_size);
    std::copy(other.val.begin(), other.val.end(), val.begin() + old_size);
    return *this;
}
template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator*(const T &a)
{
    val = mult_vector(val, a);
    return *this;
}
template <typename T>
const std::vector<int> &SparseMatrix<T>::get_row_buffer() const
{
    return row_idx;
}
template <typename T>
const std::vector<int> &SparseMatrix<T>::get_col_buffer() const
{
    return col_idx;
}
template <typename T>
const std::vector<T> &SparseMatrix<T>::get_val_buffer() const
{
    return val;
}

template <typename T>
std::vector<int> &SparseMatrix<T>::set_row_buffer()
{
    return row_idx;
}
template <typename T>
std::vector<int> &SparseMatrix<T>::set_col_buffer()
{
    return col_idx;
}
template <typename T>
std::vector<T> &SparseMatrix<T>::set_val_buffer()
{
    return val;
}
template <typename T>
int SparseMatrix<T>::get_size() const
{
    return size;
}
template class SparseMatrix<float>;
template class SparseMatrix<double>;

template <typename T>
std::vector<T> add_vector(const std::vector<T> &a, const std::vector<T> &b, const T &factor1, const T &factor2)
{
    DeviceBuffer<T> a_device(a);
    DeviceBuffer<T> b_device(b);
    int N = a.size();
    DeviceBuffer<T> c_device(N);
    ParallelFor(256)
        .apply(N,
               [c_device = c_device.viewer(), a_device = a_device.cviewer(), b_device = b_device.cviewer(), factor1, factor2] __device__(int i) mutable
               {
                   c_device(i) = a_device(i) * factor1 + b_device(i) * factor2;
               })
        .wait();
    std::vector<T> c_host(N);
    c_device.copy_to(c_host);
    return c_host;
}
template std::vector<float> add_vector<float>(const std::vector<float> &a, const std::vector<float> &b, const float &factor1, const float &factor2);
template std::vector<double> add_vector<double>(const std::vector<double> &a, const std::vector<double> &b, const double &factor1, const double &factor2);
template <typename T>
std::vector<T> mult_vector(const std::vector<T> &a, const T &b)
{
    DeviceBuffer<T> a_device(a);
    int N = a.size();
    DeviceBuffer<T> c_device(N);
    ParallelFor(256)
        .apply(N,
               [c_device = c_device.viewer(), a_device = a_device.cviewer(), b] __device__(int i) mutable
               {
                   c_device(i) = a_device(i) * b;
               })
        .wait();
    std::vector<T> c_host(N);
    c_device.copy_to(c_host);
    return c_host;
}
template std::vector<float> mult_vector<float>(const std::vector<float> &a, const float &b);
template std::vector<double> mult_vector<double>(const std::vector<double> &a, const double &b);

template <typename T>
T max_vector(const std::vector<T> &a)
{
    DeviceBuffer<T> buffer(a);
    T vec_max = 0.0f;              // Result of the reduction
    T *d_out;                      // Device memory to store the result of the reduction
    cudaMalloc(&d_out, sizeof(T)); // Allocate memory for the result

    // DeviceReduce is assumed to be part of the 'muda' library or similar
    DeviceReduce().Sum(buffer.data(), d_out, buffer.size());

    // Copy the result back to the host
    cudaMemcpy(&vec_max, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_out);
    return vec_max;
}
template float max_vector<float>(const std::vector<float> &a);
template double max_vector<double>(const std::vector<double> &a);

template <typename T>
void search_dir(const std::vector<T> &grad, const SparseMatrix<T> &hess, std::vector<T> &dir)
{
    LinearSystemContext ctx;
    int N = grad.size();
    DeviceDenseVector<T> x_device(N);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> e_grad(grad.data(), grad.size());
    DeviceDenseVector<T> grad_device(e_grad);
    DeviceTripletMatrix<T, 1> Hess;
    Hess.reshape(N, N);
    Hess.resize_triplets(hess.get_size());
    Hess.row_indices().copy_from(hess.get_row_buffer().data());
    Hess.col_indices().copy_from(hess.get_col_buffer().data());
    Hess.values().copy_from(hess.get_val_buffer().data());
    DeviceCOOMatrix<T> A_coo;
    ctx.convert(Hess, A_coo);
    DeviceCSRMatrix<T> A_csr;
    ctx.convert(A_coo, A_csr);
    ctx.solve(x_device.view(), A_csr.cview(), grad_device.cview());
    ctx.sync();
    x_device.copy_to(dir);
}
template void search_dir<float>(const std::vector<float> &grad, const SparseMatrix<float> &hess, std::vector<float> &dir);
template void search_dir<double>(const std::vector<double> &grad, const SparseMatrix<double> &hess, std::vector<double> &dir);