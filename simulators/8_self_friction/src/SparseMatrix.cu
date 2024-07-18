#include <muda/muda.h>
#include <muda/container.h>
#include "SparseMatrix.h"

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
SparseMatrix<T>::~SparseMatrix()
{
    val.clear();
    row_idx.clear();
    col_idx.clear();
}

template <typename T>
SparseMatrix<T>::SparseMatrix(SparseMatrix<T> &&rhs)
{
    size = rhs.size;
    row_idx = std::move(rhs.row_idx);
    col_idx = std::move(rhs.col_idx);
    val = std::move(rhs.val);
}

template <typename T>
SparseMatrix<T> &SparseMatrix<T>::operator=(SparseMatrix<T> &&rhs)
{
    size = rhs.size;
    row_idx = std::move(rhs.row_idx);
    col_idx = std::move(rhs.col_idx);
    val = std::move(rhs.val);
    return *this;
}
template <typename T>
SparseMatrix<T> &SparseMatrix<T>::operator=(SparseMatrix<T> &rhs)
{
    size = rhs.size;
    row_idx = rhs.row_idx;
    col_idx = rhs.col_idx;
    val = rhs.val;
    return *this;
}
template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T> &rhs)
{
    size = rhs.size;
    row_idx = rhs.row_idx;
    col_idx = rhs.col_idx;
    val = rhs.val;
}

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
    for (int i = 0; i < other.size; i++)
    {
        set_value(other.row_idx[i], other.col_idx[i], other.val[i], i + old_size);
    }
    return *this;
}
template <typename T>
SparseMatrix<T> &SparseMatrix<T>::operator*(const T &a)
{
    DeviceBuffer<T> val_device(val);
    int N = val.size();
    DeviceBuffer<T> c_device(N);
    ParallelFor(256)
        .apply(N,
               [c_device = c_device.viewer(), val_device = val_device.cviewer(), a] __device__(int i) mutable
               {
                   c_device(i) = val_device(i) * a;
               })
        .wait();
    c_device.copy_to(val);
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