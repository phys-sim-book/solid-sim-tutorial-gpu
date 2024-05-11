#include <muda/muda.h>
#include <muda/container.h>
#include <muda/cub/device/device_reduce.h>
#include <Eigen/Dense>
// utility functions
template <typename T>
T devicesum(muda::DeviceBuffer<T> &buffer);

template <typename T, int Size>
void __device__ make_PSD(const Eigen::Matrix<T, Size, Size> &hess, Eigen::Matrix<T, Size, Size> &PSD);