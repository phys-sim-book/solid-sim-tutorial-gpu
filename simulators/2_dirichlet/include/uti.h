#pragma once
#include <Eigen/Dense>
#include "SparseMatrix.h"
#include <muda/cub/device/device_reduce.h>
#include <muda/container.h>
#include <muda/muda.h>
#include <muda/ext/linear_system.h>
using namespace muda;

template <typename T>
DeviceBuffer<T> add_vector(const DeviceBuffer<T> &a, const DeviceBuffer<T> &b, const T &factor1 = 1, const T &factor2 = 1);

template <typename T>
DeviceBuffer<T> mult_vector(const DeviceBuffer<T> &a, const T &b);

template <typename T>
DeviceTripletMatrix<T, 1> add_triplet(const DeviceTripletMatrix<T, 1> &a, const DeviceTripletMatrix<T, 1> &b, const T &factor1 = 1, const T &factor2 = 1);

template <typename T>
T max_vector(const DeviceBuffer<T> &a);

template <typename T, int dim>
void search_dir(const DeviceBuffer<T> &grad, const DeviceTripletMatrix<T, 1> &hess, DeviceBuffer<T> &dir, const DeviceBuffer<int> &DBC);

template <typename T>
void display_vec(const DeviceBuffer<T> &vec);

template <typename T, int dim>
void set_DBC(DeviceBuffer<T> &grad, DeviceCSRMatrix<T> &hess, const DeviceBuffer<int> &DBC);