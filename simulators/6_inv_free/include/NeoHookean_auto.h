#pragma once
#include "Eigen/Dense"
#include <muda/cub/device/device_reduce.h>
#include <muda/container.h>
#include <muda/muda.h>
#include <muda/ext/linear_system.h>
template <typename T>
__device__ __host__ void NeoHookeanEnergyVal(T &E, const T &Mu, const T &Lambda, const Eigen::Vector<T, 6> &X, const Eigen::Matrix<T, 2, 2> &IB, const T &vol);

template <typename T>
__device__ __host__ void NeoHookeanEnergyGradient(Eigen::Vector<T, 6> &G, const T &Mu, const T &Lambda, const Eigen::Vector<T, 6> &X, const Eigen::Matrix<T, 2, 2> &IB, const T &vol);

template <typename T>
__device__ __host__ void NeoHookeanEnergyHessian(Eigen::Matrix<T, 6, 6> &H, const T &Mu, const T &Lambda, const Eigen::Vector<T, 6> &X, const Eigen::Matrix<T, 2, 2> &IB, const T &vol);