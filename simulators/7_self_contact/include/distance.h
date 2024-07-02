#pragma once
#include "Eigen/Dense"
#include <muda/muda.h>

template <typename T>
__device__ __host__ void PointLineDistanceVal(T &val, const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1);

template <typename T>
__device__ __host__ void PointLineDistanceGrad(Eigen::Vector<T, 6> &grad, const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1);

template <typename T>
__device__ __host__ void PointLineDistanceHess(Eigen::Matrix<T, 6, 6> &hess, const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1);

template <typename T>
__device__ __host__ T PointPointDistanceVal(const Eigen::Vector<T, 2> &p0, const Eigen::Vector<T, 2> &p1);

template <typename T>
__device__ __host__ Eigen::Matrix<T, 4, 1> PointPointDistanceGrad(const Eigen::Vector<T, 2> &p0, const Eigen::Vector<T, 2> &p1);

template <typename T>
__device__ __host__ Eigen::Matrix<T, 4, 4> PointPointDistanceHess(const Eigen::Vector<T, 2> &p0, const Eigen::Vector<T, 2> &p1);

template <typename T>
__device__ __host__ T PointEdgeDistanceVal(const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1);

template <typename T>
__device__ __host__ Eigen::Matrix<T, 6, 1> PointEdgeDistanceGrad(const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1);

template <typename T>
__device__ __host__ Eigen::Matrix<T, 6, 6> PointEdgeDistanceHess(const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1);

template <typename T>
__device__ __host__ bool bbox_overlap(const Eigen::Matrix<T, 2, 1> &p, const Eigen::Matrix<T, 2, 1> &e0, const Eigen::Matrix<T, 2, 1> &e1,
                  const Eigen::Matrix<T, 2, 1> &dp, const Eigen::Matrix<T, 2, 1> &de0, const Eigen::Matrix<T, 2, 1> &de1,
                  T toc_upperbound);

template <typename T>
__device__ __host__ T narrow_phase_CCD(Eigen::Matrix<T, 2, 1> p, Eigen::Matrix<T, 2, 1> e0, Eigen::Matrix<T, 2, 1> e1,
                   Eigen::Matrix<T, 2, 1> dp, Eigen::Matrix<T, 2, 1> de0, Eigen::Matrix<T, 2, 1> de1,
                   T toc_upperbound);