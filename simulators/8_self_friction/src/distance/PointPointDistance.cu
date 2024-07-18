#include "distance.h"

template <typename T>
__device__ __host__ T PointPointDistanceVal(const Eigen::Vector<T, 2> &p0, const Eigen::Vector<T, 2> &p1)
{
    Eigen::Vector<T, 2> e = p0 - p1;
    return e.dot(e);
}

template <typename T>
__device__ __host__ Eigen::Matrix<T, 4, 1> PointPointDistanceGrad(const Eigen::Vector<T, 2> &p0, const Eigen::Vector<T, 2> &p1)
{
    Eigen::Vector<T, 2> e = p0 - p1;
    Eigen::Matrix<T, 4, 1> gradient;
    gradient << 2 * e(0), 2 * e(1), -2 * e(0), -2 * e(1);
    return gradient;
}

template <typename T>
__device__ __host__ Eigen::Matrix<T, 4, 4> PointPointDistanceHess(const Eigen::Vector<T, 2> &p0, const Eigen::Vector<T, 2> &p1)
{
    Eigen::Matrix<T, 4, 4> H = Eigen::Matrix<T, 4, 4>::Zero();
    H(0, 0) = H(1, 1) = H(2, 2) = H(3, 3) = 2;
    H(0, 2) = H(1, 3) = H(2, 0) = H(3, 1) = -2;
    return H;
}

template __device__ __host__ float PointPointDistanceVal(const Eigen::Vector2f &p0, const Eigen::Vector2f &p1);
template __device__ __host__ Eigen::Matrix<float, 4, 1> PointPointDistanceGrad(const Eigen::Vector2f &p0, const Eigen::Vector2f &p1);
template __device__ __host__ Eigen::Matrix<float, 4, 4> PointPointDistanceHess(const Eigen::Vector2f &p0, const Eigen::Vector2f &p1);

template __device__ __host__ double PointPointDistanceVal(const Eigen::Vector2d &p0, const Eigen::Vector2d &p1);
template __device__ __host__ Eigen::Matrix<double, 4, 1> PointPointDistanceGrad(const Eigen::Vector2d &p0, const Eigen::Vector2d &p1);
template __device__ __host__ Eigen::Matrix<double, 4, 4> PointPointDistanceHess(const Eigen::Vector2d &p0, const Eigen::Vector2d &p1);