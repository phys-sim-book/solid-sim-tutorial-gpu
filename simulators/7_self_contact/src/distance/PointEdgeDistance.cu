#include "distance.h"

template <typename T>
__device__ __host__ T PointEdgeDistanceVal(const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1)
{
    Eigen::Vector<T, 2> e = e1 - e0;
    T ratio = e.dot(p - e0) / e.dot(e);
    if (ratio < 0)
    {
        return PointPointDistanceVal(p, e0);
    }
    else if (ratio > 1)
    {
        return PointPointDistanceVal(p, e1);
    }
    else
    {
        T val;
        PointLineDistanceVal(val, p, e0, e1);
        return val;
    }
}

template <typename T>
__device__ __host__ Eigen::Matrix<T, 6, 1> PointEdgeDistanceGrad(const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1)
{
    Eigen::Vector<T, 2> e = e1 - e0;
    T ratio = e.dot(p - e0) / e.dot(e);
    if (ratio < 0)
    {
        Eigen::Matrix<T, 4, 1> g_PP = PointPointDistanceGrad(p, e0);
        Eigen::Matrix<T, 6, 1> gradient;
        gradient << g_PP.template segment<2>(0), g_PP.template segment<2>(2), Eigen::Matrix<T, 2, 1>::Zero();
        return gradient;
    }
    else if (ratio > 1)
    {
        Eigen::Matrix<T, 4, 1> g_PP = PointPointDistanceGrad(p, e1);
        Eigen::Matrix<T, 6, 1> gradient;
        gradient << g_PP.template segment<2>(0), Eigen::Matrix<T, 2, 1>::Zero(), g_PP.template segment<2>(2);
        return gradient;
    }
    else
    {
        Eigen::Matrix<T, 6, 1> grad;
        PointLineDistanceGrad(grad, p, e0, e1);
        return grad;
    }
}

template <typename T>
__device__ __host__ Eigen::Matrix<T, 6, 6> PointEdgeDistanceHess(const Eigen::Vector<T, 2> &p, const Eigen::Vector<T, 2> &e0, const Eigen::Vector<T, 2> &e1)
{
    Eigen::Vector<T, 2> e = e1 - e0;
    T ratio = e.dot(p - e0) / e.dot(e);
    if (ratio < 0)
    {
        Eigen::Matrix<T, 4, 4> H_PP = PointPointDistanceHess(p, e0);
        Eigen::Matrix<T, 6, 6> H;
        H.setZero();
        H.template block<2, 2>(0, 0) = H_PP.template block<2, 2>(0, 0);
        H.template block<2, 2>(0, 2) = H_PP.template block<2, 2>(0, 2);
        H.template block<2, 2>(2, 0) = H_PP.template block<2, 2>(2, 0);
        H.template block<2, 2>(2, 2) = H_PP.template block<2, 2>(2, 2);
        return H;
    }
    else if (ratio > 1)
    {
        Eigen::Matrix<T, 4, 4> H_PP = PointPointDistanceHess(p, e1);
        Eigen::Matrix<T, 6, 6> H;
        H.setZero();
        H.template block<2, 2>(0, 0) = H_PP.template block<2, 2>(0, 0);
        H.template block<2, 2>(0, 4) = H_PP.template block<2, 2>(0, 2);
        H.template block<2, 2>(4, 0) = H_PP.template block<2, 2>(2, 0);
        H.template block<2, 2>(4, 4) = H_PP.template block<2, 2>(2, 2);
        return H;
    }
    else
    {
        Eigen::Matrix<T, 6, 6> hess;
        PointLineDistanceHess(hess, p, e0, e1);
        return hess;
    }
}

template __device__ __host__ float PointEdgeDistanceVal(const Eigen::Vector2f &p, const Eigen::Vector2f &e0, const Eigen::Vector2f &e1);
template __device__ __host__ Eigen::Matrix<float, 6, 1> PointEdgeDistanceGrad(const Eigen::Vector2f &p, const Eigen::Vector2f &e0, const Eigen::Vector2f &e1);
template __device__ __host__ Eigen::Matrix<float, 6, 6> PointEdgeDistanceHess(const Eigen::Vector2f &p, const Eigen::Vector2f &e0, const Eigen::Vector2f &e1);

template __device__ __host__ double PointEdgeDistanceVal(const Eigen::Vector2d &p, const Eigen::Vector2d &e0, const Eigen::Vector2d &e1);
template __device__ __host__ Eigen::Matrix<double, 6, 1> PointEdgeDistanceGrad(const Eigen::Vector2d &p, const Eigen::Vector2d &e0, const Eigen::Vector2d &e1);
template __device__ __host__ Eigen::Matrix<double, 6, 6> PointEdgeDistanceHess(const Eigen::Vector2d &p, const Eigen::Vector2d &e0, const Eigen::Vector2d &e1);
