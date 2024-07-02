#include "distance.h"
template <typename T>
__device__ __host__ bool bbox_overlap(const Eigen::Matrix<T, 2, 1> &p, const Eigen::Matrix<T, 2, 1> &e0, const Eigen::Matrix<T, 2, 1> &e1,
                                      const Eigen::Matrix<T, 2, 1> &dp, const Eigen::Matrix<T, 2, 1> &de0, const Eigen::Matrix<T, 2, 1> &de1,
                                      T toc_upperbound)
{
    Eigen::Matrix<T, 2, 1> max_p = p.cwiseMax(p + toc_upperbound * dp);                                                     // point trajectory bbox top-right
    Eigen::Matrix<T, 2, 1> min_p = p.cwiseMin(p + toc_upperbound * dp);                                                     // point trajectory bbox bottom-left
    Eigen::Matrix<T, 2, 1> max_e = e0.cwiseMax(e0 + toc_upperbound * de0).cwiseMax(e1.cwiseMax(e1 + toc_upperbound * de1)); // edge trajectory bbox top-right
    Eigen::Matrix<T, 2, 1> min_e = e0.cwiseMin(e0 + toc_upperbound * de0).cwiseMin(e1.cwiseMin(e1 + toc_upperbound * de1)); // edge trajectory bbox bottom-left

    if (min_p[0] > max_e[0] || min_p[1] > max_e[1] || min_e[0] > max_p[0] || min_e[1] > max_p[1]) // no overlap
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <typename T>
__device__ __host__ T narrow_phase_CCD(Eigen::Matrix<T, 2, 1> p, Eigen::Matrix<T, 2, 1> e0, Eigen::Matrix<T, 2, 1> e1,
                                       Eigen::Matrix<T, 2, 1> dp, Eigen::Matrix<T, 2, 1> de0, Eigen::Matrix<T, 2, 1> de1,
                                       T toc_upperbound)
{
    // use relative displacement for faster convergence
    Eigen::Matrix<T, 2, 1> mov = (dp + de0 + de1) / 3.0;
    de0 -= mov;
    de1 -= mov;
    dp -= mov;
    T maxDispMag = dp.norm() + std::sqrt(std::max(de0.dot(de0), de1.dot(de1)));
    if (maxDispMag == 0)
    {
        return toc_upperbound;
    }

    T eta = 0.1; // calculate the toc that first brings the distance to 0.1x the current distance
    T dist2_cur = PointEdgeDistanceVal(p, e0, e1);
    T dist_cur = std::sqrt(dist2_cur);
    T gap = eta * dist_cur;
    // iteratively move the point and edge towards each other and
    // grow the toc estimate without numerical errors
    T toc = 0;
    while (true)
    {
        T tocLowerBound = (1 - eta) * dist_cur / maxDispMag;

        p += tocLowerBound * dp;
        e0 += tocLowerBound * de0;
        e1 += tocLowerBound * de1;
        dist2_cur = PointEdgeDistanceVal(p, e0, e1);
        dist_cur = std::sqrt(dist2_cur);
        if (toc != 0 && dist_cur < gap)
        {
            break;
        }

        toc += tocLowerBound;
        if (toc > toc_upperbound)
        {
            return toc_upperbound;
        }
    }

    return toc;
}

template __device__ __host__ bool bbox_overlap(const Eigen::Matrix<float, 2, 1> &p, const Eigen::Matrix<float, 2, 1> &e0, const Eigen::Matrix<float, 2, 1> &e1,
                                               const Eigen::Matrix<float, 2, 1> &dp, const Eigen::Matrix<float, 2, 1> &de0, const Eigen::Matrix<float, 2, 1> &de1,
                                               float toc_upperbound);
template __device__ __host__ bool bbox_overlap(const Eigen::Matrix<double, 2, 1> &p, const Eigen::Matrix<double, 2, 1> &e0, const Eigen::Matrix<double, 2, 1> &e1,
                                               const Eigen::Matrix<double, 2, 1> &dp, const Eigen::Matrix<double, 2, 1> &de0, const Eigen::Matrix<double, 2, 1> &de1,
                                               double toc_upperbound);
template __device__ __host__ float narrow_phase_CCD(Eigen::Matrix<float, 2, 1> p, Eigen::Matrix<float, 2, 1> e0, Eigen::Matrix<float, 2, 1> e1,
                                                    Eigen::Matrix<float, 2, 1> dp, Eigen::Matrix<float, 2, 1> de0, Eigen::Matrix<float, 2, 1> de1,
                                                    float toc_upperbound);

template __device__ __host__ double narrow_phase_CCD(Eigen::Matrix<double, 2, 1> p, Eigen::Matrix<double, 2, 1> e0, Eigen::Matrix<double, 2, 1> e1,
                                                     Eigen::Matrix<double, 2, 1> dp, Eigen::Matrix<double, 2, 1> de0, Eigen::Matrix<double, 2, 1> de1,
                                                     double toc_upperbound);