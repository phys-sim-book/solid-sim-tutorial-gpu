#include "NeoHookeanEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <stdio.h>
#include "NeoHookean_auto.h"
#include "device_uti.h"

using namespace muda;

template <typename T, int dim>
struct NeoHookeanEnergy<T, dim>::Impl
{
    DeviceBuffer<T> device_x;
    DeviceBuffer<int> device_e;
    DeviceBuffer<T> device_vol;
    DeviceBuffer<Eigen::Matrix<T, 2, 2>> device_IB;
    T Mu, Lambda;
    DeviceBuffer<T> device_grad;
    DeviceTripletMatrix<T, 1> device_hess;
};

template <typename T, int dim>
NeoHookeanEnergy<T, dim>::NeoHookeanEnergy() = default;

template <typename T, int dim>
NeoHookeanEnergy<T, dim>::~NeoHookeanEnergy() = default;

template <typename T, int dim>
NeoHookeanEnergy<T, dim>::NeoHookeanEnergy(NeoHookeanEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
NeoHookeanEnergy<T, dim> &NeoHookeanEnergy<T, dim>::operator=(NeoHookeanEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
NeoHookeanEnergy<T, dim>::NeoHookeanEnergy(const NeoHookeanEnergy<T, dim> &rhs)
    : pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
NeoHookeanEnergy<T, dim>::NeoHookeanEnergy(const std::vector<T> &x, const std::vector<int> &e, T mu, T lam)
    : pimpl_{std::make_unique<Impl>()}
{
    pimpl_->device_x.copy_from(x);
    pimpl_->device_e.copy_from(e);
    pimpl_->device_vol.resize(e.size() / 3);
    pimpl_->device_IB.resize(e.size() / 3);
    pimpl_->Mu = mu;
    pimpl_->Lambda = lam;
    pimpl_->device_grad.resize(x.size());
    pimpl_->device_hess.resize_triplets(e.size() * 12);
    pimpl_->device_hess.reshape(x.size(), x.size());
    init_vol_IB();
}

template <typename T, int dim>
void NeoHookeanEnergy<T, dim>::init_vol_IB()
{
    ParallelFor(256).apply(pimpl_->device_e.size() / 3, [device_x = pimpl_->device_x.cviewer(), device_e = pimpl_->device_e.cviewer(), device_vol = pimpl_->device_vol.viewer(), device_IB = pimpl_->device_IB.viewer()] __device__(int i) mutable
                           {
        Eigen::Matrix<T, 2, 2> TB;
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                TB(k, j) = device_x(device_e(i * 3 + j+1) * 2 + k) - device_x(device_e(i * 3) * 2 + k);
            }
        }
        device_vol(i) = TB.determinant() / 2;
        device_IB(i) = TB.inverse(); })
        .wait();
}

template <typename T, int dim>
void NeoHookeanEnergy<T, dim>::update_x(const DeviceBuffer<T> &x)
{
    pimpl_->device_x.view().copy_from(x);
}

template <typename T, int dim>
T NeoHookeanEnergy<T, dim>::val()
{
    auto &device_x = pimpl_->device_x;
    auto &device_e = pimpl_->device_e;
    auto &device_vol = pimpl_->device_vol;
    auto &device_IB = pimpl_->device_IB;
    auto Mu = pimpl_->Mu;
    auto Lambda = pimpl_->Lambda;
    int N = device_e.size() / 3;
    DeviceBuffer<T> device_val(N);
    device_val.fill(0);

    ParallelFor(256).apply(N, [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_vol = device_vol.cviewer(), device_IB = device_IB.cviewer(), Mu, Lambda] __device__(int i) mutable
                           {
        T E;
          Eigen::Vector<T, 6> X;
        X << device_x(device_e(i * 3) * 2), device_x(device_e(i * 3) * 2 + 1), device_x(device_e(i * 3 + 1) * 2), device_x(device_e(i * 3 + 1) * 2 + 1), device_x(device_e(i * 3 + 2) * 2), device_x(device_e(i * 3 + 2) * 2 + 1);
         NeoHookeanEnergyVal(E,  Mu, Lambda,X,device_IB(i),device_vol(i));
         device_val(i) = E; })
        .wait();
    return devicesum(device_val);
}

template <typename T, int dim>
const DeviceBuffer<T> &NeoHookeanEnergy<T, dim>::grad()
{
    auto &device_x = pimpl_->device_x;
    auto &device_e = pimpl_->device_e;
    auto &device_vol = pimpl_->device_vol;
    auto &device_IB = pimpl_->device_IB;
    auto &device_grad = pimpl_->device_grad;
    auto Mu = pimpl_->Mu;
    auto Lambda = pimpl_->Lambda;
    int N = device_e.size() / 3;
    device_grad.fill(0);
    ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_vol = device_vol.cviewer(), device_IB = device_IB.cviewer(), device_grad = device_grad.viewer(), Mu, Lambda] __device__(int i) mutable
                           {
        Eigen::Vector<T, 6> G;
        Eigen::Vector<T, 6> X;
        X << device_x(device_e(i * 3) * 2), device_x(device_e(i * 3) * 2 + 1), device_x(device_e(i * 3 + 1) * 2), device_x(device_e(i * 3 + 1) * 2 + 1), device_x(device_e(i * 3 + 2) * 2), device_x(device_e(i * 3 + 2) * 2 + 1);
        NeoHookeanEnergyGradient(G, Mu, Lambda, X, device_IB(i), device_vol(i));
        for (int xI = 0; xI < 3; ++xI)
        {
            for (int dI = 0; dI < 2; ++dI)
            {
                atomic_add(&device_grad(device_e(i * 3 + xI) * dim + dI), G(xI * 2 + dI));
            }
        } })
        .wait();

    return device_grad;
}

template <typename T, int dim>
const DeviceTripletMatrix<T, 1> &NeoHookeanEnergy<T, dim>::hess()
{
    auto &device_x = pimpl_->device_x;
    auto &device_e = pimpl_->device_e;
    auto &device_vol = pimpl_->device_vol;
    auto &device_IB = pimpl_->device_IB;
    auto &device_hess = pimpl_->device_hess;
    auto device_hess_row_idx = device_hess.row_indices();
    auto device_hess_col_idx = device_hess.col_indices();
    auto device_hess_val = device_hess.values();
    auto Mu = pimpl_->Mu;
    auto Lambda = pimpl_->Lambda;
    int N = device_e.size() / 3;

    ParallelFor(256).apply(N, [device_x = device_x.cviewer(), device_e = device_e.cviewer(), device_vol = device_vol.cviewer(), device_IB = device_IB.cviewer(), device_hess_val = device_hess_val.viewer(), device_hess_row_idx = device_hess_row_idx.viewer(), device_hess_col_idx = device_hess_col_idx.viewer(), Mu, Lambda] __device__(int i) mutable
                           {    Eigen::Matrix<T, 6, 6> local_hess;
                            Eigen::Vector<T, 6> X;
                            X << device_x(device_e(i * 3) * 2), device_x(device_e(i * 3) * 2 + 1), device_x(device_e(i * 3 + 1) * 2), device_x(device_e(i * 3 + 1) * 2 + 1), device_x(device_e(i * 3 + 2) * 2), device_x(device_e(i * 3 + 2) * 2 + 1); 
                            NeoHookeanEnergyHessian(local_hess,  Mu, Lambda,X,device_IB(i),device_vol(i));  
        for (int xI = 0; xI < 3; ++xI)
        {
            for (int xJ = 0; xJ < 3; ++xJ)
            {
                for (int dI = 0; dI < 2; ++dI)
                {
                    for (int dJ = 0; dJ < 2; ++dJ)
                    {
                        int idx = (i * 9 + xI * 3 + xJ) * 4 + dI * 2 + dJ;
                        device_hess_row_idx(idx) = device_e(i * 3 + xI) * dim + dI;
                        device_hess_col_idx(idx) = device_e(i * 3 + xJ) * dim + dJ;
                        device_hess_val(idx) = local_hess(xI * 2 + dI, xJ * 2 + dJ);
                    }
                }
            }
        } })
        .wait();

    return device_hess;
}

template <typename T, int dim>
T NeoHookeanEnergy<T, dim>::init_step_size(const DeviceBuffer<T> &p)
{
    auto &device_x = pimpl_->device_x;
    auto &device_e = pimpl_->device_e;
    int N = device_e.size() / 3;
    DeviceBuffer<T> device_alpha(N);
    device_alpha.fill(1);

    ParallelFor(256).apply(N, [device_x = device_x.cviewer(), p = p.cviewer(), device_alpha = device_alpha.viewer(), device_e = device_e.cviewer()] __device__(int i) mutable
                           {
 Eigen::Matrix<T, dim, 1> x1, x2, x3, p1, p2, p3;

            for (int d = 0; d < dim; ++d)
            {
                x1(d) = device_x(device_e(i*3) * dim + d);
                x2(d) = device_x(device_e(i*3+ 1) * dim + d);
                x3(d) = device_x(device_e(i*3+ 2) * dim + d);

                p1(d) = p(device_e(i*3) * dim + d);
                p2(d) = p(device_e(i*3+1) * dim + d);
                p3(d) = p(device_e(i*3+2) * dim + d);
            }

            Eigen::Matrix<T, dim, 1> x21 = x2 - x1;
            Eigen::Matrix<T, dim, 1> x31 = x3 - x1;
            Eigen::Matrix<T, dim, 1> p21 = p2 - p1;
            Eigen::Matrix<T, dim, 1> p31 = p3 - p1;

            // Using determinant for 2D case instead of cross product
            T detT = x21.x() * x31.y() - x21.y() * x31.x();
            T a = (p21.x() * p31.y() - p21.y() * p31.x()) / detT;
            T b = ((x21.x() * p31.y() - x21.y() * p31.x()) + (p21.x() * x31.y() - p21.y() * x31.x())) / detT;
            T c = 0.9; // solve for alpha that first brings the new volume to 0.1x the old volume for slackness

        T critical_alpha = smallest_positive_real_root_quad(a, b, c);
        if (critical_alpha > 0)
        {
            device_alpha(i) = min(device_alpha(i), critical_alpha);
        } })
        .wait();

    return min_vector(device_alpha);
}

// // Helper functions

// template <typename T, int dim>
// __device__ void NeoHookeanEnergy<T, dim>::polar_svd(const Eigen::Matrix<T, dim, dim> &F, Eigen::Matrix<T, dim, dim> &U, Eigen::Matrix<T, dim, dim> &V, Eigen::Matrix<T, dim, 1> &s)
// {
//     Eigen::JacobiSVD<Eigen::Matrix<T, dim, dim>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
//     U = svd.matrixU();
//     V = svd.matrixV();
//     s = svd.singularValues();
//     if (U.determinant() < 0)
//     {
//         U.col(1) = -U.col(1);
//         s(1) = -s(1);
//     }
//     if (V.determinant() < 0)
//     {
//         V.col(1) = -V.col(1);
//         s(1) = -s(1);
//     }
// }

// template <typename T, int dim>
// __device__ Eigen::Matrix<T, dim, 1> NeoHookeanEnergy<T, dim>::dPsi_div_dsigma(const Eigen::Matrix<T, dim, 1> &s, T mu, T lam)
// {
//     T ln_sigma_prod = log(s.prod());
//     Eigen::Matrix<T, dim, 1> inv_s = s.cwiseInverse();
//     Eigen::Matrix<T, dim, 1> dPsi_dsigma = mu * (s - inv_s) + lam * inv_s * ln_sigma_prod;
//     return dPsi_dsigma;
// }

// template <typename T, int dim>
// __device__ Eigen::Matrix<T, dim, dim> NeoHookeanEnergy<T, dim>::d2Psi_div_dsigma2(const Eigen::Matrix<T, dim, 1> &s, T mu, T lam)
// {
//     T ln_sigma_prod = log(s.prod());
//     Eigen::Matrix<T, dim, 1> inv2_s = s.cwiseInverse().cwiseInverse();
//     Eigen::Matrix<T, dim, dim> d2Psi_dsigma2;
//     d2Psi_dsigma2(0, 0) = mu * (1 + inv2_s(0)) - lam * inv2_s(0) * (ln_sigma_prod - 1);
//     d2Psi_dsigma2(1, 1) = mu * (1 + inv2_s(1)) - lam * inv2_s(1) * (ln_sigma_prod - 1);
//     d2Psi_dsigma2(0, 1) = d2Psi_dsigma2(1, 0) = lam / s.prod();
//     return d2Psi_dsigma2;
// }

// template <typename T, int dim>
// __device__ T NeoHookeanEnergy<T, dim>::B_left_coef(const Eigen::Matrix<T, dim, 1> &s, T mu, T lam)
// {
//     T sigma_prod = s.prod();
//     return (mu + (mu - lam * log(sigma_prod)) / sigma_prod) / 2;
// }

// template <typename T, int dim>
// __device__ T NeoHookeanEnergy<T, dim>::Psi(const Eigen::Matrix<T, dim, dim> &F, T mu, T lam)
// {
//     T J = F.determinant();
//     T lnJ = log(J);
//     return mu / 2 * (F.squaredNorm() - dim) - mu * lnJ + lam / 2 * lnJ * lnJ;
// }

// template <typename T, int dim>
// __device__ Eigen::Matrix<T, dim, dim> NeoHookeanEnergy<T, dim>::dPsi_div_dF(const Eigen::Matrix<T, dim, dim> &F, T mu, T lam)
// {
//     Eigen::Matrix<T, dim, dim> FinvT = F.inverse().transpose();
//     return mu * (F - FinvT) + lam * log(F.determinant()) * FinvT;
// }

// template <typename T, int dim>
// __device__ Eigen::Matrix<T, 4, 4> NeoHookeanEnergy<T, dim>::d2Psi_div_dF2(const Eigen::Matrix<T, dim, dim> &F, T mu, T lam)
// {
//     Eigen::Matrix<T, dim, dim> U, V;
//     Eigen::Matrix<T, dim, 1> s;
//     polar_svd(F, U, V, s);

//     Eigen::Matrix<T, dim, dim> Psi_sigma_sigma;
//     make_PSD(d2Psi_div_dsigma2(s, mu, lam), Psi_sigma_sigma);

//     T B_left = B_left_coef(s, mu, lam);
//     Eigen::Matrix<T, dim, 1> Psi_sigma = dPsi_div_dsigma(s, mu, lam);
//     T B_right = (Psi_sigma.sum()) / (2 * std::max(s.sum(), T(1e-6)));
//     Eigen::Matrix<T, dim, dim> B;
//     make_PSD((Eigen::Matrix<T, dim, dim>() << B_left + B_right, B_left - B_right, B_left - B_right, B_left + B_right).finished(), B);

//     Eigen::Matrix<T, 4, 4> M;
//     M << Psi_sigma_sigma(0, 0), 0, 0, Psi_sigma_sigma(0, 1),
//         0, B(0, 0), B(0, 1), 0,
//         0, B(1, 0), B(1, 1), 0,
//         Psi_sigma_sigma(1, 0), 0, 0, Psi_sigma_sigma(1, 1);

//     Eigen::Matrix<T, 4, 4> dP_div_dF;
//     for (int j = 0; j < 2; ++j)
//     {
//         for (int i = 0; i < 2; ++i)
//         {
//             int ij = j * 2 + i;
//             for (int s = 0; s < 2; ++s)
//             {
//                 for (int r = 0; r < 2; ++r)
//                 {
//                     int rs = s * 2 + r;
//                     dP_div_dF(ij, rs) = M(0, 0) * U(i, 0) * V(0, j) * U(r, 0) * V(0, s) +
//                                         M(0, 3) * U(i, 0) * V(0, j) * U(r, 1) * V(1, s) +
//                                         M(1, 1) * U(i, 1) * V(0, j) * U(r, 1) * V(0, s) +
//                                         M(1, 2) * U(i, 1) * V(0, j) * U(r, 0) * V(1, s) +
//                                         M(2, 1) * U(i, 0) * V(1, j) * U(r, 1) * V(0, s) +
//                                         M(2, 2) * U(i, 0) * V(1, j) * U(r, 0) * V(1, s) +
//                                         M(3, 0) * U(i, 1) * V(1, j) * U(r, 0) * V(0, s) +
//                                         M(3, 3) * U(i, 1) * V(1, j) * U(r, 1) * V(1, s);
//                 }
//             }
//         }
//     }
//     return dP_div_dF;
// }
// template <typename T, int dim>
// __device__ Eigen::Matrix<T, 6, 1> NeoHookeanEnergy<T, dim>::dPsi_div_dx(const Eigen::Matrix<T, dim, dim> &P, const Eigen::Matrix<T, dim, dim> &IB)
// {
//     Eigen::Matrix<T, 6, 1> dPsi_dx;
//     dPsi_dx.setZero();

//     // Applying chain-rule, dPsi_div_dx = dPsi_div_dF * dF_div_dx
//     Eigen::Matrix<T, dim, dim> dF_dx0, dF_dx1, dF_dx2;
//     dF_dx0.col(0) = -IB.col(0);
//     dF_dx0.col(1) = -IB.col(1);
//     dF_dx1.col(0) = IB.col(0);
//     dF_dx1.col(1) = Eigen::Matrix<T, dim, 1>::Zero();
//     dF_dx2.col(0) = Eigen::Matrix<T, dim, 1>::Zero();
//     dF_dx2.col(1) = IB.col(1);

//     Eigen::Matrix<T, dim, dim> dPsi_div_dF = P;

//     Eigen::Matrix<T, dim, 1> dPsi_dx0 = dPsi_div_dF * dF_dx0;
//     Eigen::Matrix<T, dim, 1> dPsi_dx1 = dPsi_div_dF * dF_dx1;
//     Eigen::Matrix<T, dim, 1> dPsi_dx2 = dPsi_div_dF * dF_dx2;

//     dPsi_dx.template segment<dim>(0) = dPsi_dx0.colwise().sum();
//     dPsi_dx.template segment<dim>(dim) = dPsi_dx1.colwise().sum();
//     dPsi_dx.template segment<dim>(2 * dim) = dPsi_dx2.colwise().sum();

//     return dPsi_dx;
// }

// template <typename T, int dim>
// __device__ Eigen::Matrix<T, 6, 6> NeoHookeanEnergy<T, dim>::d2Psi_div_dx2(const Eigen::Matrix<T, 4, 4> &dP_div_dF, const Eigen::Matrix<T, dim, dim> &IB)
// {
//     Eigen::Matrix<T, 6, 6> result;
//     Eigen::Matrix<T, 6, 4> intermediate;

//     for (int colI = 0; colI < 4; ++colI)
//     {
//         intermediate(0, colI) = -dP_div_dF(0, colI) * IB(0, 0) - dP_div_dF(0, colI) * IB(1, 0) - dP_div_dF(2, colI) * IB(0, 1) - dP_div_dF(2, colI) * IB(1, 1);
//         intermediate(1, colI) = -dP_div_dF(1, colI) * IB(0, 0) - dP_div_dF(1, colI) * IB(1, 0) - dP_div_dF(3, colI) * IB(0, 1) - dP_div_dF(3, colI) * IB(1, 1);
//         intermediate(2, colI) = dP_div_dF(0, colI) * IB(0, 0) + dP_div_dF(2, colI) * IB(0, 1);
//         intermediate(3, colI) = dP_div_dF(1, colI) * IB(0, 0) + dP_div_dF(3, colI) * IB(0, 1);
//         intermediate(4, colI) = dP_div_dF(0, colI) * IB(1, 0) + dP_div_dF(2, colI) * IB(1, 1);
//         intermediate(5, colI) = dP_div_dF(1, colI) * IB(1, 0) + dP_div_dF(3, colI) * IB(1, 1);
//     }

//     for (int colI = 0; colI < 6; ++colI)
//     {
//         result(colI, 0) = -intermediate(colI, 0) * IB(0, 0) - intermediate(colI, 2) * IB(0, 1);
//         result(colI, 1) = -intermediate(colI, 1) * IB(0, 0) - intermediate(colI, 3) * IB(0, 1);
//         result(colI, 2) = intermediate(colI, 0) * IB(0, 0) + intermediate(colI, 2) * IB(0, 1);
//         result(colI, 3) = intermediate(colI, 1) * IB(0, 0) + intermediate(colI, 3) * IB(0, 1);
//         result(colI, 4) = intermediate(colI, 0) * IB(1, 0) + intermediate(colI, 2) * IB(1, 1);
//         result(colI, 5) = intermediate(colI, 1) * IB(1, 0) + intermediate(colI, 3) * IB(1, 1);
//     }

//     return result;
// }

// Explicit template instantiation
template class NeoHookeanEnergy<float, 2>;
template class NeoHookeanEnergy<float, 3>;
template class NeoHookeanEnergy<double, 2>;
template class NeoHookeanEnergy<double, 3>;
