#pragma once

#include <memory>
#include <Eigen/Dense>

template <typename T, int dim>
class MassSpringEnergy
{
public:
    MassSpringEnergy(const std::vector<T> &x, const std::vector<int> &e, const std::vector<T> &l2, const std::vector<T> &k);
    ~MassSpringEnergy();
    MassSpringEnergy(MassSpringEnergy &&rhs);
    MassSpringEnergy(const MassSpringEnergy &rhs);
    MassSpringEnergy &operator=(MassSpringEnergy &&rhs);
    MassSpringEnergy &operator=(const MassSpringEnergy &rhs);

    void update_x(const std::vector<T> &x);
    void update_e(const std::vector<int> &e);
    void update_l2(const std::vector<T> &l2);
    void update_k(const std::vector<T> &k);
    T val();                // Calculate the value of the energy
    std::vector<T> &grad(); // Calculate the gradient of the energy
    std::vector<T> &hess(); // Calculate the Hessian matrix of the energy

private:
    // The implementation details of the VecAdder class are placed in the implementation class declared here.
    struct Impl;
    // The private pointer to the implementation class Impl
    std::unique_ptr<Impl> pimpl_;
};