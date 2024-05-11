#pragma once
#include <Eigen/Dense>
#include "SparseMatrix.h"
template <typename T>
std::vector<T> add_vector(const std::vector<T> &a, const std::vector<T> &b, const T &factor1 = 1, const T &factor2 = 1);

template <typename T>
std::vector<T> mult_vector(const std::vector<T> &a, const T &b);

template <typename T>
T max_vector(const std::vector<T> &a);

template <typename T>
void search_dir(const std::vector<T> &grad, const SparseMatrix<T> &hess, std::vector<T> &dir);