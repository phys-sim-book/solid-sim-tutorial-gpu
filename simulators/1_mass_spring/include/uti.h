#pragma once
#include <Eigen/Dense>

template <typename T>
class SparseMatrix
{
public:
    SparseMatrix(int size);
    SparseMatrix();
    ~SparseMatrix();

    void set_value(int row, int col, T val, int loc);
    void set_diagonal(T val);

    std::vector<int> &get_row_buffer();
    std::vector<int> &get_col_buffer();
    std::vector<T> &get_val_buffer();
    SparseMatrix<T> &combine(const SparseMatrix<T> &other);

    int get_size();

    SparseMatrix<T> operator*(const T &a);

private:
    int size;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<T> val;
};

template <typename T>
std::vector<T> add_vector(const std::vector<T> &a, const std::vector<T> &b, const T &factor1 = 1, const T &factor2 = 1);

template <typename T>
std::vector<T> mult_vector(const std::vector<T> &a, const T &b);

template <typename T>
T max_vector(const std::vector<T> &a);

template <typename T>
void search_dir(const std::vector<T> &grad, SparseMatrix<T> &hess, std::vector<T> &dir);