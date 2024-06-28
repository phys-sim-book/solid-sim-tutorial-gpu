#pragma once

#include <vector>

template <typename T>
class SparseMatrix
{
public:
    SparseMatrix(int size);
    SparseMatrix();
    ~SparseMatrix();
    SparseMatrix(SparseMatrix<T> &&rhs);
    SparseMatrix<T> &operator=(SparseMatrix<T> &&rhs);
    SparseMatrix<T> &operator=(SparseMatrix<T> &rhs);
    SparseMatrix<T> &operator*(const T &a);
    SparseMatrix(const SparseMatrix<T> &rhs);
    void set_value(int row, int col, T val, int loc);
    void set_diagonal(T val);

    const std::vector<int> &get_row_buffer() const;
    const std::vector<int> &get_col_buffer() const;
    const std::vector<T> &get_val_buffer() const;
    std::vector<int> &set_row_buffer();
    std::vector<int> &set_col_buffer();
    std::vector<T> &set_val_buffer();
    SparseMatrix<T> &combine(const SparseMatrix<T> &other);

    int get_size() const;

private:
    int size;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<T> val;
};
