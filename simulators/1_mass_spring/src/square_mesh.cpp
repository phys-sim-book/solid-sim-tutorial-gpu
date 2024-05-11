#include "square_mesh.h"
template <typename T>
void generate(T side_length, int n_seg, std::vector<T> &x, std::vector<int> &e)
{
    int dim = n_seg + 1;
    x.clear();
    x.reserve(dim * dim * 2); // Preallocate space for all nodes

    T step = side_length / n_seg;

    // Populate the coordinates
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x.push_back(-side_length / 2 + i * step);
            x.push_back(-side_length / 2 + j * step);
        }
    }

    // Clear any existing data in e and reserve space for edges
    e.clear();
    // Reserve space assuming maximum edge count (horizontal + vertical + 2*diagonal)
    e.reserve(2 * n_seg * (n_seg + 1) + 4 * n_seg * n_seg);

    // Horizontal edges
    for (int i = 0; i < n_seg; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            e.push_back(i * dim + j);
            e.push_back((i + 1) * dim + j);
        }
    }

    // Vertical edges
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < n_seg; j++)
        {
            e.push_back(i * dim + j);
            e.push_back(i * dim + j + 1);
        }
    }

    // Diagonal edges
    for (int i = 0; i < n_seg; i++)
    {
        for (int j = 0; j < n_seg; j++)
        {
            e.push_back(i * dim + j);
            e.push_back((i + 1) * dim + j + 1);
            e.push_back((i + 1) * dim + j);
            e.push_back(i * dim + j + 1);
        }
    }
}

template void generate<float>(float side_length, int n_seg, std::vector<float> &x, std::vector<int> &e);
template void generate<double>(double side_length, int n_seg, std::vector<double> &x, std::vector<int> &e);
template void generate<long double>(long double side_length, int n_seg, std::vector<long double> &x, std::vector<int> &e);
