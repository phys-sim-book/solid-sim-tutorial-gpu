#include "square_mesh.h"
template <typename T>
void generate(T side_length, int n_seg, std::vector<T> &x, std::vector<int> &e)
{
    int dim = n_seg + 1;
    int num_nodes = dim * dim;
    x.resize(num_nodes * 2);

    T step = side_length / n_seg;

    // Populate the coordinates
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            x[(i * dim + j) * 2] = -side_length / 2 + i * step;
            x[(i * dim + j) * 2 + 1] = -side_length / 2 + j * step;
        }
    }

    // Clear and reserve space for edges
    e.clear();
    e.reserve(2 * n_seg * n_seg * 3);

    // Triangulate the grid
    for (int i = 0; i < n_seg; ++i)
    {
        for (int j = 0; j < n_seg; ++j)
        {
            if ((i % 2) ^ (j % 2))
            {
                e.push_back(i * dim + j);
                e.push_back((i + 1) * dim + j);
                e.push_back(i * dim + j + 1);

                e.push_back((i + 1) * dim + j);
                e.push_back((i + 1) * dim + j + 1);
                e.push_back(i * dim + j + 1);
            }
            else
            {
                e.push_back(i * dim + j);
                e.push_back((i + 1) * dim + j);
                e.push_back((i + 1) * dim + j + 1);

                e.push_back(i * dim + j);
                e.push_back((i + 1) * dim + j + 1);
                e.push_back(i * dim + j + 1);
            }
        }
    }
}

template void generate<float>(float side_length, int n_seg, std::vector<float> &x, std::vector<int> &e);
template void generate<double>(double side_length, int n_seg, std::vector<double> &x, std::vector<int> &e);
template void generate<long double>(long double side_length, int n_seg, std::vector<long double> &x, std::vector<int> &e);

void find_boundary(const std::vector<int> &e, std::vector<int> &bp, std::vector<int> &be)
{
    // index all half-edges for fast query
    std::set<std::pair<int, int>> edge_set;
    for (size_t i = 0; i < e.size()/3; i += 3)
    {
        for (int j = 0; j < 3; ++j)
        {
            edge_set.insert({e[i * 3 + j], e[i * 3 + (j + 1) % 3]});
        }
    }

    // find boundary points and edges
    std::set<int> bp_set;
    for (const auto &eI : edge_set)
    {
        if (edge_set.find({eI.second, eI.first}) == edge_set.end())
        {
            // if the inverse edge of a half-edge does not exist,
            // then it is a boundary edge
            be.push_back(eI.first);
            be.push_back(eI.second);
            bp_set.insert(eI.first);
            bp_set.insert(eI.second);
        }
    }

    // Convert set to vector
    bp.assign(bp_set.begin(), bp_set.end());
}
