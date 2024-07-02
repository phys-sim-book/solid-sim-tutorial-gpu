#pragma once
#include <vector>
#include <memory>
#include <set>
// Function to generate mesh points and edges
template <typename T>
void generate(T side_length, int n_seg, std::vector<T> &x, std::vector<int> &e);

void find_boundary(const std::vector<int> &e, std::vector<int> &bp, std::vector<int> &be);