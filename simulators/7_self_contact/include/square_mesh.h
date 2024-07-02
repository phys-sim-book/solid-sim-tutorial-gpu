#pragma once
#include <vector>
#include <memory>

// Function to generate mesh points and edges
template <typename T>
void generate(T side_length, int n_seg, std::vector<T> &x, std::vector<int> &e);
