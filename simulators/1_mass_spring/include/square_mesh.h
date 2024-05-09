#pragma once
#include <vector>
#include <memory>

// Function to generate mesh points and edges
void generate(float side_length, int n_seg, std::vector<float> &x, std::vector<int> &e);
