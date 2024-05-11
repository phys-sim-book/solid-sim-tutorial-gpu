#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include "InertialEnergy.h"
#include "MassSpringEnergy.h"
#include "square_mesh.h"
#include "uti.h"
#include <iostream>
template <typename T, int dim>
class MassSpringSimulator
{
public:
    MassSpringSimulator();
    ~MassSpringSimulator();
    MassSpringSimulator(T rho, T side_len, T initial_stretch, T K, T h, T tol, int n_seg);
    void run();
    void draw();
    void step_forward();
    void update_x(std::vector<T> new_x);
    void update_x_tilde(std::vector<T> new_x_tilde);
    void update_v(std::vector<T> new_v);
    T IP_val();
    std::vector<T> &IP_grad();
    SparseMatrix<T> &IP_hess();
    std::vector<T> search_direction();
    T screen_projection_x(T point);
    T screen_projection_y(T point);

private:
    int n_seg;
    T h, rho, side_len, initial_stretch, m, tol;
    int resolution = 900, scale = 200, offset = resolution / 2, radius = 5;
    std::vector<T> x, x_tilde, v, k, l2;
    std::vector<int> e;
    sf::RenderWindow window;
    InertialEnergy<T, dim> inertialenergy;
    MassSpringEnergy<T, dim> massspringenergy;
};
