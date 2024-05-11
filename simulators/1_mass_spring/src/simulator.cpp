#include "simulator.h"

template <typename T, int dim>
MassSpringSimulator<T, dim>::MassSpringSimulator() = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::~MassSpringSimulator() = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::MassSpringSimulator(T rho, T side_len, T initial_stretch, T K, T h_, T tol_, int n_seg) : tol(tol_), h(h_), window(sf::VideoMode(resolution, resolution), "MassSpringSimulator")
{
    generate(side_len, n_seg, x, e);
    v.resize(x.size(), 0);
    k = std::vector<T>(e.size(), K);
    l2 = std::vector<T>(e.size(), K);
    for (int i = 0; i < e.size() / 2; i++)
    {
        T diff = 0;
        int idx1 = e[2 * i], idx2 = e[2 * i + 1];
        for (int d = 0; d < dim; d++)
        {
            diff += (x[idx1 * dim + d] - x[idx2 * dim + d]) * (x[idx1 * dim + d] - x[idx2 * dim + d]);
        }
        l2[i] = diff;
    }
    m = rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1));
    // initial stretch
    int N = x.size() / dim;
    for (int i = 0; i < N; i++)
        x[i * dim + 0] *= initial_stretch;
    inertialenergy = InertialEnergy<T, dim>(N, m);
    massspringenergy = MassSpringEnergy<T, dim>(x, e, l2, k);
    update_x(x);
}

template <typename T, int dim>
void MassSpringSimulator<T, dim>::run()
{
    assert(dim == 2);
    bool running = true;
    while (running)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                running = false;
        }

        draw(); // Draw the current state

        // Update the simulation state
        step_forward();

        // Wait according to time step
        // sf::sleep(sf::milliseconds(static_cast<int>(h * 1000)));
    }

    window.close();
}

template <typename T, int dim>
void MassSpringSimulator<T, dim>::step_forward()
{
    std::vector<T> x_tilde(x.size()); // Predictive position
    update_x_tilde(add_vector<T>(x, v, 1, h));
    std::vector<T> x_n = x; // Copy current positions to x_n
    int iter = 0;
    T E_last = IP_val();
    std::vector<T> p = search_direction();
    T residual = max_vector(p) / h;
    // printf("Initial residual %f\n", residual);
    while (residual > tol)
    {
        // std::cout << "Iteration " << iter << ":\n";
        // std::cout << "residual = " << residual << "\n";

        // Line search
        T alpha = 1;
        std::vector<T> x0 = x;
        update_x(add_vector<T>(x, p, 1.0, alpha));
        while (IP_val() > E_last)
        {
            alpha /= 2;
            update_x(add_vector<T>(x0, p, 1.0, alpha));
        }
        // std::cout << "step size = " << alpha << "\n";
        E_last = IP_val();
        p = search_direction();
        residual = max_vector(p) / h;
        iter += 1;
    }
    update_v(add_vector<T>(x, x_n, 1 / h, -1 / h));
}
template <typename T, int dim>
T MassSpringSimulator<T, dim>::screen_projection_x(T point)
{
    return offset + scale * point;
}
template <typename T, int dim>
T MassSpringSimulator<T, dim>::screen_projection_y(T point)
{
    return resolution - (offset + scale * point);
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::update_x(std::vector<T> new_x)
{
    inertialenergy.update_x(new_x);
    massspringenergy.update_x(new_x);
    x = new_x;
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::update_x_tilde(std::vector<T> new_x_tilde)
{
    inertialenergy.update_x_tilde(new_x_tilde);
    x_tilde = new_x_tilde;
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::update_v(std::vector<T> new_v)
{
    v = new_v;
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::draw()
{
    window.clear(sf::Color::White); // Clear the previous frame

    // Draw springs as lines
    for (int i = 0; i < e.size() / 2; ++i)
    {
        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(screen_projection_x(x[e[i * 2] * dim]), screen_projection_y(x[e[i * 2] * dim + 1])), sf::Color::Blue),
            sf::Vertex(sf::Vector2f(screen_projection_x(x[e[i * 2 + 1] * dim]), screen_projection_y(x[e[i * 2 + 1] * dim + 1])), sf::Color::Blue)};
        window.draw(line, 2, sf::Lines);
    }

    // Draw masses as circles
    for (int i = 0; i < x.size() / dim; ++i)
    {
        sf::CircleShape circle(radius); // Set a fixed radius for each mass
        circle.setFillColor(sf::Color::Red);
        circle.setPosition(screen_projection_x(x[i * dim]) - radius, screen_projection_y(x[i * dim + 1]) - radius); // Center the circle on the mass
        window.draw(circle);
    }

    window.display(); // Display the rendered frame
}

template <typename T, int dim>
T MassSpringSimulator<T, dim>::IP_val()
{

    return inertialenergy.val() + massspringenergy.val() * h * h;
}

template <typename T, int dim>
std::vector<T> MassSpringSimulator<T, dim>::IP_grad()
{

    return add_vector<T>(inertialenergy.grad(), massspringenergy.grad(), 1.0, h * h);
}

template <typename T, int dim>
SparseMatrix<T> MassSpringSimulator<T, dim>::IP_hess()
{
    SparseMatrix<T> inertial_hess = inertialenergy.hess();
    SparseMatrix<T> massspring_hess = massspringenergy.hess();
    massspring_hess = massspring_hess * (h * h);
    inertial_hess.combine(massspring_hess);
    return inertial_hess;
}
template <typename T, int dim>
std::vector<T> MassSpringSimulator<T, dim>::search_direction()
{
    std::vector<T> dir;
    search_dir(IP_grad(), IP_hess(), dir);
    return dir;
}
